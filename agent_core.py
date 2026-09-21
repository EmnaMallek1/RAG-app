"""
agent_core.py
--------------
Agentic layer on top of rag_core.py.

Instead of a fixed pipeline (retrieve -> rerank -> generate -> verify) that
always runs the same steps in the same order, the LLM itself decides:
  - whether it needs to search the papers at all,
  - what query to search with (and whether to filter to a specific paper),
  - whether one search was enough, or it needs to reformulate and search again,
  - when it has enough context to write the final answer.

Only ONE tool is exposed: search_papers, which wraps rag_core's
retrieve_and_rerank(). There is no web access and no outside knowledge --
the model is still restricted to the 26 indexed papers, same guarantee as
before. Your existing rag_core.py is not modified; this module only adds a
new way of calling it.

Drop-in usage (same signature as rag_core.generate_answer):

    import agent_core
    answer, sources = agent_core.generate_answer_agentic(
        prompt, vectorstore, reranker, GROQ_API_KEYS, top_k=top_k, verify=verify_answers,
    )
"""

import re
import unicodedata

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

import rag_core as rc

langfuse = get_client()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from env

# A citation bracket "[...]" but NOT "\[...\]" -- the latter is LaTeX display-math
# delimiter syntax some models use instead of "$$...$$". Without the negative
# lookbehind/lookahead here, a stray "\[ ... \]" equation gets misread as a
# citation (its content doesn't match any paper, but it still steals a number
# slot from the real citations) -- this is exactly what happened when the model
# wrote "\[ \Delta W = BA \]" instead of "$$\Delta W = BA$$".
_CITATION_BRACKET_RE = re.compile(r"(?<!\\)\[([^\]]+)(?<!\\)\]")

MAX_ITERATIONS = 5          # hard safety cap so the model can't loop forever
DEFAULT_TOP_K = rc.FINAL_TOP_K

# =========================================================
# Typography-proof citation matching
# ---------------------------------------------------------
# Reasoning models like openai/gpt-oss-120b tend to substitute "nicer-looking"
# Unicode punctuation when they type -- e.g. a non-breaking hyphen (U+2011)
# instead of a plain ASCII "-" in "Low-Rank". A citation like
# "[LoRA: Low‑Rank Adaptation...]" (special hyphen) then fails an exact string
# match against rc.PAPER_TITLES' "LoRA: Low-Rank Adaptation..." (plain hyphen),
# and _build_sources silently drops that paper from the source list -- even
# though the citation is visibly right there in the text. We normalize both
# sides before comparing, and rewrite citations to the exact canonical name so
# everything downstream (source list, numbering) matches consistently.
# =========================================================
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"  # hyphen/dash variants


def _normalize_for_matching(text):
    text = unicodedata.normalize("NFKC", text)
    for ch in _DASH_CHARS:
        text = text.replace(ch, "-")
    text = text.replace("\u00a0", " ")  # non-breaking space -> normal space
    text = re.sub(r"\s+", " ", text).strip()
    return text


_PAPER_NAME_LOOKUP = {_normalize_for_matching(name): name for name in rc.PAPER_TITLES.values()}


_BARE_NUMBER_CITATION = re.compile(r"^\d+(\s*,\s*\d+)*$")


def _canonicalize_citations(text):
    """Rewrites every "[...]" citation bracket so its paper name(s) exactly match
    rc.PAPER_TITLES, regardless of typographic variants the model introduced.

    Also strips bare numeric brackets like "[1]" or "[2]" that match no known
    paper: these are almost always citation numbers LEAKED from the original
    paper's own bibliography inside a retrieved excerpt (e.g. "...as shown in
    [1]...") that the model accidentally carried over while paraphrasing --
    not an actual citation it intended to make. Left in place, they'd steal a
    number slot from the real citations and throw off the source list.

    Any OTHER unmatched (non-numeric) bracket content is kept as-is, so a
    genuine mismatch stays visible rather than being silently hidden."""
    pattern = _CITATION_BRACKET_RE

    def _replace(match):
        names = [n.strip() for n in match.group(1).split(";") if n.strip()]
        canonical_names = []
        for name in names:
            canonical = _PAPER_NAME_LOOKUP.get(_normalize_for_matching(name))
            if canonical:
                if canonical not in canonical_names:
                    canonical_names.append(canonical)
            elif _BARE_NUMBER_CITATION.match(name):
                continue  # leaked bibliography number -- drop it silently
            elif name not in canonical_names:
                canonical_names.append(name)

        if not canonical_names:
            return ""  # the whole bracket was noise -- remove it entirely

        return "[" + "; ".join(canonical_names) + "]"

    return pattern.sub(_replace, text)

# =========================================================
# What the model is told: its instructions + its one tool
# =========================================================
AGENT_SYSTEM_PROMPT = """You are a research assistant answering questions about machine learning papers.

You have access to ONE tool:

- search_papers(query: str, paper_filter: str = None): searches the indexed database of 26 ML
  papers and returns the most relevant excerpts. Use `paper_filter` (the EXACT paper name) only
  when the question explicitly names one or more specific papers you want guaranteed to be
  represented (e.g. a "LoRA vs QLoRA" comparison -- call it once per named paper, each with its
  own paper_filter, in addition to a general call if useful).

Rules:
1. Call search_papers whenever you need facts to answer -- never answer from your own outside
   knowledge. Ground every claim in what the tool returns.
2. If the first search's results are thin, irrelevant, or don't cover part of the question,
   reformulate the query and call search_papers again. You may call it multiple times.
3. Don't call search_papers more than 4 times for a single question. If after that you still
   don't have enough, say explicitly that the papers don't cover it -- don't guess.
4. Once you have enough retrieved context, STOP calling tools and write the final answer
   directly as plain text (no tool call).
5. Citation format -- follow this EXACTLY:
   - Cite ONLY as `[Exact Paper Name]`, using plain ASCII square brackets `[` and `]`.
   - Copy the paper name exactly as it appears after "Source N —" in the tool results.
   - NEVER cite as "Source 1", "Source N", a bare number, or with full-width/Japanese brackets
     (`【` `】`). The "Source N" labels in the tool results are for YOUR reference only, to tell
     excerpts apart -- they must NEVER appear in your final answer.
   - Correct: "...matrices, drastically cutting trainable parameters [LoRA: Low-Rank Adaptation
     of Large Language Models]."
   - Wrong: "...matrices, drastically cutting trainable parameters [Source 3]." or
     "...matrices, drastically cutting trainable parameters【Source 3】."
   - The retrieved excerpts may themselves contain numeric references to THEIR OWN
     bibliography (e.g. "...as shown in [1]..."). These are NOT citations to your
     source papers -- never copy a bare number like [1] or [2] into your answer.
6. Answer in the same language as the user's question.
7. Use LaTeX ONLY with dollar-sign delimiters: `$...$` for inline math, `$$...$$` for display/
   standalone equations. NEVER use `\\[...\\]` or `\\(...\\)` as math delimiters -- Streamlit does
   not render those, and they will visibly break the citation numbering.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Semantic search over the indexed database of 26 ML papers "
                "(Transformer, BERT, GPT, LoRA, diffusion, RAG, RL, etc.). "
                "Returns the most relevant excerpts for the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, phrased for semantic search.",
                    },
                    "paper_filter": {
                        "type": ["string", "null"],
                        "description": (
                            "The exact name of one paper to restrict the search to, or null "
                            "if this search should not be restricted to a specific paper. "
                            "Use a paper name when comparing specific named papers."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# =========================================================
# The tool's real implementation -- wraps your existing pipeline
# =========================================================
def _format_context_with_global_numbers(reranked_results, start_index, source_map):
    """Like rag_core.format_context, but numbers sources CONTINUOUSLY across every
    tool call in the loop (1, 2, 3... never resetting), and records number -> paper
    name in `source_map` so any "[Source N]" the model writes despite instructions
    can still be resolved back to a real paper name afterward."""
    blocks = []
    idx = start_index
    for doc, score in reranked_results:
        paper_name = doc.metadata.get("paper_name", doc.metadata.get("title", "unknown"))
        source_map[idx] = paper_name
        blocks.append(f"[Source {idx} — {paper_name}]\n{doc.page_content}")
        idx += 1
    return "\n\n---\n\n".join(blocks), idx


def _search_papers_tool(query, vectorstore, reranker, source_map, next_index,
                         paper_filter=None, top_k=DEFAULT_TOP_K):
    """Runs your actual RAG retrieval. Returns (text_for_the_model, reranked_results, next_index)."""
    if paper_filter:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k, "filter": {"paper_name": paper_filter}}
        )
        docs = retriever.invoke(rc.BGE_QUERY_INSTRUCTION + query)
        if not docs:
            return f"No results found for paper '{paper_filter}'.", [], next_index
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
    else:
        reranked = rc.retrieve_and_rerank(query, vectorstore, reranker, top_k=top_k)

    if not reranked:
        return "No relevant results found in the papers database.", [], next_index

    context_text, next_index = _format_context_with_global_numbers(reranked, next_index, source_map)
    return context_text, reranked, next_index


def _normalize_source_citations(text, source_map):
    """Safety net: even with the prompt instructions, a model can still slip and
    write "[Source 3]" or the Japanese-bracket "【Source 3】" instead of the paper
    name. This rewrites any such citation back into "[Paper Name]" using the
    number -> paper name map built while formatting tool results. Leaves already-
    correct "[Paper Name]" citations untouched."""
    pattern = re.compile(r"[\[【]\s*Source\s+(\d+)[^\]】]*[\]】]")

    def _replace(match):
        number = int(match.group(1))
        paper_name = source_map.get(number)
        return f"[{paper_name}]" if paper_name else match.group(0)

    return pattern.sub(_replace, text)


# =========================================================
# Groq multi-key fallback for a single agent step
# (same pattern as rag_core._invoke_with_fallback, but returns the
# full response object so we can read .tool_calls, not just .content)
# =========================================================
def _is_rate_limit_error(e):
    msg = str(e)
    return "rate_limit_exceeded" in msg or "429" in msg or "rate limit" in msg.lower()


def _invoke_agent_step(messages, api_keys, log=print, callbacks=None):
    if not api_keys:
        raise ValueError("No GROQ_API_KEY configured.")

    invoke_config = {"callbacks": callbacks} if callbacks else {}

    last_error = None
    for i, key in enumerate(api_keys):
        try:
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0,
                max_tokens=2048,
                reasoning_effort="low",
                reasoning_format="hidden",
                groq_api_key=key,
            ).bind_tools(TOOLS)
            return llm.invoke(messages, config=invoke_config)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                log(f"Key #{i + 1} rate-limited -- trying next key...")
                continue
            raise
    raise last_error


def _dedupe_results(reranked_results):
    seen = set()
    deduped = []
    for doc, score in reranked_results:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            deduped.append((doc, score))
    return deduped


# =========================================================
# The agent loop itself
# =========================================================
def generate_answer_agentic(query, vectorstore, reranker, api_keys, top_k=DEFAULT_TOP_K,
                             verify=True, log=print):
    """Agentic replacement for rag_core.generate_answer.
    Same return signature: (answer_text, list_of_sources).

    The LLM decides for itself how many times to call search_papers and with
    what queries -- instead of a single fixed retrieval pass."""

    all_reranked_results = []  # everything retrieved across every tool call, for sources/citations
    source_map = {}            # running "Source N" -> paper name, across ALL tool calls
    next_source_index = 1       # never resets between tool calls, unlike rag_core.format_context

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    final_text = None

    # Everything inside this span (every LLM call, every tool call) shows up as ONE
    # trace in Langfuse for this question, instead of scattered separate traces.
    with langfuse.start_as_current_observation(
        as_type="span", name="rag-agent-question", input=query
    ) as root_span:
        langfuse_handler = CallbackHandler()

        for iteration in range(MAX_ITERATIONS):
            response = _invoke_agent_step(messages, api_keys, log=log, callbacks=[langfuse_handler])
            messages.append(response)

            if not response.tool_calls:
                final_text = response.content
                log(f"Model answered directly after {iteration} search(es).")
                break

            for call in response.tool_calls:
                args = call["args"]
                log(f"Tool call: search_papers(query={args.get('query')!r}, "
                    f"paper_filter={args.get('paper_filter')!r})")

                context_text, reranked, next_source_index = _search_papers_tool(
                    args.get("query", query),
                    vectorstore, reranker,
                    source_map, next_source_index,
                    paper_filter=args.get("paper_filter"),
                    top_k=top_k,
                )
                all_reranked_results.extend(reranked)

                messages.append(ToolMessage(content=context_text, tool_call_id=call["id"]))
        else:
            # Loop hit MAX_ITERATIONS without the model stopping itself.
            log("Max iterations reached -- forcing a final answer.")
            messages.append(HumanMessage(
                content="You've reached the search limit. Answer now with what you have, "
                        "or say explicitly that the papers don't cover this."
            ))
            response = _invoke_agent_step(messages, api_keys, log=log, callbacks=[langfuse_handler])
            final_text = response.content

        root_span.update(output=final_text)

    langfuse.flush()  # Streamlit requests are short-lived -- make sure data is actually sent

    if final_text is None:
        final_text = "I could not produce an answer."

    log("=== DEBUG: raw draft from the agent loop (before any post-processing) ===")
    log(final_text)
    log("=== END raw draft ===")

    # Safety net: rewrite any "[Source N]" / "【Source N】" the model slipped in,
    # despite instructions, back into "[Paper Name]" -- BEFORE verification, so
    # the fact-checker (and everything downstream) sees clean, correct citations.
    final_text = _normalize_source_citations(final_text, source_map)
    # Then fix up typography (special hyphens, etc.) so citation text exactly
    # matches the real paper names -- see _canonicalize_citations docstring.
    final_text = _canonicalize_citations(final_text)

    log("=== DEBUG: after normalize + canonicalize, before verification ===")
    log(final_text)
    log("=== END pre-verification ===")

    deduped_results = _dedupe_results(all_reranked_results)

    # Reuse your existing verification pass unchanged -- it doesn't care
    # whether the context came from one fixed retrieval or several agent
    # tool calls, it just fact-checks the draft against whatever context text
    # it's given.
    if verify and deduped_results:
        context_for_verification = rc.format_context(deduped_results)
        final_text = rc.verify_and_correct_answer(final_text, context_for_verification, api_keys, log=log)

        log("=== DEBUG: after verification pass ===")
        log(final_text)
        log("=== END post-verification ===")

        # rc.format_context numbers 1..k for THIS call specifically (it doesn't
        # know about the agent loop's running numbers), so if the fact-checker
        # reintroduces a "[Source N]" citation, it refers to THIS numbering --
        # normalize again with a map built the same way.
        verification_map = {
            i + 1: doc.metadata.get("paper_name", doc.metadata.get("title", "unknown"))
            for i, (doc, _score) in enumerate(deduped_results)
        }
        final_text = _normalize_source_citations(final_text, verification_map)
        final_text = _canonicalize_citations(final_text)

    final_text = rc._merge_adjacent_citations(final_text)

    # Same citation -> numbered-source logic as rag_core.generate_answer,
    # just reused here since the output format is identical.
    ordered_cited_names = []
    for raw_group in _CITATION_BRACKET_RE.findall(final_text):
        for name in raw_group.split(";"):
            name = name.strip()
            if name and name not in ordered_cited_names:
                ordered_cited_names.append(name)

    sources = rc._build_sources(deduped_results, allowed_paper_names=set(ordered_cited_names) or None)
    if not sources:
        sources = rc._build_sources(deduped_results, allowed_paper_names=None)

    paper_to_number = {name: i + 1 for i, name in enumerate(ordered_cited_names)}

    def _numbered(match):
        names = [n.strip() for n in match.group(1).split(";") if n.strip()]
        numbers = [str(paper_to_number.get(n, "?")) for n in names]
        return "[" + ",".join(numbers) + "]"

    final_text = _CITATION_BRACKET_RE.sub(_numbered, final_text)

    for s in sources:
        s["number"] = paper_to_number.get(s["paper_name"])
    sources.sort(key=lambda s: s["number"] if s["number"] is not None else 999)

    return final_text, sources


# =========================================================
# Standalone test -- run this file directly to watch the loop decide
# for itself, without touching Streamlit at all.
# =========================================================
if __name__ == "__main__":
    import os

    key_names = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5"]
    keys = [os.environ.get(name) for name in key_names]
    keys = [k for k in keys if k]
    print(f"{len(keys)} Groq API key(s) found in environment.")

    def log(msg):
        print(f"  [log] {msg}")

    print("Loading pipeline (first run may take a while)...")
    rc.ensure_prebuilt_assets(log=log)
    merged = rc.load_and_merge_documents(log=log)
    chunks = rc.build_chunks(merged, log=log)
    vectorstore = rc.build_vectorstore(chunks, log=log)
    reranker = rc.load_reranker(log=log)

    question = "Compare LoRA and QLoRA -- how do they differ?"
    print(f"\nQuestion: {question}\n")

    answer, sources = generate_answer_agentic(question, vectorstore, reranker, keys, log=log)

    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for s in sources:
        print(f"[{s['number']}] {s['paper_name']} (pages: {s.get('pages_display')})")