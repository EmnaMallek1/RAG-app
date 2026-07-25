"""
rag_core.py
------------
All the "heavy" RAG pipeline logic (PDF loading, cleaning, semantic
chunking, embeddings, Chroma vector store, reranking and generation)
lives here as reusable functions.

app.py (Streamlit) imports this module and only handles the UI.
"""

import os
import re
import pickle
from collections import defaultdict, Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# =========================================================
# Config
# =========================================================
PICKLE_PATH = "all_documents.pkl"
SEMANTIC_CHUNKS_PATH = "semantic_chunks.pkl"
VECTORSTORE_DIR = "chroma_db"

RETRIEVAL_POOL_SIZE = 20
FINAL_TOP_K = 5
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 80

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

URLS = [
    "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need
    "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
    "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
    "https://arxiv.org/pdf/2303.08774.pdf",  # GPT-4
    "https://arxiv.org/pdf/2302.13971.pdf",  # LLaMA
    "https://arxiv.org/pdf/1910.10683.pdf",  # T5
    "https://arxiv.org/pdf/2204.02311.pdf",  # PaLM
    "https://arxiv.org/pdf/2106.09685.pdf",  # LoRA
    "https://arxiv.org/pdf/2305.14314.pdf",  # QLoRA
    "https://arxiv.org/pdf/2201.11903.pdf",  # Chain-of-Thought
    "https://arxiv.org/pdf/2203.02155.pdf",  # InstructGPT
    "https://arxiv.org/pdf/1512.03385.pdf",  # ResNet
    "https://arxiv.org/pdf/2010.11929.pdf",  # ViT
    "https://arxiv.org/pdf/2103.00020.pdf",  # CLIP
    "https://arxiv.org/pdf/1406.2661.pdf",   # GAN
    "https://arxiv.org/pdf/1312.6114.pdf",   # VAE
    "https://arxiv.org/pdf/2006.11239.pdf",  # DDPM
    "https://arxiv.org/pdf/2112.10752.pdf",  # Stable Diffusion
    "https://arxiv.org/pdf/2005.11401.pdf",  # RAG
    "https://arxiv.org/pdf/2004.04906.pdf",  # DPR
    "https://arxiv.org/pdf/1712.01815.pdf",  # AlphaGo
    "https://arxiv.org/pdf/1707.06347.pdf",  # PPO
    "https://arxiv.org/pdf/1312.5602.pdf",   # DQN
    "https://arxiv.org/pdf/1301.3781.pdf",   # Word2Vec
    "https://arxiv.org/pdf/1207.0580.pdf",   # Dropout
    "https://arxiv.org/pdf/1412.6980.pdf",   # Adam
]

PAPER_TITLES = {
    "1706.03762.pdf": "Attention Is All You Need",
    "1810.04805.pdf": "BERT: Pre-training of Deep Bidirectional Transformers",
    "2005.14165.pdf": "GPT-3: Language Models are Few-Shot Learners",
    "2303.08774.pdf": "GPT-4 Technical Report",
    "2302.13971.pdf": "LLaMA: Open and Efficient Foundation Language Models",
    "1910.10683.pdf": "T5: Exploring the Limits of Transfer Learning",
    "2204.02311.pdf": "PaLM: Scaling Language Modeling with Pathways",
    "2106.09685.pdf": "LoRA: Low-Rank Adaptation of Large Language Models",
    "2305.14314.pdf": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "2201.11903.pdf": "Chain-of-Thought Prompting",
    "2203.02155.pdf": "InstructGPT: Training with Human Feedback",
    "1512.03385.pdf": "ResNet: Deep Residual Learning for Image Recognition",
    "2010.11929.pdf": "Vision Transformer (ViT)",
    "2103.00020.pdf": "CLIP: Learning Transferable Visual Models",
    "1406.2661.pdf": "GAN: Generative Adversarial Networks",
    "1312.6114.pdf": "VAE: Auto-Encoding Variational Bayes",
    "2006.11239.pdf": "DDPM: Denoising Diffusion Probabilistic Models",
    "2112.10752.pdf": "Stable Diffusion: High-Resolution Latent Diffusion Models",
    "2005.11401.pdf": "RAG: Retrieval-Augmented Generation",
    "2004.04906.pdf": "DPR: Dense Passage Retrieval",
    "1712.01815.pdf": "AlphaZero (AlphaGo family)",
    "1707.06347.pdf": "PPO: Proximal Policy Optimization",
    "1312.5602.pdf": "DQN: Playing Atari with Deep Reinforcement Learning",
    "1301.3781.pdf": "Word2Vec: Efficient Estimation of Word Representations",
    "1207.0580.pdf": "Dropout: Preventing Overfitting",
    "1412.6980.pdf": "Adam: A Method for Stochastic Optimization",
}

# =========================================================
# Text cleaning (references / boilerplate / bibliography lines)
# =========================================================
REFERENCE_SECTION_PATTERN = re.compile(
    r"\n\s*(References|Bibliography|Acknowledg(e)?ments?)\s*\n", re.IGNORECASE
)

COPYRIGHT_NOTICE_PATTERNS = [
    re.compile(r"Provided proper attribution.*?scholarly works\.", re.IGNORECASE | re.DOTALL),
    re.compile(r"Preprint\.\s*Under review\.", re.IGNORECASE),
    re.compile(r"Work (performed|done) while at [^.\n]+\.", re.IGNORECASE),
    re.compile(r"Equal contribution\.?", re.IGNORECASE),
    re.compile(r"Correspondence to:.*?(?=\n|$)", re.IGNORECASE),
]

BIBLIO_LINE_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z\-]+,?\s+[A-Z]\.?\s?[A-Za-z\.]*.*\b(19|20)\d{2}\b.*$"
)


def strip_references_section(text):
    match = REFERENCE_SECTION_PATTERN.search(text)
    if match:
        return text[:match.start()]
    return text


def strip_boilerplate(text):
    for pattern in COPYRIGHT_NOTICE_PATTERNS:
        text = pattern.sub("", text)
    return text


def strip_biblio_like_lines(text):
    lines = text.split("\n")
    kept = [ln for ln in lines if not BIBLIO_LINE_PATTERN.match(ln.strip())]
    return "\n".join(kept)


def clean_text(text):
    text = strip_references_section(text)
    text = strip_boilerplate(text)
    text = strip_biblio_like_lines(text)
    text = re.sub(r"[∗†‡§¶]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_low_quality_chunk(text):
    stripped = text.strip()
    if len(stripped) < MIN_CHUNK_CHARS:
        return True

    words = re.findall(r"[A-Za-z]{2,}", stripped)
    if len(words) < 8:
        return True

    alpha_chars = sum(c.isalpha() for c in stripped)
    if alpha_chars / max(len(stripped), 1) < 0.5:
        return True

    if not re.search(r"[.!?]\s", stripped) and len(stripped) < 400:
        return True

    return False


# =========================================================
# Step 1-2: loading + merging pages into full documents
# =========================================================
def load_and_merge_documents(log=print):
    """Loads the PDFs (with pickle cache), merges them per paper and
    cleans the text. Returns a list of Document (one per paper)."""

    if os.path.exists(PICKLE_PATH):
        log("Cache found — loading from disk (no download needed).")
        with open(PICKLE_PATH, "rb") as f:
            all_documents = pickle.load(f)
        log(f"{len(all_documents)} pages loaded from cache.")
    else:
        log("No cache found — downloading and processing the PDFs...")
        all_documents = []
        failed_urls = []

        for url in URLS:
            try:
                log(f"Loading: {url}")
                loader = PyPDFLoader(url)
                docs = loader.load()

                title = url.split("/")[-1]
                readable_title = PAPER_TITLES.get(title, title)
                for doc in docs:
                    doc.metadata["title"] = title
                    doc.metadata["paper_name"] = readable_title

                all_documents.extend(docs)
                log(f"  -> {len(docs)} pages loaded")
            except Exception as e:
                log(f"  FAILED: {e}")
                failed_urls.append(url)

        log(f"Total pages loaded: {len(all_documents)}")
        if failed_urls:
            log(f"Failed URLs ({len(failed_urls)}): {failed_urls}")

        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(all_documents, f)
        log(f"Saved to {PICKLE_PATH} for future runs.")

    pages_by_paper = defaultdict(list)
    for doc in all_documents:
        pages_by_paper[doc.metadata["title"]].append(doc)

    merged_documents = []
    for title, pages in pages_by_paper.items():
        full_text = "\n\n".join(p.page_content for p in pages)
        paper_name = pages[0].metadata.get("paper_name", title)
        merged_documents.append(Document(
            page_content=full_text,
            metadata={"title": title, "paper_name": paper_name}
        ))

    for doc in merged_documents:
        doc.page_content = clean_text(doc.page_content)

    log(f"Merged into {len(merged_documents)} documents (one per paper), cleaned.")
    return merged_documents


# =========================================================
# Step 4: semantic chunking
# =========================================================
def build_chunks(merged_documents, log=print):
    """Splits the documents into semantic chunks (with pickle cache)."""

    if os.path.exists(SEMANTIC_CHUNKS_PATH):
        log("Cached chunks found — loading (no re-chunking needed).")
        with open(SEMANTIC_CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        log(f"{len(chunks)} chunks loaded from cache.")
        return chunks

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    log("No cache found — running semantic chunking (this can take a few minutes)...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    raw_semantic_chunks = semantic_splitter.split_documents(merged_documents)
    log(f"Raw chunks created: {len(raw_semantic_chunks)}")

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    oversized_count = 0
    dropped_count = 0

    for chunk in raw_semantic_chunks:
        text = chunk.page_content

        if is_low_quality_chunk(text):
            dropped_count += 1
            continue

        if len(text) > MAX_CHUNK_CHARS:
            oversized_count += 1
            sub_docs = fallback_splitter.split_documents([chunk])
            for sub in sub_docs:
                if is_low_quality_chunk(sub.page_content):
                    dropped_count += 1
                    continue
                chunks.append(sub)
        else:
            chunks.append(chunk)

    log(f"Oversized chunks re-split: {oversized_count}")
    log(f"Low-quality/noise chunks discarded: {dropped_count}")
    log(f"Final chunk count: {len(chunks)}")

    with open(SEMANTIC_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    log(f"Saved to {SEMANTIC_CHUNKS_PATH}.")

    return chunks


def chunk_stats(chunks):
    titles = [c.metadata["title"] for c in chunks]
    counts = Counter(titles)
    lengths = [len(c.page_content) for c in chunks]
    return {
        "per_paper": dict(counts),
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
        "total": len(chunks),
    }


# =========================================================
# Step 6: embeddings + vector store
# =========================================================
def build_vectorstore(chunks, log=print):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(VECTORSTORE_DIR):
        log(f"Existing vector store found ({VECTORSTORE_DIR}) — loading.")
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embedding_model,
        )
        log(f"Vector store loaded. Entry count: {vectorstore._collection.count()}")
    else:
        log(f"No vector store found — embedding {len(chunks)} chunks (this can take a while)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=VECTORSTORE_DIR,
        )
        log(f"Vector store created and saved to '{VECTORSTORE_DIR}'.")

    return vectorstore


# =========================================================
# Step 7: retrieval + reranking
# =========================================================
def load_reranker(log=print):
    from sentence_transformers import CrossEncoder
    log("Loading reranker (BAAI/bge-reranker-base)...")
    return CrossEncoder("BAAI/bge-reranker-base")


def retrieve_and_rerank(query, vectorstore, reranker, top_k=FINAL_TOP_K, pool_size=RETRIEVAL_POOL_SIZE):
    retriever = vectorstore.as_retriever(search_kwargs={"k": pool_size})
    prefixed_query = BGE_QUERY_INSTRUCTION + query
    candidates = retriever.invoke(prefixed_query)

    if not candidates:
        return []

    pairs = [(query, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


# =========================================================
# Step 8: generation with citations
# =========================================================
RAG_SYSTEM_PROMPT = """You are a research assistant answering questions about machine learning papers,
writing for someone who wants to genuinely UNDERSTAND the concept, not just see a list of citations.

Rules you MUST follow:
1. Ground every factual claim in the provided context chunks. Do not use outside knowledge for
   facts, numbers, architectures, or results.
2. Structure matters. Do not just concatenate fragments from different sources back to back.
   For broad/conceptual questions (e.g. "what is an encoder?"):
   a. Start with 1-2 plain-language sentences giving a general, unified explanation of the concept —
      you MAY phrase this in your own words as long as it accurately reflects what the sources
      collectively describe, without adding facts absent from them.
   b. Then explain how it's implemented in each relevant paper as a Markdown bullet list, one
      bullet per paper, formatted as: `- **Paper Name** — explanation [Paper Name]`
      (bold the paper name at the start of the bullet, still include the [Paper Name] citation).
   c. If sources define the term differently depending on context (e.g. NLP vs vision), say so
      explicitly instead of blending them into one confusing sentence.
   For narrow/factual questions with a single clear answer, plain prose is fine — only use the
   bullet-per-paper structure when multiple papers/sources are being compared or listed.
3. Use Markdown formatting to aid readability: **bold** key terms or paper names, use bullet lists
   for multi-item or multi-paper answers. Never produce a sequence of disconnected, choppy
   sentences stitched from different chunks.
4. If the context does not contain enough information to answer, say so explicitly instead of guessing.
5. If different chunks give conflicting information, point out the conflict rather than silently picking one.
6. Always answer in the SAME language as the user's question, regardless of the language of the context chunks (which are in English). If the question is in French, answer in French; if in English, answer in English; etc.
7. Never introduce specific examples, numbers, devices, hardware, or use cases that are not explicitly
   stated in the context chunks. This restriction applies to concrete specifics only — it does NOT
   forbid you from explaining or paraphrasing the general concept clearly, per rule 2a.
8. Be concise: clear and complete, but no generic filler or repetition.
"""


def load_llm(api_key=None):
    from langchain_groq import ChatGroq

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "No GROQ_API_KEY set. Get a free key at "
            "https://console.groq.com/keys"
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )


def format_context(reranked_results):
    blocks = []
    for i, (doc, score) in enumerate(reranked_results, start=1):
        paper_name = doc.metadata.get("paper_name", doc.metadata.get("title", "unknown"))
        blocks.append(f"[Source {i} — {paper_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


VERIFICATION_SYSTEM_PROMPT = """You are a fact-checking editor for a research assistant's answers.
Your goal is accuracy WITHOUT sacrificing clarity — a correct but unreadable answer is a failure too.

You will be given:
- A set of numbered source excerpts (the ONLY allowed source of truth)
- A draft answer that cites those sources using [Paper Name]

Your job: check every SPECIFIC claim (a fact, example, number, device, use case, comparison,
architectural detail) against the source excerpts, and rewrite the answer accordingly.

Rules you MUST follow:
1. If a specific claim is NOT explicitly and literally supported by the source excerpts, REMOVE it
   or rephrase it to only state what IS explicitly supported. Do not soften it into a hedge like
   "possibly" — just remove unsupported specifics.
2. Do NOT remove or flag general, unifying explanatory sentences (e.g. "an encoder is a component
   that transforms an input into a representation") as long as they are a fair, accurate summary of
   what the sources collectively describe — these are not "unsupported claims", they are legitimate
   synthesis. Only strip out concrete specifics that have no basis in the text.
3. Preserve the structure and formatting of the draft: if it opens with a general explanation
   followed by a Markdown bullet list (one bullet per paper, paper name in bold), keep that
   structure and formatting intact. Do not turn it into a disconnected list of fragments, and do
   not strip out Markdown formatting (bold, bullets) that was already there.
4. Do not add any new information that wasn't in the draft or the sources.
5. Keep the citation format [Paper Name] for every specific claim that remains.
6. Keep the same language as the draft answer (do not translate it).
7. Keep the answer concise and technical — do not pad it, but do not make it choppy either.
8. If, after removing unsupported specifics, the answer becomes empty or too thin, say explicitly
   that the sources don't fully support a detailed answer, rather than inventing filler.
9. Output ONLY the corrected answer text. No preamble, no explanation of what you changed, no
   meta-commentary.
"""


def verify_and_correct_answer(draft_answer, context, llm):
    """Second LLM pass: re-reads the generated answer against the full
    source excerpts and removes/rewrites any claim not literally grounded
    in the context (e.g. invented examples or details)."""
    verification_message = f"""Source excerpts:

{context}

Draft answer to fact-check:

{draft_answer}

Rewrite the draft answer following your fact-checking rules."""

    response = llm.invoke([
        ("system", VERIFICATION_SYSTEM_PROMPT),
        ("user", verification_message),
    ])

    return response.content


def generate_answer(query, vectorstore, reranker, llm, top_k=FINAL_TOP_K, verify=True):
    """Full retrieve -> rerank -> generate -> (verify/correct) pipeline.
    Returns (answer_text, list_of_sources_with_excerpts)."""
    reranked_results = retrieve_and_rerank(query, vectorstore, reranker, top_k=top_k)

    if not reranked_results:
        return "No relevant context found in the knowledge base.", []

    context = format_context(reranked_results)

    user_message = f"""Context:

{context}

Question: {query}

Answer using only the context above, citing sources as [Paper Name]."""

    response = llm.invoke([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", user_message),
    ])

    final_answer = response.content

    if verify:
        final_answer = verify_and_correct_answer(final_answer, context, llm)

    sources = []
    for doc, score in reranked_results:
        sources.append({
            "paper_name": doc.metadata.get("paper_name", doc.metadata.get("title", "unknown")),
            "score": float(score),
            "excerpt": doc.page_content[:300],
        })

    return final_answer, sources