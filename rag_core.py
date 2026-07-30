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
import shutil
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

# =========================================================
# Hugging Face Hub — prebuilt index storage
# =========================================================
# Public dataset repo where the final chroma_db/ + pickle caches are
# uploaded once (via upload_to_hf.py) so that cold starts on Streamlit
# Cloud (or any fresh container) can download a ready-made index instead
# of rebuilding it from the 26 source PDFs every time.
HF_REPO_ID = "EmnaMALLEK/rag-app"
HF_REPO_TYPE = "dataset"

# =========================================================
# Page tracking: invisible markers embedded during merging so page numbers
# can be recovered after chunking, even though chunks no longer line up
# with page boundaries. Uses U+2060 (WORD JOINER), an invisible character
# that won't show up in rendered text and won't interfere with cleaning
# regexes or the embedding/reranking models.
# =========================================================
_PAGE_MARKER_TEMPLATE = "\u2060PAGEMARK{page}\u2060"
_PAGE_MARKER_PATTERN = re.compile(r"\u2060PAGEMARK(\d+)\u2060")


def _insert_page_markers(pages):
    """Joins a paper's per-page Documents into one string, with an invisible
    page marker inserted right before each page's text."""
    parts = []
    for doc in pages:
        page_number = doc.metadata.get("page", len(parts)) + 1  # 1-indexed, human-readable
        parts.append(_PAGE_MARKER_TEMPLATE.format(page=page_number) + doc.page_content)
    return "\n\n".join(parts)


def _strip_page_markers(text):
    return _PAGE_MARKER_PATTERN.sub("", text)

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
# Hugging Face Hub — download prebuilt assets on cold start
# =========================================================
def ensure_prebuilt_assets(log=print):
    """
    Cold-start helper: if the local chroma_db/ folder and pickle caches are
    missing (e.g. a fresh Streamlit Cloud container after a restart or
    redeploy), download the prebuilt versions from the Hugging Face Hub
    dataset repo instead of rebuilding the whole pipeline from the 26
    source PDFs.

    Safe to call every time: it's a no-op if the files are already present
    locally (e.g. on your own machine during development, or if a previous
    call already downloaded them into this same container).
    """
    needs_vectorstore = not os.path.exists(VECTORSTORE_DIR)
    needs_documents_pkl = not os.path.exists(PICKLE_PATH)
    needs_chunks_pkl = not os.path.exists(SEMANTIC_CHUNKS_PATH)

    if not (needs_vectorstore or needs_documents_pkl or needs_chunks_pkl):
        log("Prebuilt assets already present locally — skipping HF download.")
        return

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        log("huggingface_hub not installed — cannot fetch prebuilt assets, "
            "falling back to building from scratch.")
        return

    try:
        if needs_vectorstore:
            log(f"Downloading prebuilt vector store from {HF_REPO_ID} ...")
            snapshot_path = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                allow_patterns=["chroma_db/*"],
            )
            downloaded_chroma_dir = os.path.join(snapshot_path, "chroma_db")
            if os.path.isdir(downloaded_chroma_dir):
                shutil.copytree(downloaded_chroma_dir, VECTORSTORE_DIR)
                log("Vector store downloaded and ready.")

        if needs_documents_pkl:
            log("Downloading all_documents.pkl ...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                filename="all_documents.pkl",
            )
            shutil.copy(downloaded, PICKLE_PATH)

        if needs_chunks_pkl:
            log("Downloading semantic_chunks.pkl ...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                filename="semantic_chunks.pkl",
            )
            shutil.copy(downloaded, SEMANTIC_CHUNKS_PATH)

        log("Prebuilt assets ready — pipeline will load them instead of rebuilding.")

    except Exception as e:
        log(f"Could not download prebuilt assets from Hugging Face Hub: {e}. "
            "Falling back to building from scratch (slower cold start).")


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
        full_text = _insert_page_markers(pages)
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

    # Recover each chunk's page number(s) from the invisible markers embedded
    # during merging, then strip those markers out before the text is stored/
    # embedded — they were only ever meant to survive long enough to be read
    # back out here, not to appear in the actual indexed text.
    current_page_by_title = {}
    for chunk in chunks:
        title = chunk.metadata.get("title")
        marker_matches = list(_PAGE_MARKER_PATTERN.finditer(chunk.page_content))
        if marker_matches:
            page_start = int(marker_matches[0].group(1))
            page_end = int(marker_matches[-1].group(1))
            current_page_by_title[title] = page_end
        else:
            # No marker fell inside this chunk — it sits entirely within
            # whichever page we last saw a marker for.
            page_start = page_end = current_page_by_title.get(title)

        chunk.metadata["page_start"] = page_start
        chunk.metadata["page_end"] = page_end
        chunk.page_content = _strip_page_markers(chunk.page_content).strip()

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
        try:
            count = vectorstore._collection.count()
            if count == 0:
                raise ValueError("vector store loaded but contains 0 entries")
            log(f"Vector store loaded. Entry count: {count}")
        except Exception as e:
            log(f"Existing vector store looks corrupted/empty ({e}) — "
                f"deleting and rebuilding from chunks.")
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=VECTORSTORE_DIR,
            )
            log(f"Vector store rebuilt and saved to '{VECTORSTORE_DIR}'.")
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


# =========================================================
# Query decomposition: guarantee both papers surface in comparison questions
# ---------------------------------------------------------
# MMR helps diversity, but doesn't GUARANTEE that two named papers both get
# retrieved — one can still crowd out the other (e.g. QLoRA's paper also
# discusses LoRA extensively, so a "LoRA vs QLoRA" query can still end up
# retrieving only QLoRA chunks on pure similarity/MMR). When a question
# explicitly names 2+ known papers, we instead run a SEPARATE, filtered
# retrieval pass per named paper, guaranteeing each has real representation
# in the candidate pool before reranking — rather than hoping the general
# search surfaces both.
# =========================================================
_PAPER_KEYWORDS = {
    "QLoRA": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "LoRA": "LoRA: Low-Rank Adaptation of Large Language Models",
    "BERT": "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-4": "GPT-4 Technical Report",
    "GPT-3": "GPT-3: Language Models are Few-Shot Learners",
    "InstructGPT": "InstructGPT: Training with Human Feedback",
    "LLaMA": "LLaMA: Open and Efficient Foundation Language Models",
    "PaLM": "PaLM: Scaling Language Modeling with Pathways",
    "T5": "T5: Exploring the Limits of Transfer Learning",
    "ResNet": "ResNet: Deep Residual Learning for Image Recognition",
    "ViT": "Vision Transformer (ViT)",
    "Vision Transformer": "Vision Transformer (ViT)",
    "CLIP": "CLIP: Learning Transferable Visual Models",
    "GAN": "GAN: Generative Adversarial Networks",
    "VAE": "VAE: Auto-Encoding Variational Bayes",
    "DDPM": "DDPM: Denoising Diffusion Probabilistic Models",
    "Stable Diffusion": "Stable Diffusion: High-Resolution Latent Diffusion Models",
    "DPR": "DPR: Dense Passage Retrieval",
    "AlphaGo": "AlphaZero (AlphaGo family)",
    "AlphaZero": "AlphaZero (AlphaGo family)",
    "PPO": "PPO: Proximal Policy Optimization",
    "DQN": "DQN: Playing Atari with Deep Reinforcement Learning",
    "Word2Vec": "Word2Vec: Efficient Estimation of Word Representations",
    "Dropout": "Dropout: Preventing Overfitting",
    "Adam": "Adam: A Method for Stochastic Optimization",
    "Chain-of-Thought": "Chain-of-Thought Prompting",
}


def _detect_mentioned_papers(query):
    """Returns the list of known papers explicitly named in the query
    (e.g. "LoRA vs QLoRA" -> both papers), in first-mention order."""
    mentioned = []
    for keyword, paper_name in _PAPER_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            if paper_name not in mentioned:
                mentioned.append(paper_name)
    return mentioned


def retrieve_and_rerank(query, vectorstore, reranker, top_k=FINAL_TOP_K, pool_size=RETRIEVAL_POOL_SIZE):
    prefixed_query = BGE_QUERY_INSTRUCTION + query
    candidates = []
    seen_content = set()

    def _add(docs):
        for d in docs:
            key = d.page_content[:200]  # de-dupe if the same chunk surfaces twice
            if key not in seen_content:
                seen_content.add(key)
                candidates.append(d)

    mentioned_papers = _detect_mentioned_papers(query)
    if len(mentioned_papers) >= 2:
        # Guarantee each explicitly-named paper gets its own retrieval pass
        # (filtered to just that paper), so neither can crowd the other out.
        per_paper_k = max(pool_size // len(mentioned_papers), 4)
        for paper_name in mentioned_papers:
            paper_retriever = vectorstore.as_retriever(
                search_kwargs={"k": per_paper_k, "filter": {"paper_name": paper_name}},
            )
            _add(paper_retriever.invoke(prefixed_query))

    # Always also run the normal broad MMR retrieval — covers everything
    # else relevant, including questions that don't name specific papers.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": pool_size,
            "fetch_k": pool_size * 3,
            "lambda_mult": 0.5,
        },
    )
    _add(retriever.invoke(prefixed_query))

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

2. Structure depends on how many DISTINCT papers actually appear in the context, not how many
   chunks were retrieved:
   a. If the retrieved chunks come from a SINGLE paper (even if there are 5 chunks from it),
      write in plain, flowing prose. Do NOT create one bullet per chunk — that produces
      repetitive, padded-looking output. Synthesize the chunks into a single coherent
      explanation, as if you were summarizing one paper's ideas, not listing sources.
   b. Only use the bullet-per-paper structure when the context genuinely contains chunks from
      MULTIPLE DIFFERENT papers and the question calls for comparing/listing them. In that case:
      - Start with 1-2 plain-language sentences giving a general, unified explanation of the
        concept — you MAY phrase this in your own words as long as it accurately reflects what
        the sources collectively describe, without adding facts absent from them.
      - Then one bullet per PAPER (never one bullet per chunk — merge multiple chunks from the
        same paper into a single bullet), formatted EXACTLY like this:
        `- **Paper Name** — explanation of that paper's contribution [Paper Name]`
        Example: `- **LoRA: Low-Rank Adaptation of Large Language Models** — freezes the
        pretrained weights and injects small trainable rank-decomposition matrices into each
        layer, drastically cutting the number of trainable parameters [LoRA: Low-Rank Adaptation
        of Large Language Models]`
      - If sources define the term differently depending on context (e.g. NLP vs vision), say so
        explicitly instead of blending them into one confusing sentence.
   For narrow/factual questions with a single clear answer, plain prose is fine regardless of how
   many chunks were retrieved.

3. Citation format: cite ONLY as `[Paper Name]` — never `[Source N — Paper Name]`, never a bare
   number, and never cite the same paper twice in the same sentence or bullet. One citation per
   claim is enough.

3b. Do not repeat the same paper's citation after every single sentence. Instead, group
    consecutive sentences that draw from the same paper into one block, and place that paper's
    citation ONCE, at the END of that block (after the last sentence belonging to it) — not at
    the start, and not repeated mid-paragraph. If the answer then moves on to a claim grounded in
    a DIFFERENT paper, place that paper's citation once at the end of that new block, in the same
    way. If the whole answer only ever draws from one paper, this means the citation appears
    exactly once, at the very end of the answer.

3c. Never write an inline attribution phrase like "as described in [Paper Name], ..." or "as
    discussed in [Paper Name], ..." in the middle of a sentence. Always place the citation at the
    END of the sentence or clause it supports, after the content, not interrupting it. Bad:
    "QLoRA, as discussed in [Paper], builds upon LoRA by quantizing..." Good: "QLoRA builds upon
    LoRA by quantizing the model to 4-bit precision [Paper]."

3d. If a single claim or sentence is genuinely supported by MORE THAN ONE paper at once, combine
    them into ONE bracket citation, separated by a semicolon and space:
    `[Paper A; Paper B]`. NEVER write two separate brackets stacked back-to-back like
    `[Paper A] [Paper B]` — that reads as a formatting error, not a real distinction between
    sources. Only combine papers that both genuinely support that exact claim — do not pad a
    citation with an extra paper "just in case."

3e. Every citation must point to the paper where that SPECIFIC claim actually appears — not
    merely a paper that appears somewhere among the sources. Before writing a citation, check
    which numbered source excerpt the claim's specific facts came from, and cite that paper, not
    a different one that happens to be topically related. A citation naming the wrong paper for
    a claim is worse than no citation at all.

4. Use Markdown formatting to aid readability: **bold** key terms or paper names, use bullet lists
   only per rule 2b. Never produce a sequence of disconnected, choppy sentences stitched from
   different chunks.

5. If the context does not contain enough information to answer, say so explicitly instead of guessing.

6. If different chunks give conflicting information, point out the conflict rather than silently picking one.

7. Always answer in the SAME language as the user's question, regardless of the language of the context chunks (which are in English). If the question is in French, answer in French; if in English, answer in English; etc.

8. Never introduce specific examples, numbers, devices, hardware, or use cases that are not explicitly
   stated in the context chunks. This restriction applies to concrete specifics only — it does NOT
   forbid you from explaining or paraphrasing the general concept clearly, per rule 2a.

9. Mathematical formulas and equations MUST be written in LaTeX, wrapped in single dollar signs for
   inline math (e.g. `$\\theta_{t+1} = \\theta_t - \\alpha \\cdot g_t$`) or double dollar signs for
   standalone/display equations (e.g. `$$\\theta_{t+1} = \\theta_t - \\frac{\\alpha}{\\sqrt{v_t} + \\epsilon} m_t$$`).
   Never write formulas as plain unformatted text with concatenated symbols — always use proper
   LaTeX subscript (`_`), superscript (`^`), and fraction (`\\frac{}{}`) notation.

10. Be concise: clear and complete, but no generic filler or repetition.
"""


# =========================================================
# Multi-key Groq fallback
# ---------------------------------------------------------
# Groq's free tier has a daily token cap per account. Instead of a single
# hard-coded LLM client, we now accept a LIST of API keys and try them in
# order, moving on to the next key whenever the current one comes back
# rate-limited (HTTP 429). This spreads usage across several free-tier
# accounts instead of failing outright once one account's quota is hit.
# =========================================================
def _is_rate_limit_error(e):
    msg = str(e)
    return "rate_limit_exceeded" in msg or "429" in msg or "rate limit" in msg.lower()


def _invoke_with_fallback(messages, api_keys, log=print):
    """Tries each Groq API key in turn. If a key is rate-limited (429),
    moves on to the next one instead of failing the whole request.
    Any non-rate-limit error is raised immediately (no point retrying
    a different key for e.g. a malformed request)."""
    from langchain_groq import ChatGroq

    if not api_keys:
        raise ValueError(
            "No GROQ_API_KEY configured. Get a free key at "
            "https://console.groq.com/keys"
        )

    last_error = None
    for i, key in enumerate(api_keys):
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=1024,
                groq_api_key=key,
            )
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                log(f"Key #{i + 1} rate-limited — trying next key...")
                continue
            raise  # non-rate-limit errors surface immediately, no point retrying

    # every key was rate-limited
    raise last_error


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

3. Preserve the structure and formatting of the draft: if it's plain prose, keep it as plain prose.
   If it opens with a general explanation followed by a Markdown bullet list (one bullet PER PAPER,
   paper name in bold), keep that structure and formatting intact. Do not turn a single-paper
   plain-prose answer into a bulleted list, and do not turn a legitimate multi-paper bullet list
   into a disconnected list of fragments. Do not strip out Markdown formatting (bold, bullets)
   that was already there.

4. Citation format: every citation must read exactly `[Paper Name]` — never `[Source N — Paper
   Name]`. If the draft cites the same paper twice in one sentence or bullet, remove the duplicate
   and keep only one citation for that claim.

4b. Citations belong at the END of a block of consecutive sentences drawn from the same paper —
    not repeated after each sentence, and not placed at the start of the block. If the draft
    already places a citation after every sentence for the same paper, collapse those into a
    single citation at the end of that block, BUT ONLY if doing so does not strip the ONLY
    citation away from a sentence that has no other citation covering it — never leave a claim
    completely uncited just to reduce repetition. If the whole answer only cites one paper and
    every sentence is genuinely part of one continuous block, it should appear exactly once, at
    the very end. Do NOT collapse citations that belong to genuinely different papers — each
    distinct paper still gets its own citation at the end of its own block.

4c. If the draft has two or more citation brackets stacked back-to-back with nothing but
    whitespace between them (e.g. `[Paper A] [Paper B]`), combine them into a single bracket
    separated by a semicolon: `[Paper A; Paper B]`.

4d. For every citation in the draft, check it against the numbered source excerpts and confirm it
    names the CORRECT paper for that specific claim — not merely a paper that appears somewhere
    in the provided context. If a claim's facts actually come from a different numbered excerpt
    than the one cited, correct the citation to the right paper. A wrong citation is a factual
    error and must be fixed like any other unsupported claim.

5. Preserve any LaTeX math formatting exactly as given (`$...$` or `$$...$$`). Do not convert
   equations into plain text, and do not strip the dollar-sign delimiters.

6. Do not add any new information that wasn't in the draft or the sources.

7. Keep the same language as the draft answer (do not translate it).

8. Keep the answer concise and technical — do not pad it, but do not make it choppy either.

9. If, after removing unsupported specifics, the answer becomes empty or too thin, say explicitly
   that the sources don't fully support a detailed answer, rather than inventing filler.

10. Output ONLY the corrected answer text. No preamble, no explanation of what you changed, no
    meta-commentary.
"""


def verify_and_correct_answer(draft_answer, context, api_keys, log=print):
    """Second LLM pass: re-reads the generated answer against the full
    source excerpts and removes/rewrites any claim not literally grounded
    in the context (e.g. invented examples, details, or wrong citations)."""
    verification_message = f"""Source excerpts:

{context}

Draft answer to fact-check:

{draft_answer}

Rewrite the draft answer following your fact-checking rules."""

    return _invoke_with_fallback([
        ("system", VERIFICATION_SYSTEM_PROMPT),
        ("user", verification_message),
    ], api_keys, log=log)


def _merge_adjacent_citations(text: str) -> str:
    """If two or more citation brackets appear back-to-back with nothing
    but whitespace between them (e.g. "...models [CLIP] [ViT]."), merges
    them into a single combined bracket ("...models [CLIP; ViT]."), since
    stacking separate brackets like that reads as a formatting glitch, not
    a real distinction between sources.

    This ONLY touches citations with literally nothing but whitespace
    between them — it never removes a citation that has real sentence
    content attached to it, so it can never strand a claim with no
    attribution (unlike the old cross-sentence "collapse" approach, which
    could and did — see the code comment history / report for the bug this
    replaced).
    """
    pattern = re.compile(r"\[[^\]]+\](?:\s*\[[^\]]+\])+")

    def _merge(match):
        names = re.findall(r"\[([^\]]+)\]", match.group(0))
        flat = []
        for n in names:
            for part in n.split(";"):
                part = part.strip()
                if part and part not in flat:
                    flat.append(part)
        return "[" + "; ".join(flat) + "]"

    return pattern.sub(_merge, text)


def _build_sources(reranked_results, allowed_paper_names=None):
    """Aggregates reranked chunks into one entry per distinct paper, merging
    together the page numbers of every chunk retrieved for that paper."""
    by_paper = {}
    for doc, score in reranked_results:
        paper_name = doc.metadata.get("paper_name", doc.metadata.get("title", "unknown"))
        if allowed_paper_names is not None and paper_name not in allowed_paper_names:
            continue

        entry = by_paper.setdefault(paper_name, {
            "paper_name": paper_name,
            "score": float(score),
            "excerpt": doc.page_content[:300],
            "pages": set(),
        })
        entry["score"] = max(entry["score"], float(score))

        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        if page_start is not None and page_end is not None:
            entry["pages"].add(
                str(page_start) if page_start == page_end else f"{page_start}\u2013{page_end}"
            )

    sources = []
    for entry in by_paper.values():
        pages_sorted = sorted(entry["pages"], key=lambda p: int(p.split("\u2013")[0]))
        entry["pages_display"] = ", ".join(pages_sorted) if pages_sorted else None
        del entry["pages"]
        sources.append(entry)

    sources.sort(key=lambda s: s["score"], reverse=True)
    return sources


def generate_answer(query, vectorstore, reranker, api_keys, top_k=FINAL_TOP_K, verify=True, log=print):
    """Full retrieve -> rerank -> generate -> (verify/correct) pipeline.
    `api_keys` is a list of Groq API keys tried in order, falling back to
    the next one whenever the current one is rate-limited.
    Returns (answer_text, list_of_sources_with_excerpts)."""
    reranked_results = retrieve_and_rerank(query, vectorstore, reranker, top_k=top_k)

    if not reranked_results:
        return "No relevant context found in the knowledge base.", []

    context = format_context(reranked_results)

    user_message = f"""Context:

{context}

Question: {query}

Answer using only the context above, citing sources as [Paper Name]."""

    final_answer = _invoke_with_fallback([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", user_message),
    ], api_keys, log=log)

    if verify:
        final_answer = verify_and_correct_answer(final_answer, context, api_keys, log=log)

    final_answer = _merge_adjacent_citations(final_answer)

    # Only report sources whose paper actually got cited in the final answer —
    # otherwise this always shows top_k regardless of what the model actually
    # used, which is misleading (e.g. "5 sources used" for a single-paper
    # answer). Citations can be combined ("[Paper A; Paper B]"), so parse out
    # each individual paper name from every bracket rather than a simple
    # substring check. Track first-appearance ORDER too, since that order
    # becomes the reference numbering below.
    ordered_cited_names = []
    for raw_group in re.findall(r"\[([^\]]+)\]", final_answer):
        for name in raw_group.split(";"):
            name = name.strip()
            if name and name not in ordered_cited_names:
                ordered_cited_names.append(name)

    sources = _build_sources(reranked_results, allowed_paper_names=set(ordered_cited_names) or None)

    # Fallback: if for some reason no citation matched literally (formatting drift),
    # don't silently show zero sources — fall back to the full retrieved set.
    if not sources:
        sources = _build_sources(reranked_results, allowed_paper_names=None)

    # Turn repeated full-name citations into short numbered references —
    # [Attention Is All You Need] appearing 4 times in one answer is visually
    # heavy; [1] appearing 4 times, with the source list below spelling out
    # what [1] is, reads far more like a real paper and far less repetitive,
    # without removing a single citation's worth of grounding.
    paper_to_number = {name: i + 1 for i, name in enumerate(ordered_cited_names)}

    def _numbered(match):
        names = [n.strip() for n in match.group(1).split(";") if n.strip()]
        numbers = [str(paper_to_number.get(n, "?")) for n in names]
        return "[" + ",".join(numbers) + "]"

    final_answer = re.sub(r"\[([^\]]+)\]", _numbered, final_answer)

    for s in sources:
        s["number"] = paper_to_number.get(s["paper_name"])
    sources.sort(key=lambda s: s["number"] if s["number"] is not None else 999)

    return final_answer, sources