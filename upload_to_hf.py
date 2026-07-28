"""
upload_to_hf.py
------------------
One-time script: uploads your locally-built RAG index (chroma_db/ folder +
the two pickle caches) to a public Hugging Face Hub dataset repo, so your
deployed Streamlit app can download a ready-made index on cold start
instead of rebuilding it from 26 PDFs every time.

Prerequisites:
    - You've already run the app locally at least once, so chroma_db/,
      all_documents.pkl, and semantic_chunks.pkl exist in this folder.
    - pip install huggingface_hub
    - huggingface-cli login   (paste a token with WRITE access, from
      https://huggingface.co/settings/tokens)

Usage:
    python upload_to_hf.py
"""

from huggingface_hub import HfApi, create_repo

REPO_ID = "EmnaMALLEK/rag-app"   # <-- change to your actual HF username/repo
REPO_TYPE = "dataset"


def main():
    api = HfApi()

    # Creates the repo if it doesn't exist. Safe to re-run (exist_ok=True).
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)

    print(f"Uploading chroma_db/ -> {REPO_ID} ...")
    api.upload_folder(
        folder_path="chroma_db",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="chroma_db",
    )

    print("Uploading all_documents.pkl ...")
    api.upload_file(
        path_or_fileobj="all_documents.pkl",
        path_in_repo="all_documents.pkl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print("Uploading semantic_chunks.pkl ...")
    api.upload_file(
        path_or_fileobj="semantic_chunks.pkl",
        path_in_repo="semantic_chunks.pkl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print("\nDone. Your prebuilt index now lives at:")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()