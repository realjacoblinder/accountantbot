# ingest_background.py
# Ingests a .jsonl file (one JSON object per line) into ChromaDB
# using the configured embedding model. Safe to re-run — skips already-embedded chunks.

import json
from llm_client import create_embeddings
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from embedding_chunks import chunk_documents, load_embedding_config

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_FILE      = "./data/background/chunks.jsonl"  # path to your .jsonl file
STORE_DIR       = "./chroma_tax_code"
COLLECTION_NAME = "tax_code"
BATCH_SIZE      = load_embedding_config(purpose="ingestion")["batch_size"]
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ Skipping malformed line {line_num}: {e}")
    return records


def records_to_documents(records: list[dict]) -> list[Document]:
    docs = []
    for record in records:
        text = record.get("text", "").strip()
        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "doc_id":      record["id"],           # e.g. "chunk_00001"
                "path":        record.get("path", ""), # e.g. "26 U.S.C. § 1 s1 a"
                "identifier":  record.get("identifier", ""), # e.g. "/us/usc/t26/s1/a"
                "char_count":  record.get("char_count", 0),
                "token_est":   record.get("token_est", 0),
                "source_type": "background",
                "corpus":      "us_tax_code",
            }
        ))
    # Also enforce token budgets when ingesting historical JSONL files.
    return chunk_documents(docs, kind="tax_code")


def get_existing_ids(store: Chroma) -> set[str]:
    try:
        result = store.get(include=["metadatas"])
        return {m["doc_id"] for m in result["metadatas"] if "doc_id" in m}
    except Exception:
        return set()


def batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def ingest():
    jsonl_path = Path(JSONL_FILE)
    if not jsonl_path.exists():
        print(f"✗ File not found: {JSONL_FILE}")
        return

    print(f"Loading {jsonl_path.name}...")
    records = load_jsonl(jsonl_path)
    print(f"  → {len(records)} records loaded")

    docs = records_to_documents(records)
    print(f"  → {len(docs)} valid documents built")

    embeddings = create_embeddings(purpose="ingestion")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=STORE_DIR,
        embedding_function=embeddings
    )

    existing_ids = get_existing_ids(store)
    print(f"  → {len(existing_ids)} chunks already in store (will skip)\n")

    new_docs = [d for d in docs if d.metadata["doc_id"] not in existing_ids]
    if not new_docs:
        print("✓ All chunks already embedded. Nothing to do.")
        return

    print(f"Embedding {len(new_docs)} new chunks in batches of {BATCH_SIZE}...\n")

    total_added = 0
    for doc_batch in batch(new_docs, BATCH_SIZE):
        ids = [d.metadata["doc_id"] for d in doc_batch]
        store.add_documents(documents=doc_batch, ids=ids)
        total_added += len(doc_batch)
        print(f"  → Progress: {total_added}/{len(new_docs)}", end="\r")

    print(f"\n\n{'─'*50}")
    print(f"✓ Ingestion complete")
    print(f"  Added:   {total_added} chunks")
    print(f"  Skipped: {len(docs) - total_added} chunks (already embedded)")
    print(f"  Store:   {STORE_DIR}")


if __name__ == "__main__":
    ingest()