# test_background_query.py
# Runs queries directly against the background tax code vector store.
# Tests retrieval only — no LLM, just shows what chunks get returned.
# Run: python test_background_query.py

from llm_client import create_embeddings
from pathlib import Path
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
STORE_DIR       = "./chroma_tax_code"
COLLECTION_NAME = "tax_code"
TOP_K           = 5                    # number of chunks to retrieve per query
SHOW_FULL_TEXT  = True                 # set False to truncate long chunks in output
MAX_PREVIEW_LEN = 1000                # chars to show per chunk when truncating
# ─────────────────────────────────────────────────────────────────────────────


def load_store() -> Chroma:
    if not Path(STORE_DIR).exists():
        raise FileNotFoundError(
            f"No store found at '{STORE_DIR}'. "
            "Run ingest_background.py first."
        )
    embeddings = create_embeddings()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=STORE_DIR,
        embedding_function=embeddings
    )
    count = store._collection.count()
    print(f"✓ Loaded store: '{STORE_DIR}'")
    print(f"  Collection '{COLLECTION_NAME}' contains {count:,} chunks\n")
    return store


def run_query(store: Chroma, query: str):
    print(f"{'═'*60}")
    print(f"QUERY: {query}")
    print(f"{'═'*60}")

    results = store.similarity_search_with_relevance_scores(query, k=TOP_K)

    if not results:
        print("  No results returned.\n")
        return

    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(f"\n  Result {i} — Relevance Score: {score:.4f}")
        print(f"  ID:         {m.get('doc_id', 'N/A')}")
        print(f"  Path:       {m.get('path', 'N/A')}")
        print(f"  Identifier: {m.get('identifier', 'N/A')}")
        print(f"  Tokens:     {m.get('token_count', m.get('token_est', 'N/A'))}")
        print(f"  {'─'*54}")

        text = doc.page_content
        if not SHOW_FULL_TEXT and len(text) > MAX_PREVIEW_LEN:
            text = text[:MAX_PREVIEW_LEN] + "... [truncated]"
        print(f"  {text}\n")


def main():
    store = load_store()

    # ── Predefined test queries ───────────────────────────────────────────────
    # Edit these to test specific areas of the tax code you care about
    test_queries = [
        "What are the tax brackets for married filing jointly?",
        "What is the standard deduction?",
        "Home office deduction requirements",
        "Capital gains tax rates",
        "Self-employment tax calculation",
    ]

    mode = input(
        "Run mode:\n"
        "  [1] Run predefined test queries\n"
        "  [2] Interactive mode (type your own queries)\n"
        "  [3] Both\n"
        "Choice: "
    ).strip()

    if mode in {"1", "3"}:
        print(f"\nRunning {len(test_queries)} predefined queries...\n")
        for query in test_queries:
            run_query(store, query)

    if mode in {"2", "3"}:
        print("\nInteractive mode — type 'quit' to exit\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in {"quit", "exit", "q"}:
                break
            if not query:
                continue
            run_query(store, query)


if __name__ == "__main__":
    main()