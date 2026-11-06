# test_user_query.py
# Queries the user document store to inspect what the LLM retrieves.
# Mirrors test_background_query.py but shows user-doc-specific metadata:
# person name, employer, payer, tax year, SSN last 4, extraction method.
# Run: python test_user_query.py

from llm_client import create_embeddings
from pathlib import Path
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
STORE_DIR       = "./chroma_user_docs"
COLLECTION_NAME = "user_docs"
TOP_K           = 6
SHOW_FULL_TEXT  = True
MAX_PREVIEW_LEN = 800
# ─────────────────────────────────────────────────────────────────────────────


def load_store() -> Chroma:
    if not Path(STORE_DIR).exists():
        raise FileNotFoundError(
            f"No store found at '{STORE_DIR}'. "
            "Run ingest_user.py first."
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


def print_store_summary(store: Chroma):
    """Print a breakdown of all documents in the store by file and identity."""
    print(f"{'═'*60}")
    print("STORE CONTENTS SUMMARY")
    print(f"{'═'*60}\n")

    results = store.get(include=["metadatas"])
    metadatas = results["metadatas"]

    # Group by source file
    by_file: dict[str, list[dict]] = {}
    for m in metadatas:
        fname = m.get("source_file", "unknown")
        by_file.setdefault(fname, []).append(m)

    for fname, metas in sorted(by_file.items()):
        # Collect unique identity values seen across all chunks of this file
        persons   = set(m["person_name"]   for m in metas if m.get("person_name"))
        employers = set(m["employer_name"] for m in metas if m.get("employer_name"))
        payers    = set(m["payer_name"]    for m in metas if m.get("payer_name"))
        years     = set(m["tax_year"]      for m in metas if m.get("tax_year"))
        methods   = set(m.get("extraction_method", "?") for m in metas)
        doc_types = set(m.get("document_type", "?") for m in metas)

        print(f"  📄 {fname}")
        print(f"     Chunks:     {len(metas)}")
        print(f"     Type:       {', '.join(doc_types)}")
        print(f"     Method:     {', '.join(methods)}")
        if persons:
            print(f"     Person(s):  {', '.join(persons)}")
        if employers:
            print(f"     Employer:   {', '.join(employers)}")
        if payers:
            print(f"     Payer:      {', '.join(payers)}")
        if years:
            print(f"     Tax year:   {', '.join(sorted(years))}")
        print()


def run_query(store: Chroma, query: str):
    print(f"\n{'═'*60}")
    print(f"QUERY: {query}")
    print(f"{'═'*60}")

    results = store.similarity_search_with_relevance_scores(query, k=TOP_K)

    if not results:
        print("  No results returned.\n")
        return

    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata

        # Build identity line from whatever was detected during ingestion
        identity_parts = []
        if m.get("person_name"):
            identity_parts.append(f"👤 {m['person_name']}")
        if m.get("employer_name"):
            identity_parts.append(f"🏢 {m['employer_name']}")
        if m.get("payer_name"):
            identity_parts.append(f"💳 {m['payer_name']}")
        if m.get("tax_year"):
            identity_parts.append(f"📅 {m['tax_year']}")
        if m.get("ssn_last4"):
            identity_parts.append(f"SSN: ***{m['ssn_last4']}")
        identity_str = "  |  ".join(identity_parts) if identity_parts else "Identity: not detected"

        print(f"\n  Result {i} — Relevance: {score:.4f}")
        print(f"  File:     {m.get('source_file', 'N/A')}  (page {m.get('page', '?')}/{m.get('total_pages', '?')})")
        print(f"  Type:     {m.get('document_type', 'N/A')}  |  Method: {m.get('extraction_method', 'N/A')}")
        print(f"  Identity: {identity_str}")

        # Flag low-confidence detections
        if m.get("detection_confidence") == "low":
            print(f"  ⚠ Low detection confidence — verify document type manually")

        print(f"  {'─'*54}")

        text = doc.page_content
        if not SHOW_FULL_TEXT and len(text) > MAX_PREVIEW_LEN:
            text = text[:MAX_PREVIEW_LEN] + "... [truncated]"
        print(f"  {text.replace(chr(10), chr(10) + '  ')}\n")


# ── Predefined queries targeted at personal tax documents ─────────────────────
test_queries = [
    # W-2 / income
    "wages and federal income tax withheld",
    "Social Security and Medicare tax withheld",
    "401k contributions box 12",
    # 1099 income
    "nonemployee compensation self employment income",
    "interest income dividends",
    "capital gains proceeds cost basis",
    # Deductions / credits
    "mortgage interest paid",
    "charitable contributions donations",
    "student loan interest",
    # Identity / cross-person
    "employee name employer name",
    "Social Security number SSN",
]


def main():
    store = load_store()

    print("Run mode:")
    print("  [1] Store summary — list all documents and detected identities")
    print("  [2] Predefined test queries")
    print("  [3] Interactive mode")
    print("  [4] All of the above")
    mode = input("Choice: ").strip()

    if mode in {"1", "4"}:
        print_store_summary(store)

    if mode in {"2", "4"}:
        print(f"\nRunning {len(test_queries)} predefined queries...\n")
        for q in test_queries:
            run_query(store, q)

    if mode in {"3", "4"}:
        print("\nInteractive mode — type 'quit' to exit\n")
        while True:
            try:
                q = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                run_query(store, q)


if __name__ == "__main__":
    main()