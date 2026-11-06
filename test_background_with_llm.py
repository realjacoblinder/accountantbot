# test_background_query.py
from pathlib import Path
from llm_client import create_chat_model, load_llm_config, create_embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Import your custom class from the separate file you created
from custom_langchain import HybridEnsembleRetriever

# ── Config ────────────────────────────────────────────────────────────────────
STORE_DIR       = "./chroma_tax_code"
COLLECTION_NAME = "tax_code"
TOP_K           = 10
SHOW_FULL_TEXT  = True
MAX_PREVIEW_LEN = 400
# ─────────────────────────────────────────────────────────────────────────────

test_queries = [
    "What are the tax brackets for married filing jointly?",
    "What is the standard deduction?",
    "Home office deduction requirements",
    "Capital gains tax rates",
    "Self-employment tax calculation",
]


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


def build_rag_chain(store: Chroma):
    """
    Builds a RAG chain using the custom HybridEnsembleRetriever 
    to combine Vector and BM25 search.
    """
    # 1. Extract all documents from Chroma to initialize the BM25 index
    raw_data = store.get()
    all_docs = [
        Document(page_content=text, metadata=met) 
        for text, met in zip(raw_data['documents'], raw_data['metadatas'])
    ]
    
    # 2. Setup the individual retrievers
    vector_retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = TOP_K

    # 3. Initialize the Hybrid Ensemble
    # We weight Vector at 0.7 and BM25 at 0.3 for the tax code
    hybrid_retriever = HybridEnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    llm = create_chat_model()

    prompt = ChatPromptTemplate.from_template("""
    You are a US tax law assistant. Answer the question using only the provided
    tax code excerpts. Cite the specific USC section (e.g. "26 U.S.C. § 1(a)")
    when referencing a rule. If the context does not contain enough information
    to answer, say so clearly — do not guess.

    Tax Code Context:
    {context}

    Question: {question}

    Answer:""")

    def format_docs(docs):
        sections = []
        for doc in docs:
            path = doc.metadata.get("path", "Unknown section")
            sections.append(f"[{path}]\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)

    # Note: Use the hybrid_retriever in the chain
    chain = (
        {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, hybrid_retriever


# ── Retrieval-only test ───────────────────────────────────────────────────────

def run_retrieval_query(store: Chroma, query: str):
    print(f"\n{'═'*60}")
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


# ── LLM test ─────────────────────────────────────────────────────────────────

def run_llm_query(chain, retriever, query: str):
    print(f"\n{'═'*60}")
    print(f"QUERY: {query}")
    print(f"{'═'*60}")

    # Show retrieved sources first so you can see what the LLM was given
    source_docs = retriever.invoke(query)
    print(f"\n  Sources retrieved ({len(source_docs)}):")
    for doc in source_docs:
        score_label = doc.metadata.get("path", doc.metadata.get("doc_id", "?"))
        print(f"    • {score_label}")

    print(f"\n  {'─'*54}")
    print("  LLM Answer:\n")

    # Stream the answer token by token so you don't stare at a blank screen
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)

    print("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    store = load_store()

    print("Run mode:")
    print("  [1] Predefined queries  — retrieval only")
    print("  [2] Interactive         — retrieval only")
    print("  [3] Both                — retrieval only")
    print("  [4] Predefined queries  — LLM answers")
    print("  [5] Interactive         — LLM answers")
    print("  [6] Both                — LLM answers")
    mode = input("Choice: ").strip()

    # Only build the LLM chain if a mode that needs it was selected
    chain, retriever = (None, None)
    if mode in {"4", "5", "6"}:
        print(f"\nLoading {load_llm_config()['model']}...")
        chain, retriever = build_rag_chain(store)
        print("✓ LLM ready\n")

    if mode in {"1", "3"}:
        print(f"\nRunning {len(test_queries)} predefined retrieval queries...\n")
        for q in test_queries:
            run_retrieval_query(store, q)

    if mode in {"2", "3"}:
        print("\nInteractive retrieval mode — type 'quit' to exit\n")
        while True:
            q = input("Query: ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                run_retrieval_query(store, q)

    if mode in {"4", "6"}:
        print(f"\nRunning {len(test_queries)} predefined LLM queries...\n")
        for q in test_queries:
            run_llm_query(chain, retriever, q)

    if mode in {"5", "6"}:
        print("\nInteractive LLM mode — type 'quit' to exit\n")
        while True:
            q = input("Query: ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                run_llm_query(chain, retriever, q)


if __name__ == "__main__":
    main()