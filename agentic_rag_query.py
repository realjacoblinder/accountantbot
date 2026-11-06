# agentic_rag_query.py
from pathlib import Path
import argparse
from agent_diagnostics import LOGGER, TracedRetriever, configure_logging, endpoint_name
from query_prompts import AGENT_SYSTEM_PROMPT, RETRIEVED_IMAGE_PROMPT, FINAL_ANSWER_PROMPT
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from typing import List

from custom_langchain import HybridEnsembleRetriever, DualStoreRetriever
from llm_client import create_chat_model, create_embeddings, load_llm_config
from retrieval_content import keyword_documents, retrieval_content, source_text

BACKGROUND_STORE = "./chroma_tax_code"
USER_STORE = "./chroma_user_docs"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT


def build_tools(background_store: Chroma, user_store: Chroma):
    """
    Creates two retrieval tools using Hybrid Search (BM25 + Vector).
    """
    
    # --- 1. Setup Hybrid for Background (Tax Code) ---
    LOGGER.info("Loading tax-code records for BM25")
    bg_data = background_store.get()
    bg_docs = [
        Document(page_content=t, metadata=m) 
        for t, m in zip(bg_data['documents'], bg_data['metadatas'])
    ]
    LOGGER.info("Building tax-code BM25 records=%d", len(bg_docs))
    bg_bm25 = BM25Retriever.from_documents(bg_docs)
    bg_bm25.k = 5
    
    hybrid_background = HybridEnsembleRetriever(
        retrievers=[TracedRetriever(retriever=background_store.as_retriever(search_kwargs={"k": 5}), label="tax_code.vector"),
                    TracedRetriever(retriever=bg_bm25, label="tax_code.bm25")],
        weights=[0.7, 0.3]
    )

    # --- 2. Setup Hybrid for User Docs ---
    LOGGER.info("Loading user-document records for BM25")
    u_data = user_store.get()
    u_docs = [
        Document(page_content=t, metadata=m) 
        for t, m in zip(u_data['documents'], u_data['metadatas'])
    ]
    LOGGER.info("Building user-document BM25 records=%d keyword_records=%d", len(u_docs), len(keyword_documents(u_docs)))
    u_bm25 = BM25Retriever.from_documents(keyword_documents(u_docs))
    u_bm25.k = 4
    
    hybrid_user = HybridEnsembleRetriever(
        retrievers=[TracedRetriever(retriever=user_store.as_retriever(search_kwargs={"k": 4}), label="user_docs.vector"),
                    TracedRetriever(retriever=u_bm25, label="user_docs.bm25")],
        weights=[0.5, 0.5]
    )

    def format_docs(docs: List[Document]) -> str:
        out = []
        for d in docs:
            out.append(source_text(d))
        return "\n\n---\n\n".join(out) if out else "No results found."

    @tool
    def search_tax_code(query: str) -> str:
        """Retrieve ranked Title 26 excerpts; annual IRS guidance may be unavailable."""
        docs = hybrid_background.invoke(query)
        return format_docs(docs)

    @tool
    def search_user_docs(query: str) -> list:
        """Retrieve ranked personal-document excerpts and page images, not a complete inventory."""
        docs = hybrid_user.invoke(query)
        return retrieval_content(docs)

    return [search_tax_code, search_user_docs]


# ── Agentic loop ──────────────────────────────────────────────────────────────

def build_agent():
    if not Path(BACKGROUND_STORE).exists():
        raise FileNotFoundError(f"Background store not found at '{BACKGROUND_STORE}'. Run ingest_background.py first.")
    if not Path(USER_STORE).exists():
        raise FileNotFoundError(f"User store not found at '{USER_STORE}'. Run ingest_user.py first.")

    config = load_llm_config()
    LOGGER.info("Query model=%s endpoint=%s max_tokens=%d reasoning_effort=%s timeout=%ss",
                config["model"], endpoint_name(config["api_base"]), config["max_tokens"],
                config["reasoning_effort"], config["timeout"])
    embeddings = create_embeddings()
    LOGGER.info("Query embeddings model=%s endpoint=%s", embeddings.settings["model"],
                endpoint_name(embeddings.settings["api_base"]))

    background_store = Chroma(
        collection_name="tax_code",
        persist_directory=BACKGROUND_STORE,
        embedding_function=embeddings
    )
    user_store = Chroma(
        collection_name="user_docs",
        persist_directory=USER_STORE,
        embedding_function=embeddings
    )

    tools      = build_tools(background_store, user_store)
    tool_map   = {t.name: t for t in tools}

    llm = create_chat_model()
    llm_tools  = llm.bind_tools(tools)

    return llm_tools, tool_map


def run_query(llm_tools, tool_map: dict, question: str, verbose: bool = True, *,
              trace_content=False, stream=True, heartbeat_seconds=10) -> str:
    """
    Agentic ReAct loop:
      1. LLM decides which tool(s) to call
      2. Tools execute and return results as ToolMessages
      3. LLM sees results and either calls more tools or gives a final answer
      4. Repeat until no more tool calls or MAX_ITERATIONS reached
    """
    from agent_diagnostics import QueryTrace
    trace = QueryTrace(enabled=verbose, trace_content=trace_content, stream=stream,
                       heartbeat_seconds=heartbeat_seconds)
    trace.info("query started question_chars=%d max_rounds=%d", len(question), MAX_ITERATIONS)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        response = trace.model_call(llm_tools, messages, str(iteration + 1))
        messages.append(response)

        # If no tool calls, the LLM has produced its final answer
        if not response.tool_calls:
            return trace.finish(response.content)

        trace.info("round=%d executing %d tool calls", iteration + 1, len(response.tool_calls))

        page_images = []
        for tool_call in response.tool_calls:
            tool_name  = tool_call["name"]
            tool_args  = tool_call["args"]
            tool_id    = tool_call["id"]

            result = trace.tool_call(tool_map, tool_call, str(iteration + 1))

            # Keep tool results textual and supply visual evidence in a human
            # message after all pending tool calls have their matching result.
            if isinstance(result, list):
                caption = "Retrieved PDF page"
                for part in result:
                    if part.get("type") == "text":
                        caption = part["text"].split("\n", 1)[0]
                    elif part.get("type") == "image_url":
                        page_images.extend([{"type": "text", "text": caption}, part])
                result = [part for part in result if part.get("type") == "text"]

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_id
            ))

        trace.info("round=%d attaching page_images=%d", iteration + 1,
                   sum(p.get("type") == "image_url" for p in page_images))
        if page_images:
            messages.append(HumanMessage(content=[{
                "type": "text", "text": RETRIEVED_IMAGE_PROMPT,
            }, *page_images]))

    # Safety: if max iterations hit, ask for a final answer with what we have
    trace.warning("retrieval round limit reached; requesting a supported partial/final answer")
    messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
    final = trace.model_call(llm_tools, messages, "final")
    if final.tool_calls:
        trace.warning("final response requested more tools after the round limit")
    return trace.finish(final.content)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Query personal tax documents with agent diagnostics.")
    parser.add_argument("--debug", action="store_true", help="Enable debug-level diagnostics.")
    parser.add_argument("--log-file", help="Also write timestamped diagnostics to this file.")
    parser.add_argument("--trace-content", action="store_true", help="Include bounded previews of model text, tool arguments and retrieved text (may contain personal data).")
    parser.add_argument("--no-stream", action="store_true", help="Use blocking model calls with heartbeat logs instead of streaming progress.")
    parser.add_argument("--heartbeat-seconds", type=float, default=10, help="Seconds between waiting messages (default: 10).")
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    configure_logging(debug=args.debug, log_file=args.log_file)
    print("Loading stores and model...", flush=True)
    try:
        llm_tools, tool_map = build_agent()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        exit(1)
    print("✓ Tax RAG agent ready. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        print()
        answer = run_query(llm_tools, tool_map, query, verbose=True,
                           trace_content=args.trace_content, stream=not args.no_stream,
                           heartbeat_seconds=args.heartbeat_seconds)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)
