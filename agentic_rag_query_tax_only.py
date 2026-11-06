from pathlib import Path
from query_prompts import TAX_ONLY_SYSTEM_PROMPT, FINAL_ANSWER_PROMPT
from retrieval_content import source_text
from llm_client import create_chat_model, create_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from typing import List

# Import your custom classes
from custom_langchain import HybridEnsembleRetriever

BACKGROUND_STORE = "./chroma_tax_code"
MAX_ITERATIONS   = 5

SYSTEM_PROMPT = TAX_ONLY_SYSTEM_PROMPT


def build_tools(background_store: Chroma):
    """
    Creates a retrieval tool using Hybrid Search (BM25 + Vector) for the tax code.
    """
    
    # --- Setup Hybrid for Background (Tax Code) ---
    bg_data = background_store.get()
    bg_docs = [
        Document(page_content=t, metadata=m) 
        for t, m in zip(bg_data['documents'], bg_data['metadatas'])
    ]
    bg_bm25 = BM25Retriever.from_documents(bg_docs)
    bg_bm25.k = 5
    
    hybrid_background = HybridEnsembleRetriever(
        retrievers=[background_store.as_retriever(search_kwargs={"k": 5}), bg_bm25],
        weights=[0.7, 0.3]
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

    return [search_tax_code]


# ── Agentic loop ──────────────────────────────────────────────────────────────

def build_agent():
    if not Path(BACKGROUND_STORE).exists():
        raise FileNotFoundError(f"Background store not found at '{BACKGROUND_STORE}'. Run ingest_background.py first.")

    embeddings = create_embeddings()

    background_store = Chroma(
        collection_name="tax_code",
        persist_directory=BACKGROUND_STORE,
        embedding_function=embeddings
    )

    tools      = build_tools(background_store)
    tool_map   = {t.name: t for t in tools}

    llm = create_chat_model()
    llm_tools  = llm.bind_tools(tools)

    return llm_tools, tool_map


def run_query(llm_tools, tool_map: dict, question: str, verbose: bool = True) -> str:
    """
    Agentic ReAct loop:
      1. LLM decides which tool(s) to call
      2. Tools execute and return results as ToolMessages
      3. LLM sees results and either calls more tools or gives a final answer
      4. Repeat until no more tool calls or MAX_ITERATIONS reached
    """
    print(f"  Running query: '{question[:60]}...' " if len(question) > 60 else f"  Running query: '{question}'")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        response = llm_tools.invoke(messages)
        messages.append(response)

        # If no tool calls, the LLM has produced its final answer
        if not response.tool_calls:
            return response.content

        # Execute every tool call the LLM requested this round
        if verbose:
            print(f"\n  [Round {iteration + 1}] LLM called {len(response.tool_calls)} tool(s):")

        for tool_call in response.tool_calls:
            tool_name  = tool_call["name"]
            tool_args  = tool_call["args"]
            tool_id    = tool_call["id"]

            if verbose:
                print(f"    → {tool_name}({tool_args.get('query', '')})")

            result = tool_map[tool_name].invoke(tool_args)

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_id
            ))

    # Safety: if max iterations hit, ask for a final answer with what we have
    messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
    final = llm_tools.invoke(messages)
    return final.content


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading tax code store and model...")
    try:
        llm_tools, tool_map = build_agent()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        exit(1)
    print("✓ Pure Tax Code RAG agent ready. Type 'quit' to exit.\n")

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
        answer = run_query(llm_tools, tool_map, query, verbose=True)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)