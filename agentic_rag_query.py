# agentic_rag_query.py
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from typing import List

from custom_langchain import HybridEnsembleRetriever, DualStoreRetriever
from llm_client import create_chat_model, create_embeddings
from retrieval_content import keyword_documents, retrieval_content, source_text

BACKGROUND_STORE = "./chroma_tax_code"
USER_STORE = "./chroma_user_docs"
MAX_ITERATIONS = 5

# SYSTEM_PROMPT = """You are a tax assistant with access to two retrieval tools:
# ... (rest of SYSTEM_PROMPT)

# - `search_tax_code`: searches the US tax code for laws, rules, and definitions
# - `search_user_docs`: searches the user's personal tax documents (W-2s, 1099s, etc.)

# Instructions:
# 1. Always search user documents first to ground your answer in their specific situation.
# 2. Use the tax code tool to find relevant laws, brackets, or deduction rules.
# 3. You may call tools multiple times if the first result is insufficient — for example,
#    if an answer references another code section, look that up too.
# 4. Only answer once you have enough context from the documents.
# 5. Cite the specific USC section (e.g. "26 U.S.C. § 1(a)") for every rule you apply.
# 6. Never fabricate figures or rules not found in the retrieved context."""
SYSTEM_PROMPT = """You are an expert US tax assistant with access to two retrieval tools:

- `search_tax_code`: searches the full US tax code (Title 26 USC) for laws, rules, brackets, and definitions
- `search_user_docs`: searches the user's personal tax documents such as W-2s, 1099s, brokerage statements, and receipts

## How to Answer

1. **Always search user documents first.** Ground every answer in the user's actual figures before applying tax law.
2. **Then search the tax code** to find the relevant statute, bracket, deduction rule, or definition.
3. **Call tools multiple times** when needed — for example, if a deduction rule cross-references another section, look that up too before answering.
4. **Only give a final answer once you have enough retrieved context.** If the documents don't contain a needed figure, say so clearly and explain what the user would need to provide.
5. **Cite every rule** with its specific USC section (e.g. "26 U.S.C. § 1(a)", "26 U.S.C. § 162(a)", "26 U.S.C. § 199A").
6. **Never fabricate numbers, brackets, or rules.** If the retrieved context doesn't contain the answer, say so rather than guessing.
7. **Show your work** for any tax calculations — list each figure, the rule applied, and the arithmetic step by step.

## What You Will See in User Documents

User documents may include, but are not limited to, the following. Extract specific figures carefully:

**W-2 (Wage and Tax Statement)**
- Box 1: Wages, tips, other compensation (gross taxable wages)
- Box 2: Federal income tax withheld
- Box 3/4: Social Security wages and tax withheld
- Box 5/6: Medicare wages and tax withheld
- Box 12: Coded entries (e.g. 12a: D = 401k contributions, 12b: DD = employer health coverage cost)
- Box 14: Other (state disability, union dues, employer-paid tuition)
- Box 15–17: State wages and withholding

**1099-NEC (Nonemployee Compensation)**
- Box 1: Nonemployee compensation (self-employment income — subject to SE tax)

**1099-INT (Interest Income)**
- Box 1: Interest income
- Box 4: Federal income tax withheld
- Box 8: Tax-exempt interest (municipal bonds)

**1099-DIV (Dividends)**
- Box 1a: Total ordinary dividends
- Box 1b: Qualified dividends (taxed at capital gains rates)
- Box 2a: Total capital gain distributions
- Box 5: Section 199A dividends (REIT/pass-through)

**1099-B (Broker / Barter Exchange)**
- Proceeds from sale of stocks, bonds, or crypto
- Cost basis, acquisition date, sale date
- Short-term vs. long-term classification
- Wash sale loss disallowed amounts

**1098 (Mortgage Interest)**
- Box 1: Mortgage interest received (potentially deductible under 26 U.S.C. § 163(h))
- Box 2: Outstanding mortgage principal
- Box 5: Mortgage insurance premiums

**Brokerage Year-End Statements**
- Realized gains/losses summary (short-term and long-term)
- Dividend and interest income totals
- Cost basis method used (FIFO, specific lot, etc.)

**SSA-1099 (Social Security Benefits)**
- Box 5: Net Social Security benefits received
- Up to 85% may be taxable depending on combined income (26 U.S.C. § 86)

**Schedule K-1 (Partnership / S-Corp / Trust)**
- Box 1: Ordinary business income/loss
- Box 2: Net rental real estate income
- Box 14: Self-employment earnings

## Example Reasoning Pattern

User asks: *"How much do I owe in federal income tax?"*

1. Call `search_user_docs("W-2 wages federal income tax withheld")` → get Box 1 wages and Box 2 withholding
2. Call `search_user_docs("1099 interest dividends capital gains")` → check for other income sources
3. Call `search_tax_code("standard deduction 2024 married filing jointly")` → get the applicable deduction amount
4. Call `search_tax_code("tax brackets married filing jointly")` → get the rate table (26 U.S.C. § 1(a))
5. Compute: Adjusted Gross Income → subtract standard deduction → apply brackets → subtract withholding
6. Present the calculation step by step with citations

If at step 2 the user has a 1099-B with capital gains, also call `search_tax_code("capital gains tax rates")` before answering.

## Tone and Format

- Be precise and numerical — always state exact dollar figures from documents, never round silently
- Flag anything that could trigger an audit or requires professional review
- If a question requires information not present in the retrieved documents (e.g. filing status, number of dependents), ask the user for that specific detail before proceeding
- For complex situations (AMT exposure, QBI deduction, passive activity rules), note that a CPA review is advisable"""


def build_tools(background_store: Chroma, user_store: Chroma):
    """
    Creates two retrieval tools using Hybrid Search (BM25 + Vector).
    """
    
    # --- 1. Setup Hybrid for Background (Tax Code) ---
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

    # --- 2. Setup Hybrid for User Docs ---
    u_data = user_store.get()
    u_docs = [
        Document(page_content=t, metadata=m) 
        for t, m in zip(u_data['documents'], u_data['metadatas'])
    ]
    u_bm25 = BM25Retriever.from_documents(keyword_documents(u_docs))
    u_bm25.k = 4
    
    hybrid_user = HybridEnsembleRetriever(
        retrievers=[user_store.as_retriever(search_kwargs={"k": 4}), u_bm25],
        weights=[0.5, 0.5]
    )

    def format_docs(docs: List[Document]) -> str:
        out = []
        for d in docs:
            out.append(source_text(d))
        return "\n\n---\n\n".join(out) if out else "No results found."

    @tool
    def search_tax_code(query: str) -> str:
        """Search the US tax code for laws, rules, brackets, and definitions."""
        docs = hybrid_background.invoke(query)
        return format_docs(docs)

    @tool
    def search_user_docs(query: str) -> list:
        """Search the user's personal tax documents such as W-2s, 1099s, and receipts."""
        docs = hybrid_user.invoke(query)
        return retrieval_content(docs)

    return [search_tax_code, search_user_docs]


# ── Agentic loop ──────────────────────────────────────────────────────────────

def build_agent():
    if not Path(BACKGROUND_STORE).exists():
        raise FileNotFoundError(f"Background store not found at '{BACKGROUND_STORE}'. Run ingest_background.py first.")
    if not Path(USER_STORE).exists():
        raise FileNotFoundError(f"User store not found at '{USER_STORE}'. Run ingest_user.py first.")

    embeddings = create_embeddings()

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

        page_images = []
        for tool_call in response.tool_calls:
            tool_name  = tool_call["name"]
            tool_args  = tool_call["args"]
            tool_id    = tool_call["id"]

            if verbose:
                print(f"    → {tool_name}({tool_args.get('query', '')})")

            result = tool_map[tool_name].invoke(tool_args)

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

        if page_images:
            messages.append(HumanMessage(content=[{
                "type": "text", "text": "These are the cited PDF pages from the retrieved sources. "
                "Read the images for scanned text, tables, and exact field values. "
                "Cite the document and page; do not infer missing figures.",
            }, *page_images]))

    # Safety: if max iterations hit, ask for a final answer with what we have
    messages.append(HumanMessage(content="Please provide your final answer based on the context retrieved so far."))
    final = llm_tools.invoke(messages)
    return final.content


# ── Entry point ───────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("Loading stores and model...")
#     llm_tools, tool_map = build_agent()
#     print("✓ Tax RAG agent ready. Type 'quit' to exit.\n")

#     while True:
#         query = input("Your question: ").strip()
#         if query.lower() in {"quit", "exit"}:
#             break
#         if not query:
#             continue

#         print("\nThinking...\n")
#         answer = run_query(llm_tools, tool_map, query, verbose=True)
#         print(f"\nAnswer:\n{answer}\n")
#         print("-" * 60)
if __name__ == "__main__":
    print("Loading stores and model...")
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
        answer = run_query(llm_tools, tool_map, query, verbose=True)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)
