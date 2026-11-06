# rag_query.py
from query_prompts import RAG_SYSTEM_PROMPT
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from custom_langchain import HybridEnsembleRetriever, DualStoreRetriever
from llm_client import create_chat_model, create_embeddings
from retrieval_content import keyword_documents, retrieval_content

BACKGROUND_STORE = "./chroma_tax_code"
USER_STORE = "./chroma_user_docs"



def build_chain():
    embeddings = create_embeddings()

    # Load stores
    bg_store = Chroma(collection_name="tax_code", persist_directory=BACKGROUND_STORE, embedding_function=embeddings)
    u_store = Chroma(collection_name="user_docs", persist_directory=USER_STORE, embedding_function=embeddings)

    # --- Setup Hybrid for Background (Tax Code) ---
    bg_data = bg_store.get()
    bg_docs = [Document(page_content=t, metadata=m) for t, m in zip(bg_data['documents'], bg_data['metadatas'])]
    bg_bm25 = BM25Retriever.from_documents(bg_docs)
    bg_bm25.k = 5
    
    hybrid_background = HybridEnsembleRetriever(
        retrievers=[bg_store.as_retriever(search_kwargs={"k": 5}), bg_bm25],
        weights=[0.7, 0.3]
    )

    # --- Setup Hybrid for User Docs ---
    u_data = u_store.get()
    u_docs = [Document(page_content=t, metadata=m) for t, m in zip(u_data['documents'], u_data['metadatas'])]
    u_bm25 = BM25Retriever.from_documents(keyword_documents(u_docs))
    u_bm25.k = 3
    
    hybrid_user = HybridEnsembleRetriever(
        retrievers=[u_store.as_retriever(search_kwargs={"k": 5}), u_bm25],
        weights=[0.5, 0.5]
    )

    ensemble_retriever = DualStoreRetriever(
        user_retriever=hybrid_user,
        background_retriever=hybrid_background
    )

    llm = create_chat_model()

    instructions = RAG_SYSTEM_PROMPT

    def prompt(values):
        return [
            SystemMessage(content=instructions),
            HumanMessage(content=[{"type": "text", "text": values["question"]},
                                  *retrieval_content(values["documents"])]),
        ]

    return (
        {"documents": ensemble_retriever, "question": RunnablePassthrough()}
        | RunnableLambda(prompt) | llm | StrOutputParser()
    )

if __name__ == "__main__":
    chain = build_chain()
    print("Tax Hybrid RAG (v1.3.0 Optimized) ready. Type 'quit' to exit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in {"quit", "exit"}: break
        if not query: continue
        print("\nThinking...\n")
        print(f"Answer:\n{chain.invoke(query)}\n" + "-"*60)
