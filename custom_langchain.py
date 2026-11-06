# custom_langchain.py
from typing import List, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class HybridEnsembleRetriever(BaseRetriever):
    """
    Future-proof replacement for EnsembleRetriever. 
    Merges Vector and Keyword (BM25) results using RRF.
    """
    retrievers: List[BaseRetriever]
    weights: List[float]
    c: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_docs = [r.invoke(query) for r in self.retrievers]
        weighted_scores = {}
        doc_map = {}

        for retriever_idx, docs in enumerate(all_docs):
            weight = self.weights[retriever_idx]
            for rank, doc in enumerate(docs):
                # Unique key prevents duplicates between BM25 and Vector results
                content_key = doc.page_content
                score = weight * (1.0 / (self.c + rank))
                weighted_scores[content_key] = weighted_scores.get(content_key, 0) + score
                if content_key not in doc_map:
                    doc_map[content_key] = doc

        sorted_docs = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]

class DualStoreRetriever(BaseRetriever):
    """Fetches from the User and Background hybrid retrievers and merges them."""
    user_retriever: BaseRetriever
    background_retriever: BaseRetriever
    user_weight: float = 0.6
    background_weight: float = 0.4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        user_docs = self.user_retriever.invoke(query)
        background_docs = self.background_retriever.invoke(query)
        
        # Merge logic similar to HybridEnsemble but across the two distinct stores
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(user_docs):
            key = doc.metadata.get("doc_id", doc.page_content[:64])
            scores[key] = scores.get(key, 0) + (self.user_weight * (1.0 / (60 + rank)))
            doc_map[key] = doc

        for rank, doc in enumerate(background_docs):
            key = doc.metadata.get("doc_id", doc.page_content[:64])
            scores[key] = scores.get(key, 0) + (self.background_weight * (1.0 / (60 + rank)))
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]