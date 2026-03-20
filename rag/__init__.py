"""RAG: Qdrant + Ollama embeddings for Atlas."""

from .retriever import RAGRetriever, build_rag_retriever_if_configured

__all__ = ["RAGRetriever", "build_rag_retriever_if_configured"]
