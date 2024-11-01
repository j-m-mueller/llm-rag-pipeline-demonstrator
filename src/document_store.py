"""src.document_store.py -- Implements DocumentStoreManager class for managing FAISS document store.
Provides methods for document storage, embedding generation, and retriever management."""

from typing import List
from haystack.schema import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever


class DocumentStoreManager:
    def __init__(self, embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        """
        Initialize the DocumentStoreManager with FAISS document store and embedding retriever.

        :param embedding_model: The name or path of the embedding model to use
        """
        self.document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat",
            return_embedding=True
        )
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model
        )
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the document store and generate embeddings.

        :param documents: List of Document objects to add to the store
        :return: None
        """
        self.document_store.write_documents(documents)
        self.document_store.update_embeddings(self.retriever)
    
    def get_retriever(self) -> EmbeddingRetriever:
        """
        Get the retriever instance.

        :return: The configured EmbeddingRetriever instance
        """
        return self.retriever 
