"""src.pipeline.document_store -- Implements DocumentStoreManager class for managing FAISS document store."""

import os
from pathlib import Path
from typing import List, Optional

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document


class DocumentStoreManager:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 db_path: Optional[str] = None,
                 index_path: Optional[str] = None):
        """
        Initialize the DocumentStoreManager with FAISS document store and embedding retriever.

        :param embedding_model: The name or path of the embedding model to use
        :param db_path: Optional path to the SQLite database file
        :param index_path: Optional path to the FAISS index file
        """
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Use provided paths or default to data directory
        self.db_path = db_path or str(data_dir / "faiss_document_store.db")
        self.index_path = index_path or str(data_dir / "faiss_document_store.faiss")
        
        # Clean up existing files to start fresh
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        
        # Create new document store
        self.document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{self.db_path}",
            embedding_dim=768,  # Dimension for the specified embedding model
            similarity="dot_product"
        )
        
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the document store and generate embeddings."""
        self.document_store.write_documents(documents)
        self.document_store.update_embeddings(self.retriever)
        # Save the updated index
        self.document_store.save(self.index_path)
    
    def get_retriever(self) -> EmbeddingRetriever:
        """Get the retriever instance."""
        return self.retriever
    
    def has_documents(self) -> bool:
        """Check if the document store contains any documents."""
        return self.document_store.get_document_count() > 0
    
    def cleanup(self):
        """Clean up resources used by the document store."""
        if hasattr(self, 'document_store'):
            self.document_store.delete_documents()
        
        # Remove database files if they exist
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
