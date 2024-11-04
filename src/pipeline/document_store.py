"""src.pipeline.document_store.py -- Implements DocumentStoreManager class for managing FAISS document store."""

import os
from pathlib import Path
from typing import List, Optional

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document


class DocumentStoreManager:
    """
    Manages FAISS document store operations including initialization, document addition, and cleanup.
    
    :param embedding_model: Name or path of the embedding model to use for document embeddings
    :param db_path: Path to the SQLite database file, defaults to data/faiss_document_store.db
    :param index_path: Path to the FAISS index file, defaults to data/faiss_document_store.faiss
    """

    def __init__(self, 
                 embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 db_path: Optional[str] = None,
                 index_path: Optional[str] = None,
                 clean_start: bool = False):
        """
        Initialize the DocumentStoreManager with FAISS document store and embedding retriever.
        
        :param embedding_model: Name or path of the embedding model to use
        :param db_path: Optional path to the SQLite database file
        :param index_path: Optional path to the FAISS index file
        :param clean_start: Flag to control file deletion
        """
        # create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # use provided paths or default to data directory
        self.db_path = db_path or str(data_dir / "faiss_document_store.db")
        self.index_path = index_path or str(data_dir / "faiss_document_store.faiss")
        
        # Only delete existing files if clean_start is True
        if clean_start:
            self._cleanup_existing_files()
        
        # Initialize document store based on whether index exists
        if os.path.exists(self.index_path):
            # If index exists, load it
            self.document_store = FAISSDocumentStore.load(
                index_path=self.index_path
            )
        else:
            # If no index exists, create new store
            self.document_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{self.db_path}",
                return_embedding=True,
                embedding_dim=768
            )
        
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model
        )
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the document store and generate embeddings.
        
        :param documents: List of Document objects to add to the store
        """
        self.document_store.write_documents(documents)
        self.document_store.update_embeddings(self.retriever)
        # save the updated index
        self.document_store.save(self.index_path)
    
    def get_retriever(self) -> EmbeddingRetriever:
        """
        Get the retriever instance.
        
        :return: Configured EmbeddingRetriever instance
        """
        return self.retriever
    
    def has_documents(self) -> bool:
        """
        Check if the document store contains any documents.
        
        :return: True if documents exist in store, False otherwise
        """
        return self.document_store.get_document_count() > 0
    
    def _cleanup_existing_files(self):
        """
        Clean up resources used by the document store.
        Deletes all documents and removes database files.
        """
        if hasattr(self, 'document_store'):
            self.document_store.delete_documents()
        
        # remove database files if they exist
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
