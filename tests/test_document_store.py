"""src.tests.test_document_store.py -- Test document store functionality."""

from haystack.schema import Document
from typing import List

from src.pipeline.document_store import DocumentStoreManager



def test_document_store_initialization(doc_store: DocumentStoreManager) -> None:
    """
    Test document store initialization.

    :param doc_store: Document store instance
    """
    assert doc_store.document_store is not None
    assert doc_store.retriever is not None


def test_add_documents(doc_store: DocumentStoreManager, test_docs: List[Document]) -> None:
    """
    Test adding documents to store.

    :param doc_store: Document store instance
    :param test_docs: Documents to add
    """
    doc_store.add_documents(test_docs)
    assert doc_store.has_documents()
    assert doc_store.document_store.get_document_count() == len(test_docs)


def test_get_retriever(doc_store: DocumentStoreManager) -> None:
    """
    Test retriever access.

    :param doc_store: Document store instance
    """
    retriever = doc_store.get_retriever()
    assert retriever is not None
    assert retriever == doc_store.retriever


def test_cleanup(doc_store: DocumentStoreManager, test_docs: List[Document]) -> None:
    """
    Test cleanup functionality.

    :param doc_store: Document store instance
    :param test_docs: Test documents
    """
    doc_store.add_documents(test_docs)
    assert doc_store.has_documents()
    
    doc_store.cleanup()
    assert not doc_store.has_documents()
