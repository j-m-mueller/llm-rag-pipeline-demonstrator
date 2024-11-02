"""Test document store functionality."""

import pytest
from src.pipeline.document_store import DocumentStoreManager


def test_document_store_initialization(doc_store):
    """Test document store initialization."""
    assert doc_store.document_store is not None
    assert doc_store.retriever is not None


def test_add_documents(doc_store, test_docs):
    """Test adding documents to store."""
    doc_store.add_documents(test_docs)
    assert doc_store.has_documents()
    assert doc_store.document_store.get_document_count() == len(test_docs)


def test_get_retriever(doc_store):
    """Test retriever access."""
    retriever = doc_store.get_retriever()
    assert retriever is not None
    assert retriever == doc_store.retriever


def test_cleanup(doc_store, test_docs):
    """Test cleanup functionality."""
    doc_store.add_documents(test_docs)
    assert doc_store.has_documents()
    
    doc_store.cleanup()
    assert not doc_store.has_documents()