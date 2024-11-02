"""Test preprocessing functionality."""

from pathlib import Path
from src.pipeline.preprocessing import load_documents, preprocess_documents


def test_load_documents(test_dir):
    """Test document loading from directory."""
    documents = load_documents(str(test_dir))
    assert len(documents) == 2
    assert all(doc.content for doc in documents)
    assert all(doc.meta.get("file_path") for doc in documents)


def test_preprocess_documents(test_docs):
    """Test document preprocessing."""
    processed_docs = preprocess_documents(test_docs)
    assert len(processed_docs) >= len(test_docs)
    assert all(doc.content for doc in processed_docs)
