"""src.tests.test_preprocessing.py -- Test preprocessing functionality."""

from haystack.schema import Document
from pathlib import Path
from typing import List

from src.pipeline.preprocessing import load_documents, preprocess_documents


def test_load_documents(test_dir: Path) -> None:
    """    
    Test document loading functionality from directory.

    :param test_dir: Directory containing test documents
    :return: None
    """
    documents = load_documents(str(test_dir))
    assert len(documents) == 2
    assert all(doc.content for doc in documents), "content attributes missing"  # check presence of content field for each document
    assert all(doc.meta.get("file_path") for doc in documents), "file path metadata missing"  # check if file path was set


def test_preprocess_documents(test_docs: List[Document]) -> None:
    """
    Test document preprocessing functionality.

    :param test_docs: Input test documents
    """
    processed_docs = preprocess_documents(test_docs)
    assert len(processed_docs) >= len(test_docs), "processed document count mismatch"
    assert all(doc.content for doc in processed_docs), "content attributes missing"  # check presence of content field for each processed document
    