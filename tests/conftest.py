"""src.tests.conftest.py -- Test fixtures and configuration."""

import os
import sys
import time

from pathlib import Path

sys.path.insert(0, os.path.abspath(Path().parent))

import pytest

from haystack.schema import Document
from typing import List

from src.pipeline.document_store import DocumentStoreManager


@pytest.fixture
def test_docs() -> List[Document]:
    """
    Create test documents for testing purposes.

    :return: List of test documents
    """
    return [
        Document(
            content="This is a test document.",
            meta={"file_path": "test1.txt"}
        ),
        Document(
            content="This is another test document.",
            meta={"file_path": "test2.txt"}
        )
    ]


@pytest.fixture
def doc_store(tmp_path) -> DocumentStoreManager:
    """
    Create a test document store instance.

    :param tmp_path: Pytest fixture providing temporary directory
    :return: A configured document store manager
    """
    # Create unique paths for each test
    db_path = str(tmp_path / "test_document_store.db")
    index_path = str(tmp_path / "test_document_store.faiss")
    
    # Create fresh document store
    store = DocumentStoreManager(
        db_path=db_path,
        index_path=index_path,
        clean_start=True
    )
    
    yield store
    
    # Cleanup after test
    try:
        store._cleanup_existing_files()
    finally:
        # Force cleanup of files
        for file_path in [db_path, index_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except (PermissionError, OSError):
                pass


@pytest.fixture
def test_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory with test files.

    :param tmp_path: Pytest fixture providing temporary directory
    :return: Path to the created test directory
    """
    # Create test directory
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()
    
    # Create test files
    (test_dir / "test1.txt").write_text("This is a test document.")
    (test_dir / "test2.txt").write_text("This is another test document.")
    
    return test_dir
