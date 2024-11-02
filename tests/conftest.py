"""Test fixtures and configuration."""

import os
import pytest
from pathlib import Path
from typing import List

from haystack.schema import Document
from src.pipeline.document_store import DocumentStoreManager


@pytest.fixture
def test_docs() -> List[Document]:
    """Create test documents."""
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
def doc_store() -> DocumentStoreManager:
    """Create a test document store."""
    store = DocumentStoreManager(
        db_path="test_document_store.db",
        index_path="test_document_store.faiss"
    )
    yield store
    store.cleanup()


@pytest.fixture
def test_dir(tmp_path) -> Path:
    """Create a temporary directory with test files."""
    # Create test directory
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()
    
    # Create test files
    (test_dir / "test1.txt").write_text("This is a test document.")
    (test_dir / "test2.txt").write_text("This is another test document.")
    
    return test_dir