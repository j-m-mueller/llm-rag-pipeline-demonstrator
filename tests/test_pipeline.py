"""src.tests.test_pipeline.py -- Test pipeline functionality."""

import pytest

from haystack.schema import Document
from typing import List

from src.pipeline.pipeline import QueryPipeline
from src.pipeline.document_store import DocumentStoreManager


def test_pipeline_initialization(doc_store: DocumentStoreManager) -> None:
    """
    Test pipeline initialization.

    :param doc_store: Document store instance
    """
    pipeline = QueryPipeline(retriever=doc_store.get_retriever())
    assert pipeline.retriever is not None
    assert pipeline.prompt_node is not None
    assert pipeline.pipeline is not None


def test_pipeline_run(doc_store: DocumentStoreManager, test_docs: List[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test pipeline execution.

    :param doc_store: Document store instance
    :param test_docs: Test documents
    :param monkeypatch: Pytest monkeypatch fixture
    """
    # mock the PromptNode response
    def mock_run(*args, **kwargs):
        return {
            "answers": [{"answer": "Test answer"}]
        }
    
    doc_store.add_documents(test_docs)
    pipeline = QueryPipeline(retriever=doc_store.get_retriever())
    monkeypatch.setattr(pipeline.pipeline, "run", mock_run)
    
    result = pipeline.run("test query")
    assert "answers" in result
    assert result["answers"][0]["answer"] == "Test answer"
