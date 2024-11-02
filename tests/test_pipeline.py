"""Test pipeline functionality."""

import pytest
from src.pipeline.pipeline import QueryPipeline


def test_pipeline_initialization(doc_store):
    """Test pipeline initialization."""
    pipeline = QueryPipeline(retriever=doc_store.get_retriever())
    assert pipeline.retriever is not None
    assert pipeline.prompt_node is not None
    assert pipeline.pipeline is not None


def test_pipeline_run(doc_store, test_docs, monkeypatch):
    """Test pipeline execution."""
    # Mock the PromptNode response
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