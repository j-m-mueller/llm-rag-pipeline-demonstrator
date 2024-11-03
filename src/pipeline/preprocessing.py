"""src.pipeline.preprocessing.py -- Document loading and preprocessing utilities."""

from pathlib import Path
from typing import List
from haystack.nodes import PreProcessor
from haystack.schema import Document
from haystack.utils import convert_files_to_docs


def load_documents(doc_dir: str) -> List[Document]:
    """
    Load documents from a specified directory.
    
    :param doc_dir: Path to the directory containing documents
    :return: List of Document objects
    """
    documents = []
    doc_dir = Path(doc_dir)
    
    documents = convert_files_to_docs(doc_dir)
    
    return documents


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Preprocess documents into chunks.
    
    :param documents: List of documents to process
    :return: List of processed Document objects
    """
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_overlap=50
    )
    
    processed_docs = []
    for doc in documents:
        processed_docs.extend(preprocessor.process([doc]))
    
    return processed_docs
