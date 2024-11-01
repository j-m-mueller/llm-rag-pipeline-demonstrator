from pathlib import Path
from typing import List
from haystack.nodes import PreProcessor
from haystack.schema import Document

def load_documents(doc_dir: str) -> List[Document]:
    """
    Load documents from a specified directory.

    :param doc_dir: Path to the directory containing documents
    :return: List of loaded Document objects
    """
    documents = []
    doc_dir = Path(doc_dir)
    
    for file_path in doc_dir.glob("**/*.*"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(
                    Document(
                        content=content,
                        meta={"file_path": str(file_path)}
                    )
                )
    
    return documents

def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Preprocess documents into chunks.

    :param documents: List of Document objects to process
    :return: List of processed and chunked Document objects
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