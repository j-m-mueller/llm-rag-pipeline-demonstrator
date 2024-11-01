"""main.py -- Main entry point for document query pipeline application.
Handles command line arguments, document loading, and pipeline execution with logging support."""

import argparse
import logging
import os

from dotenv import load_dotenv

from src.document_store import DocumentStoreManager
from src.pipeline import QueryPipeline
from src.preprocessing import load_documents, preprocess_documents


def main():
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # load environment variables
    load_dotenv()
    
    # verify API key is present
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # CLI arguments
    parser = argparse.ArgumentParser(description="Document Query Pipeline")
    parser.add_argument("--doc_dir", required=True, help="Directory containing documents")
    parser.add_argument("--query", help="Query to run against the documents")
    parser.add_argument("--embedding_model", 
                        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                        help="Name of the embedding model to use")
    parser.add_argument("--llm_model", 
                        default="gpt-3.5-turbo",
                        help="Name of the LLM model to use")
    
    args = parser.parse_args()
    
    # load and preprocess documents
    logging.info("Loading documents...")
    documents = load_documents(args.doc_dir)
    processed_docs = preprocess_documents(documents)
    
    # initialize document store and add documents
    logging.info("Initializing document store and generating embeddings...")
    doc_store_manager = DocumentStoreManager(embedding_model=args.embedding_model)
    doc_store_manager.add_documents(processed_docs)
    
    # initialize pipeline
    logging.info("Initializing query pipeline...")
    pipeline = QueryPipeline(
        retriever=doc_store_manager.get_retriever(),
        model_name=args.llm_model
    )
    
    # run query if provided
    if args.query:
        logging.info("Running query...")
        result = pipeline.run(args.query)
        logging.info("\nQuery Result:")
        logging.info(result["answers"][0].answer)


if __name__ == "__main__":
    main()
