"""main.py -- Main entry point for document query pipeline application.
Handles command line arguments, document loading, and pipeline execution with logging support."""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from src.pipeline.document_store import DocumentStoreManager
from src.pipeline.pipeline import QueryPipeline
from src.pipeline.preprocessing import load_documents, preprocess_documents


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
    parser.add_argument("--doc_dir", help="Directory containing documents")
    parser.add_argument("--query", required=True, help="Query to run against the documents")
    parser.add_argument("--embedding_model", 
                        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                        help="Name of the embedding model to use")
    parser.add_argument("--llm_model", 
                        default="gpt-4o-mini",
                        help="Name of the LLM model to use")
    
    args = parser.parse_args()
    
    doc_store_manager = DocumentStoreManager(
        embedding_model=args.embedding_model
    )
    
    if args.doc_dir:
        # load and preprocess new documents
        logging.info("Loading documents...")
        documents = load_documents(args.doc_dir)
        processed_docs = preprocess_documents(documents)
        
        # initialize document store and add documents
        logging.info("Initializing document store and generating embeddings...")
        doc_store_manager.add_documents(processed_docs)
    else:
        # check if existing document store has documents
        if not doc_store_manager.has_documents():
            logging.error("No documents found in store. Please provide --doc_dir for initial setup")
            sys.exit(1)
        logging.info("Using existing document store...")
    
    # initialize pipeline
    logging.info("Initializing query pipeline...")
    pipeline = QueryPipeline(
        retriever=doc_store_manager.get_retriever(),
        model_name=args.llm_model
    )
    
    # run query
    logging.info("Running query...")
    result = pipeline.run(args.query)
    logging.info("\nQuery Result:")
    
    # handle response more robustly
    if "answers" in result and result["answers"]:
        if hasattr(result["answers"][0], "answer"):
            logging.info(result["answers"][0].answer)
        else:
            logging.info(result["answers"][0].get("answer", "No answer generated."))
    else:
        logging.info(result.get("results", ["No answer generated."])[0])


if __name__ == "__main__":
    main()
