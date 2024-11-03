"""src.dashboard.app.py -- Streamlit dashboard for document query pipeline.
Provides a web interface for document upload, query input, and result visualization."""

import os
import sys

from pathlib import Path

sys.path.insert(0, os.path.abspath(Path().parent.parent))

import shutil
import streamlit as st
import tempfile

from dotenv import load_dotenv
from src.pipeline.document_store import DocumentStoreManager
from src.pipeline.pipeline import QueryPipeline
from src.pipeline.preprocessing import load_documents, preprocess_documents


def load_css():
    """Load custom CSS styling."""
    with open("./src/dashboard/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize streamlit session state variables."""
    if 'doc_store_manager' not in st.session_state:
        st.session_state.doc_store_manager = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None


def process_uploaded_documents(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile],
                               embedding_model: str,
                               llm_model: str):
    """
    Process uploaded documents and initialize the pipeline.
    
    :param uploaded_files: List of uploaded file objects from Streamlit
    :param embedding_model: Name of the embedding model to use
    :param llm_model: Name of the language model to use
    """
    with st.spinner("Processing documents..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            # save uploaded files to temporary directory
            for uploaded_file in uploaded_files:
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
            
            # process documents
            documents = load_documents(temp_dir)
            processed_docs = preprocess_documents(documents)
            
            # initialize document store
            st.session_state.doc_store_manager = DocumentStoreManager(
                embedding_model=embedding_model,
                db_path="data/faiss_document_store.db",
                index_path="data/faiss_document_store.faiss"
            )
            st.session_state.doc_store_manager.add_documents(processed_docs)
            
            # initialize pipeline
            st.session_state.pipeline = QueryPipeline(
                retriever=st.session_state.doc_store_manager.get_retriever(),
                model_name=llm_model
            )
        
        st.success("Documents processed successfully!")


def main():
    # page configuration and styling
    st.set_page_config(
        page_title="Document Query Pipeline",
        page_icon="🔍",
        layout="wide"
    )
    load_css()
    
    # API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in environment variables")
        st.stop()
    
    initialize_session_state()
    
    st.title("Document Query Pipeline")
    
    # sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/multi-qa-mpnet-base-dot-v1"],
            index=0
        )
        
        llm_model = st.selectbox(
            "Language Model",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0  # default to gpt-4o-mini
        )
        
        # file uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx']
        )
        
        if uploaded_files and st.button("Process Documents"):
            process_uploaded_documents(uploaded_files, embedding_model, llm_model)
    
    # main content area
    st.header("Query Documents")
    
    # check if pipeline is initialized
    if st.session_state.pipeline is None:
        st.warning("Please upload and process documents first.")
        st.stop()
    
    # query handling
    query = st.text_area("Enter your query:", height=100)
    
    if st.button("Send Query"):
        if not query:
            st.error("Please enter a query.")
            st.stop()
        
        with st.spinner("Processing query..."):
            result = st.session_state.pipeline.run(query)
            
            # display results
            st.subheader("Response:")
            
            if "answers" in result and result["answers"]:
                if hasattr(result["answers"][0], "answer"):
                    st.write(result["answers"][0].answer)
                else:
                    st.write(result["answers"][0].get("answer", "No answer generated."))
            
            # display source documents
            with st.expander("Source Documents"):
                for doc in result["documents"]:
                    st.markdown(f"**Source:** {doc.meta.get('file_path', 'Unknown')}")
                    st.markdown(f"**Content:** {doc.content[:500]}...")
                    st.markdown("---")


if __name__ == "__main__":
    main()
