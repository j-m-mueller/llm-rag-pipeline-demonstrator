# Haystack Document Query Pipeline

This project implements a document query pipeline using Haystack. It allows you to load documents, process them, store them in a vector store, and query them using an LLM.

## Project Components

### Pipeline Module
- `preprocessing.py`: Document loading and chunking
- `document_store.py`: FAISS vector store management
- `pipeline.py`: Query pipeline implementation
- `main.py`: CLI interface

### Dashboard Module
- `app.py`: Streamlit web interface

## Dependencies

- farm-haystack: Document processing and pipeline
- sentence-transformers: Document embedding
- streamlit: Web interface
- pytest: Testing framework
- python-dotenv: Environment management
- OpenAI API: LLM integration

## Setup

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

Or create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key
```

## Usage

### Command Line Interface

Run the pipeline using the command line interface:

```bash
python -m src.pipeline.main --doc_dir /path/to/documents --query "Your query here"
```

Optional arguments:
- `--embedding_model`: Specify a different embedding model (default: sentence-transformers/multi-qa-mpnet-base-dot-v1)
- `--llm_model`: Specify a different LLM model (default: gpt-3.5-turbo)

### Dashboard Interface

Run the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard provides:
- Document upload interface
- Model selection
- Interactive query input
- Result visualization
- Source document inspection

## Features

- Document loading from specified directory
- Document preprocessing and chunking
- Vector store using FAISS
- Embedding generation using Sentence Transformers
- Query pipeline with retrieval and LLM-based answer generation
- Web-based dashboard interface
- Comprehensive test suite

## Development

### Running Tests

Run the test suite:

```bash
pytest tests/ -v
```

For test coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```
