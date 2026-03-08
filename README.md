# LangChain + ChromaDB RAG

A minimal **Retrieval-Augmented Generation (RAG)** demo that indexes a long text document (e.g. a speech transcript), stores embeddings in ChromaDB, and answers questions using LangChain, OpenAI embeddings, and an LLM.

Inspired by this [Kaggle notebook](https://www.kaggle.com/code/gpreda/rag-using-llama-2-langchain-and-chromadb/notebook).

## What it does

1. **Load** a source text file (e.g. a speech transcript) via `TextLoader`.
2. **Split** it into chunks with overlap using `RecursiveCharacterTextSplitter` (tiktoken-based, chunk_size=800, chunk_overlap=200).
3. **Embed** chunks with OpenAI and **store** them in ChromaDB (persisted under `chroma_db/`; reuses existing DB if present).
4. **Query** via an MMR retriever (k=3, fetch_k=10) and an LCEL RAG chain: retriever → `format_docs` → prompt → `ChatOpenAI` (gpt-4o-mini) → `StrOutputParser`.

The demo is set up for `biden's_state_of_union_speech.txt` in the project root. You can replace it with any `.txt` file and adjust the path in `main.py`.

## Prerequisites

- **Python 3.12+**
- **OpenAI API key** (for embeddings and the chat model)

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the pipeline:

```bash
uv run python main.py
```

Or activate the venv and run:

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python main.py
```

The script will load the speech file (or reuse the existing index in `chroma_db/`), run a retriever lookup and a full RAG query, then print the first retrieved chunk, the LCEL chain graph (ASCII), and the final RAG answer.

## Project structure

```
.
├── main.py                          # RAG pipeline (load → split → embed → store → query)
├── biden's_state_of_union_speech.txt # Default source document (optional)
├── chroma_db/                       # ChromaDB data (created on first run)
├── .env                             # OPENAI_API_KEY (not committed)
├── pyproject.toml                   # Dependencies and project config
└── README.md
```

## Dependencies

Key packages: `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langchain-core`, `chromadb`, `openai`, `python-dotenv`. See `pyproject.toml` for versions.
