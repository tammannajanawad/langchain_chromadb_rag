# LangChain + ChromaDB RAG

A minimal **Retrieval-Augmented Generation (RAG)** demo that indexes a long text document (e.g. a speech transcript), stores embeddings in ChromaDB, and answers questions using LangChain, OpenAI embeddings, and an LLM.

## What it does

1. **Load** a source text file (e.g. a speech transcript).
2. **Split** it into chunks with overlap using `RecursiveCharacterTextSplitter` (tiktoken-based).
3. **Embed** chunks with OpenAI and **store** them in ChromaDB.
4. **Query** via a retriever (MMR) and an LLM using `RetrievalQA`.

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

The script will load the speech file, build the ChromaDB index under `chroma_db/`, run a similarity search and a RetrievalQA query, and print results.

## Project structure

```
.
├── main.py              # RAG pipeline (load → split → embed → store → query)
├── chroma_db/           # ChromaDB data (created on first run)
├── .env                 # OPENAI_API_KEY (not committed)
├── pyproject.toml       # Dependencies and project config
└── README.md
```

## Dependencies

Key packages: `langchain`, `langchain-community`, `langchain-classic`, `langchain-openai`, `langchain-text-splitters`, `chromadb`, `openai`, `dotenv`. See `pyproject.toml` for versions.
