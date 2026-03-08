"""
LangChain + ChromaDB RAG demo.

Builds a retrieval-augmented pipeline that:
  1. Loads a source text file (e.g. speech transcript).
  2. Splits it into chunks with overlap (RecursiveCharacterTextSplitter).
  3. Embeds chunks via OpenAI and stores them in ChromaDB.
  4. Runs queries using a retriever + LLM (RetrievalQA from langchain_classic).

Requires OPENAI_API_KEY in .env.
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# LLM and text splitter
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",
    chunk_size=800,
    chunk_overlap=200,
)

# ---------------------------------------------------------------------------
# Load document and split into chunks
# ---------------------------------------------------------------------------
transcript_path = os.path.join(
    os.path.dirname(__file__),
    "biden's_state_of_union_speech.txt",
)
documents = TextLoader(transcript_path, encoding="utf-8").load()
splits = splitter.split_documents(documents)

# ---------------------------------------------------------------------------
# Embed chunks and store in ChromaDB
# ---------------------------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="chroma_db",
)

# ---------------------------------------------------------------------------
# Retriever and QA chain
# RetrievalQA is from langchain_classic.chains (not langchain_community.retrievers).
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# Similarity search only (no LLM)
query = "What is the main topic of the speech?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

# Full RAG: retriever + LLM via RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
)
result = qa_chain.invoke("What is the main topic of the speech?")
print(result["result"])
