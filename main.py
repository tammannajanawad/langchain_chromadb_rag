"""
LangChain + ChromaDB RAG demo.

Builds a retrieval-augmented generation (RAG) pipeline that answers questions
over a source document (e.g. a speech transcript) by retrieving relevant chunks
and passing them to an LLM.

Pipeline steps:
  1. Load a source text file (e.g. speech transcript) via TextLoader.
  2. Split it into chunks with overlap using RecursiveCharacterTextSplitter
     (token-aware via tiktoken, chunk_size=800, chunk_overlap=200).
  3. Embed chunks with OpenAI and store them in ChromaDB (persisted to
     chroma_db/; reuses existing DB if present).
  4. Run queries using a retriever (MMR search, k=3, fetch_k=10) + LLM via a
     LangChain LCEL chain: retriever → format_docs → prompt → LLM → StrOutputParser.

Usage:
  Set OPENAI_API_KEY in .env, then run:
    python main.py

  The script loads "biden's_state_of_union_speech.txt" (or reuses chroma_db/ if
  it exists), runs a retriever lookup and a full RAG query, prints the first
  retrieved chunk, the chain graph (ASCII), and the final RAG answer.

Dependencies:
  - openai (ChatOpenAI, OpenAIEmbeddings)
  - langchain-community (TextLoader, Chroma)
  - langchain-openai, langchain-text-splitters, langchain-core
  - chromadb, python-dotenv
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
# documents = TextLoader(transcript_path, encoding="utf-8").load()
# splits = splitter.split_documents(documents)

# ---------------------------------------------------------------------------
# Embed chunks and store in ChromaDB
# ---------------------------------------------------------------------------
if os.path.exists("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings()
    )
else:
    documents = TextLoader(transcript_path, encoding="utf-8").load()
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="chroma_db",
    )# ---------------------------------------------------------------------------
# Retriever and QA chain
# RetrievalQA is from langchain_classic.chains (not langchain_community.retrievers).
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant answering questions about a speech.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)

# Similarity search only (no LLM)
query = "What is the main topic of the speech?"
# docs = vectorstore.similarity_search(query)
docs = retriever.invoke(query)
print(docs[0].page_content)

# Full RAG: retriever + LLM via RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     chain_type="stuff",
#     retriever=retriever,
# )
# result = qa_chain.invoke("What is the main topic of the speech?")
# print(result["result"])
def format_docs(docs):
    """Format a list of LangChain documents into a single string for the prompt.

    Each document is prefixed with '[Source N]' (1-based index) followed by
    its page_content. Documents are joined with double newlines.

    Args:
        docs: List of documents with .page_content (e.g. from the retriever).

    Returns:
        A single string suitable for the {context} placeholder in the RAG prompt.
    """
    return "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()
)
rag_chain.get_graph().print_ascii()
result = rag_chain.invoke("What is the main topic of the speech?")
print(result)