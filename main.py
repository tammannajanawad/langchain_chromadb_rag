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
  - langchain-openai (ChatOpenAI, OpenAIEmbeddings)
  - langchain-community (TextLoader, Chroma)
  - langchain-text-splitters, langchain-core
  - chromadb, python-dotenv
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env (e.g. OPENAI_API_KEY) so the OpenAI
# client can authenticate without hardcoding secrets.
load_dotenv()

# ---------------------------------------------------------------------------
# LLM and text splitter
# ---------------------------------------------------------------------------
# Chat model used for the final answer. temperature=0 for deterministic output.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Splits long documents into smaller chunks so we can embed and retrieve them.
# Uses tiktoken (gpt-4o-mini's tokenizer) so chunk_size is in tokens, not chars.
# Overlap helps avoid cutting sentences in the middle and preserves context.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",
    chunk_size=800,
    chunk_overlap=200,
)

# Path to the source document. __file__ ensures it works from any working dir.
transcript_path = os.path.join(
    os.path.dirname(__file__),
    "biden's_state_of_union_speech.txt",
)

# ---------------------------------------------------------------------------
# Vector store: load existing index or build from the transcript
# ---------------------------------------------------------------------------
# If chroma_db/ exists, we reuse it so we don't re-embed the whole document.
if os.path.exists("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings()
    )
else:
    # First run: load text, split into chunks, embed with OpenAI, persist to chroma_db/.
    documents = TextLoader(transcript_path, encoding="utf-8").load()
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="chroma_db",
    )

# ---------------------------------------------------------------------------
# Retriever: how we fetch relevant chunks for each query
# ---------------------------------------------------------------------------
# MMR = Maximal Marginal Relevance: balances similarity and diversity so we
# don't get near-duplicate chunks. k=3 chunks returned, fetch_k=10 candidates.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# ---------------------------------------------------------------------------
# RAG prompt: instructs the LLM to answer only from the provided context
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Demo: run a retriever-only lookup (no LLM)
# ---------------------------------------------------------------------------
# Shows the raw top retrieved chunk. Useful to see what the retriever returns.
query = "What is the main topic of the speech?"
docs = retriever.invoke(query)
print(docs[0].page_content)

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

# ---------------------------------------------------------------------------
# RAG chain: retriever + formatter + prompt + LLM + parser (LCEL)
# ---------------------------------------------------------------------------
# LCEL composes runnables with |. For each query: (1) retriever fetches docs,
# (2) format_docs turns them into one context string, (3) question is passed
# through, (4) prompt fills {context} and {question}, (5) LLM generates answer,
# (6) StrOutputParser returns a plain string.
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()
)

# Print the chain as an ASCII graph so we can see the flow.
rag_chain.get_graph().print_ascii()

# Run the full RAG pipeline and print the final answer.
result = rag_chain.invoke("What is the main topic of the speech?")
print(result)