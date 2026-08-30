"""
PDF RAG pipeline built with LangGraph.

Ingests a PDF into a local FAISS vector store, then answers questions against
it through a two-node LangGraph graph (retrieve -> generate). Every run is
traced to Langfuse via observability.traced_run(), and can optionally be
scored by the LLM-as-judge evaluator in observability.evaluate_answer().

Usage:
    python rag_graph.py ingest ./data/sample.pdf
    python rag_graph.py ask "What is the termination clause?"
    python rag_graph.py ask "What is the termination clause?" --evaluate
"""

import os
import sys
from typing import List, TypedDict

from dotenv import load_dotenv

load_dotenv()

from loguru import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from observability import evaluate_answer, flush_langfuse, traced_run

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
# Open-source, runs locally via sentence-transformers — no API key/cost.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ANSWER_PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say so plainly.

Context:
{context}

Question: {question}

Answer:"""


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_pdf(pdf_path: str, store_dir: str = VECTOR_STORE_DIR) -> int:
    """Load a PDF, split it into chunks, embed, and persist a FAISS index."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(PyPDFLoader(pdf_path).load())

    FAISS.from_documents(chunks, _embeddings()).save_local(store_dir)
    logger.info(f"[rag] Ingested {pdf_path} -> {len(chunks)} chunks -> {store_dir}")
    return len(chunks)


def _load_store(store_dir: str = VECTOR_STORE_DIR) -> FAISS:
    if not os.path.isdir(store_dir):
        raise FileNotFoundError(f"No vector store at {store_dir}. Run `ingest` first.")
    return FAISS.load_local(store_dir, _embeddings(), allow_dangerous_deserialization=True)


# ── LangGraph state & nodes ──────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def build_graph(store_dir: str = VECTOR_STORE_DIR):
    store = _load_store(store_dir)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    def retrieve(state: RAGState) -> dict:
        return {"context": store.similarity_search(state["question"], k=TOP_K)}

    def generate(state: RAGState) -> dict:
        context_text = "\n\n".join(d.page_content for d in state["context"])
        prompt = ANSWER_PROMPT.format(context=context_text, question=state["question"])
        return {"answer": llm.invoke(prompt).content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str, store_dir: str = VECTOR_STORE_DIR, run_evaluation: bool = False) -> dict:
    """Run the RAG graph for one question, traced end-to-end in Langfuse."""
    app = build_graph(store_dir)

    with traced_run(trace_name="rag-langfuse-project-query", tags=["rag_langfuse_project", "pdf-rag"]) as (config, trace_id):
        result = app.invoke({"question": question, "context": [], "answer": ""}, config=config)

    flush_langfuse()

    context_text = "\n\n".join(d.page_content for d in result["context"])
    response = {
        "question": question,
        "answer": result["answer"],
        "source_pages": [d.metadata.get("page") for d in result["context"]],
        "trace_id": trace_id,
    }

    if run_evaluation:
        response["evaluation"] = evaluate_answer(question, result["answer"], context_text, trace_id=trace_id)

    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command, args = sys.argv[1], sys.argv[2:]

    if command == "ingest":
        pdf_path = args[0] if args else os.getenv("PDF_PATH")
        if not pdf_path:
            print('Usage: python rag_graph.py ingest <pdf_path>')
            sys.exit(1)
        n = ingest_pdf(pdf_path)
        print(f"Ingested {n} chunks into {VECTOR_STORE_DIR}")

    elif command == "ask":
        do_eval = "--evaluate" in args
        question = " ".join(a for a in args if a != "--evaluate")
        if not question:
            print('Usage: python rag_graph.py ask "<question>" [--evaluate]')
            sys.exit(1)
        result = ask(question, run_evaluation=do_eval)
        print(f"\nQ: {result['question']}\nA: {result['answer']}\nSources (pages): {result['source_pages']}")
        if do_eval:
            print(f"Evaluation: {result['evaluation']}")

    else:
        print(__doc__)
        sys.exit(1)
