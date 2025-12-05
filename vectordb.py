"""Vector DB utilities for policy PDFs with local persistence.

This module:
- Reads one PDF containing company policies
- Chunks and embeds it into a FAISS vector store
- Saves the FAISS index locally the first time it is built
- On later runs, loads the saved index instead of rebuilding
- Exposes `query_vector_db` for agents/tools to retrieve passages
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PDF_PATH = Path("Sample Policies.pdf")
DEFAULT_INDEX_DIR = Path("policy_faiss_index")

# Global handle; populated lazily by `get_vectorstore`.
VECTORSTORE: FAISS | None = None


def _validate_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")


def load_pdf_text(pdf_path: Path) -> str:
    """Read text from every page with basic sanity checks."""
    _validate_pdf_path(pdf_path)
    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        raise ValueError(f"{pdf_path} does not contain readable pages.")

    text = "\n".join(
        (page.extract_text() or "").strip() for page in reader.pages
    ).strip()

    if not text:
        raise ValueError(f"No text extracted from {pdf_path}.")
    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split text into overlapping windows for embedding."""
    if not text:
        raise ValueError("Cannot chunk empty text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("Text splitter produced no chunks.")
    return chunks


def create_vector_db(
    pdf_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    """Create a FAISS store from a policy PDF (in-memory only)."""
    text = load_pdf_text(Path(pdf_path))
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    return FAISS.from_texts(chunks, embeddings)


def save_vector_db(vectorstore: FAISS, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    """Persist a FAISS vector store to a local directory."""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))


def load_vector_db(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
) -> FAISS:
    """Load a FAISS vector store from a local directory."""
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Vector index not found at {index_path}")

    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def init_vectorstore(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    """
    Initialize the global VECTORSTORE:
    - If a local FAISS index exists, load it.
    - Otherwise, build from the PDF once and persist.
    """
    global VECTORSTORE

    index_path = Path(index_dir)
    if index_path.exists():
        VECTORSTORE = load_vector_db(index_path, model_name=model_name)
    else:
        store = create_vector_db(
            pdf_path=pdf_path,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        save_vector_db(store, index_path)
        VECTORSTORE = store

    return VECTORSTORE


def get_vectorstore() -> FAISS:
    """
    Return the global vectorstore, initializing it on first use.
    """
    global VECTORSTORE
    if VECTORSTORE is None:
        VECTORSTORE = init_vectorstore()
    return VECTORSTORE


def query_vector_db(
    query: str,
    k: int = 5,
) -> Iterable[str]:
    """Return the top-k matching passages as plain strings from the stored index."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return (doc.page_content for doc in docs)


if __name__ == "__main__":
    # One-off manual test:
    print("Initializing vector store (this will build & cache if not present)...")
    init_vectorstore()
    test_query = "What is the policy on vacation?"
    print(f"\nQuery: {test_query}\n")
    for idx, answer in enumerate(query_vector_db(test_query, k=3), start=1):
        print(f"Result #{idx}:\n{answer}\n{'-' * 40}")


