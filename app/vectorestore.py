import os
import threading
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings

_lock = threading.Lock()
_vectorstore: FAISS | None = None


def get_embeddings() -> Embeddings:
    """
    Use OpenAI embeddings if an OpenAI key is configured, otherwise fall
    back to a free local sentence-transformers model (works with Groq,
    which has no embeddings endpoint of its own).
    """
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _index_path() -> str:
    os.makedirs(settings.vectorstore_dir, exist_ok=True)
    return settings.vectorstore_dir


def load_or_create_vectorstore() -> FAISS:
    global _vectorstore
    with _lock:
        if _vectorstore is not None:
            return _vectorstore

        embeddings = get_embeddings()
        path = _index_path()
        index_file = os.path.join(path, "index.faiss")

        if os.path.exists(index_file):
            _vectorstore = FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            # Bootstrap an empty index with a throwaway document, then clear it,
            # since FAISS.from_documents requires at least one document.
            placeholder = Document(page_content="placeholder", metadata={"bootstrap": True})
            _vectorstore = FAISS.from_documents([placeholder], embeddings)
            _vectorstore.delete([_vectorstore.index_to_docstore_id[0]])
            _vectorstore.save_local(path)
        return _vectorstore


def save_vectorstore() -> None:
    if _vectorstore is not None:
        _vectorstore.save_local(_index_path())


def add_documents(docs: list[Document]) -> list[str]:
    vs = load_or_create_vectorstore()
    with _lock:
        ids = vs.add_documents(docs)
        save_vectorstore()
    return ids


def get_retriever(k: int | None = None):
    vs = load_or_create_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k or settings.retriever_k})


def list_indexed_documents() -> dict[str, dict]:
    """Aggregate chunk counts per doc_id from the docstore metadata."""
    vs = load_or_create_vectorstore()
    docs_by_id: dict[str, dict] = {}
    for _id, doc in vs.docstore._dict.items():  # type: ignore[attr-defined]
        meta = doc.metadata
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        entry = docs_by_id.setdefault(
            doc_id, {"doc_id": doc_id, "filename": meta.get("source", "unknown"), "chunks": 0}
        )
        entry["chunks"] += 1
    return docs_by_id