"""
PDF ingestion: load, split into chunks, and attach metadata used later
for citations and source highlighting.
"""
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings


def load_and_chunk_pdf(file_path: str, original_filename: str) -> tuple[str, list[Document]]:
    """
    Returns (doc_id, chunks). Each chunk carries metadata:
      - doc_id: groups chunks belonging to the same uploaded file
      - source: original filename (shown in citations)
      - page: 1-indexed page number
      - chunk_id: unique id for this chunk
    """
    doc_id = str(uuid.uuid4())[:8]

    loader = PyPDFLoader(file_path)
    pages = loader.load()  # one Document per page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in pages:
        page_number = page.metadata.get("page", 0) + 1
        for split in splitter.split_text(page.page_content):
            if not split.strip():
                continue
            chunks.append(
                Document(
                    page_content=split,
                    metadata={
                        "doc_id": doc_id,
                        "source": original_filename,
                        "page": page_number,
                        "chunk_id": str(uuid.uuid4())[:8],
                    },
                )
            )
    return doc_id, chunks