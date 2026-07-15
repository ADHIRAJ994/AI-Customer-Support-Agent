"""
FastAPI entrypoint for the AI Customer Support Agent.

Endpoints:
  POST /documents/upload  -> upload a PDF, chunk + embed + index it
  GET  /documents         -> list indexed documents
  POST /chat               -> ask a question (multi-turn via thread_id)
  POST /feedback           -> submit thumbs up/down on an answer
  GET  /feedback/stats     -> aggregate feedback counts
  GET  /health              -> liveness check
"""
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    UploadResponse,
    DocumentsListResponse,
    DocumentInfo,
)
from app.ingestion import load_and_chunk_pdf
from app.vectorestore import add_documents, list_indexed_documents, load_or_create_vectorstore
from app.graph import run_turn, reset_thread
from app.feedback import init_db, save_feedback, get_stats

app = FastAPI(title="AI Customer Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vectorstore_dir, exist_ok=True)
    init_db()
    load_or_create_vectorstore()  # warm the index on boot


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest_path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex[:8]}_{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id, chunks = load_and_chunk_pdf(dest_path, file.filename)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF")

    add_documents(chunks)

    return UploadResponse(doc_id=doc_id, filename=file.filename, chunks_indexed=len(chunks))


@app.get("/documents", response_model=DocumentsListResponse)
def list_documents():
    docs = list_indexed_documents()
    return DocumentsListResponse(
        documents=[
            DocumentInfo(doc_id=d["doc_id"], filename=d["filename"], chunks=d["chunks"])
            for d in docs.values()
        ]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    answer, citations, _ = run_turn(req.thread_id, req.question)
    message_id = str(uuid.uuid4())

    return ChatResponse(
        message_id=message_id,
        thread_id=req.thread_id,
        answer=answer,
        citations=citations,
    )


@app.post("/chat/reset")
def chat_reset(thread_id: str):
    reset_thread(thread_id)
    return {"status": "reset", "thread_id": thread_id}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    save_feedback(req.message_id, req.thread_id, req.rating, req.comment)
    return {"status": "recorded"}


@app.get("/feedback/stats")
def feedback_stats():
    return get_stats()