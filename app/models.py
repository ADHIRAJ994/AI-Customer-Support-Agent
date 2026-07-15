from typing import Optional, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    snippet: str


class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation/session identifier")
    question: str


class ChatResponse(BaseModel):
    message_id: str
    thread_id: str
    answer: str
    citations: list[Citation]


class FeedbackRequest(BaseModel):
    message_id: str
    thread_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class DocumentsListResponse(BaseModel):
    documents: list[DocumentInfo]