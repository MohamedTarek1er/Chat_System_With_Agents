from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value



class Source(BaseModel):
    text: str
    doc_name: str
    page: Optional[int] = None
    chunk_id: str
    score: Optional[float] = None

class DeleteResponse(BaseModel):
    document_name: str
    chunks_deleted: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    evidence_complete: bool


class UploadResponse(BaseModel):
    document_name: str
    chunks_stored: int


class DocumentList(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: str


class VoiceQueryResponse(QueryResponse):
    transcript: str
    language: str
    duration_seconds: float    