from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    location: str


class AskOut(BaseModel):
    answer: str
    sources: list[Source]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    status: str
    chunk_count: int
    error: str | None
    created_at: datetime
