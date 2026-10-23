
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Вопрос пользователя по документам",
        examples=["Что такое RAG?"],
    )


class SourceInfo(BaseModel):
    source: str
    chunk_id: str
    score: float
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    model_version: str
