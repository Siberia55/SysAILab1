
from fastapi import FastAPI

from app.rag_service import RAGService
from app.schemas import AskRequest, AskResponse


app = FastAPI(
    title="RAG Assistant",
    description="Поиск информации в учебных документах об искусственном интеллекте",
    version="1.0.0",
)

rag_service = RAGService()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "RAG Assistant работает",
        "docs": "/docs",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    result = rag_service.ask(request.question)
    return AskResponse(**result)
