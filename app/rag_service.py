
from app.config import MODEL_VERSION, SIMILARITY_THRESHOLD, TOP_K
from app.embeddings import embed_query
from app.vector_store import VectorStore


FALLBACK_ANSWER = "Не нашёл информации по вашему вопросу"


class RAGService:
    """Сервис поиска информации по документам."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()

    def ask(self, question: str) -> dict:
        # Преобразуем вопрос в вектор той же моделью,
        # которая использовалась для индексации документов.
        query_embedding = embed_query(question)

        # Ищем наиболее близкие фрагменты в ChromaDB.
        results = self.vector_store.search(
            query_embedding,
            top_k=TOP_K,
        )

        # Оставляем только фрагменты, прошедшие порог.
        relevant_results = [
            result
            for result in results
            if result["score"] >= SIMILARITY_THRESHOLD
        ]

        # Если релевантной информации нет, не пытаемся
        # придумывать ответ.
        if not relevant_results:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "model_version": MODEL_VERSION,
            }

        # Используем максимум два лучших фрагмента.
        # Ответ состоит только из текста найденных документов.
        selected_results = relevant_results[:2]

        answer = "\n\n".join(
            result["text"] for result in selected_results
        )

        sources = [
            {
                "source": result["source"],
                "chunk_id": result["chunk_id"],
                "score": result["score"],
                "distance": result["distance"],
            }
            for result in selected_results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "model_version": MODEL_VERSION,
        }
