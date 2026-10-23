from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Загружает модель эмбеддингов один раз."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Преобразует фрагменты документов в векторы.

    Для модели E5 используется префикс passage:.
    """
    if not texts:
        return []

    model = get_embedding_model()
    prepared_texts = [f"passage: {text}" for text in texts]

    embeddings = model.encode(
        prepared_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Преобразует пользовательский вопрос в вектор.

    Для модели E5 используется префикс query:.
    """
    model = get_embedding_model()

    embedding = model.encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.tolist()
