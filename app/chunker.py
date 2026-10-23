from functools import lru_cache

from transformers import AutoTokenizer

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_tokenizer():
    """Загружает токенизатор один раз и переиспользует его."""
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Разбивает текст на чанки по количеству токенов.

    Перекрытие сохраняет часть предыдущего фрагмента
    в начале следующего.

    Возвращает список текстовых чанков.
    """
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше нуля.")

    if chunk_overlap < 0:
        raise ValueError("Перекрытие не может быть отрицательным.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Перекрытие должно быть меньше размера чанка."
        )

    text = text.strip()

    if not text:
        return []

    tokenizer = get_tokenizer()

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not token_ids:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(token_ids), step):
        token_chunk = token_ids[start:start + chunk_size]

        if not token_chunk:
            break

        chunk_text = tokenizer.decode(
            token_chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if start + chunk_size >= len(token_ids):
            break

    return chunks
