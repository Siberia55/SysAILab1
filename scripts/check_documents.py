import sys
from pathlib import Path

# Добавляем корневую папку проекта в пути поиска Python.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from app.document_loader import load_documents
from app.chunker import split_text


def main():
    documents = load_documents(DATA_DIR)

    print(f"Загружено документов: {len(documents)}")

    total_chunks = 0

    for document in documents:
        chunks = split_text(document["text"])
        total_chunks += len(chunks)

        print(
            f'{document["source"]}: '
            f'{len(chunks)} чанков'
        )

        for index, chunk in enumerate(chunks, start=1):
            token_count = len(
                __import__("app.chunker", fromlist=["get_tokenizer"])
                .get_tokenizer()
                .encode(chunk, add_special_tokens=False)
            )

            if token_count > CHUNK_SIZE:
                raise AssertionError(
                    f"Чанк {index} превышает размер: "
                    f"{token_count} токенов"
                )

    print(f"\nВсего чанков: {total_chunks}")
    print(f"Размер чанка: {CHUNK_SIZE} токенов")
    print(f"Перекрытие: {CHUNK_OVERLAP} токенов")
    print("Проверка завершена успешно.")


if __name__ == "__main__":
    main()
