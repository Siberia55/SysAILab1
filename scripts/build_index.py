import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import split_text
from app.config import DATA_DIR
from app.document_loader import load_documents
from app.vector_store import VectorStore


def main() -> None:
    documents = load_documents(DATA_DIR)

    if not documents:
        print(f"Документы не найдены в папке: {DATA_DIR}")
        return

    chunks = []

    for document in documents:
        document_chunks = split_text(document["text"])

        for index, chunk_text in enumerate(document_chunks):
            chunks.append({
                "id": f'{document["source"]}_chunk_{index}',
                "text": chunk_text,
                "source": document["source"],
            })

    if not chunks:
        print("Не удалось получить чанки из документов.")
        return

    store = VectorStore()
    store.reset()
    store.add_chunks(chunks)

    print("Индексация завершена успешно.")
    print(f"Документов обработано: {len(documents)}")
    print(f"Чанков сохранено: {len(chunks)}")
    print(f"Векторная база: {store.collection.name}")


if __name__ == "__main__":
    main()
