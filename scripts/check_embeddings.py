import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import embed_documents, embed_query


def main():
    documents = [
        "passage: RAG использует поиск по документам.",
        "passage: Эмбеддинги представляют текст в виде числовых векторов.",
    ]

    vectors = embed_documents(documents)
    query_vector = embed_query("Что такое RAG?")

    print(f"Количество векторов документов: {len(vectors)}")
    print(f"Размерность вектора документа: {len(vectors[0])}")
    print(f"Размерность вектора вопроса: {len(query_vector)}")
    print("Проверка эмбеддингов завершена успешно.")


if __name__ == "__main__":
    main()
