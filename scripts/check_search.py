
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import embed_query
from app.vector_store import VectorStore


def main() -> None:
    store = VectorStore()

    questions = [
        "Что такое RAG и как он работает?",
        "Что такое эмбеддинги и зачем они нужны?",
        "Как правильно разбивать документы на чанки?",
        "Как оценивают качество RAG-системы?",
        "Как приготовить борщ?",
    ]

    for question in questions:
        results = store.search(embed_query(question), top_k=3)

        print("\n" + "=" * 70)
        print(f"Вопрос: {question}")

        if not results:
            print("Подходящие фрагменты не найдены.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\nРезультат {index}:")
            print(f"  Источник: {result['source']}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Текст: {result['text'][:180]}")


if __name__ == "__main__":
    main()
