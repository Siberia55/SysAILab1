import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME
from app.embeddings import embed_documents


class VectorStore:
    """Обёртка над коллекцией ChromaDB."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Пересоздаёт коллекцию."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Сохраняет чанки, векторы и метаданные."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_documents(texts)

        self.collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["id"],
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        """Возвращает наиболее близкие к вопросу фрагменты."""
        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        found = []

        for index, text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][index]
            distance = results["distances"][0][index]

            found.append({
                "text": text,
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "distance": float(distance),
                "score": float(1.0 - distance),
            })

        return found