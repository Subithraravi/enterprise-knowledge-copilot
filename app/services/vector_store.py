import os
import uuid

from qdrant_client import QdrantClient, models

from app.services.embedding_service import (
    generate_embeddings,
    generate_query_embedding,
)


COLLECTION_NAME = "enterprise_documents"
VECTOR_SIZE = 384


def get_qdrant_client() -> QdrantClient:
    """
    Create a persistent local Qdrant client.
    """
    storage_path = os.getenv(
        "QDRANT_PATH",
        "./qdrant_storage",
    )

    os.makedirs(storage_path, exist_ok=True)

    return QdrantClient(path=storage_path)


def ensure_collection(client: QdrantClient) -> None:
    """
    Create the collection if it does not already exist.
    """
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def store_document_chunks(
    chunks: list[str],
    filename: str,
) -> int:
    """
    Embed document chunks and save them in Qdrant.
    """
    if not chunks:
        return 0

    client = get_qdrant_client()

    try:
        ensure_collection(client)

        vectors = generate_embeddings(chunks)

        points = []

        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filename": filename,
                        "text": chunk,
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        return len(points)

    finally:
        client.close()


def search_relevant_chunks(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    client = get_qdrant_client()

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return []

        query_vector = generate_query_embedding(question)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_chunks = []

        for result in results:
            payload = result.payload or {}

            retrieved_chunks.append(
                {
                    "filename": payload.get(
                        "filename",
                        "Unknown document",
                    ),
                    "text": payload.get("text", ""),
                    "score": float(result.score),
                }
            )

        return retrieved_chunks

    finally:
        client.close()