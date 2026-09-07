from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore

from insurance_rag.utils.config import load_config



config = load_config()


# =============================================================================
# LOAD CHUNKS
# =============================================================================

def load_documents(
    chunks_path: str,
) -> list[Document]:

    chunks_path = Path(chunks_path)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    with open(
        chunks_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data
    ]


# =============================================================================
# BUILD EMBEDDING MODEL
# =============================================================================

def build_embeddings():

    return HuggingFaceEmbeddings(
        model_name=config.vectorstore.embedding_model,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# =============================================================================
# CREATE QDRANT CLIENT
# =============================================================================

def build_qdrant_client():

    return QdrantClient(
        url="http://localhost:6333"
    )


# =============================================================================
# CREATE COLLECTION
# =============================================================================

def create_collection(
    client,
    embeddings,
):

    test_vector = embeddings.embed_query("test")

    vector_size = len(test_vector)

    existing_collections = (
        client.get_collections()
    )

    collection_names = [
        collection.name
        for collection in existing_collections.collections
    ]

    collection_name = (
        config.vectorstore.collection_name
    )

    if collection_name in collection_names:

        client.delete_collection(
            collection_name=collection_name
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


# =============================================================================
# BUILD VECTORSTORE
# =============================================================================

def build_vectorstore(
    documents,
    embeddings,
    client,
):

    collection_name = (
        config.vectorstore.collection_name
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    vectorstore.add_documents(
        documents
    )

    return vectorstore


# =============================================================================
# MAIN
# =============================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    chunks_path = (
        project_root
        / config.vectorstore.chunks_file
    )

    documents = load_documents(
        str(chunks_path)
    )

    embeddings = build_embeddings()

    client = build_qdrant_client()

    create_collection(
        client=client,
        embeddings=embeddings,
    )

    build_vectorstore(
        documents=documents,
        embeddings=embeddings,
        client=client,
    )


if __name__ == "__main__":
    main()