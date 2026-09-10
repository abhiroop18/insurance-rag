import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import cohere

from insurance_rag.utils.config import load_config


class InsuranceRetriever:

    def __init__(
        self,
        cohere_api_key: str | None = None,
    ):

        self.config = load_config()

        # ============================================================
        # COHERE API KEY
        # ============================================================

        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not set."
            )

        # ============================================================
        # EMBEDDING MODEL
        # ============================================================

        self.embedding_model = SentenceTransformer(
            self.config.vectorstore.embedding_model
        )

        # ============================================================
        # QDRANT
        # ============================================================

        self.qdrant = QdrantClient(
            url="http://localhost:6333"
        )

        # ============================================================
        # COHERE
        # ============================================================

        self.cohere = cohere.ClientV2(
            api_key=cohere_api_key
        )

    # ============================================================
    # DENSE RETRIEVAL
    # ============================================================

    def dense_search(
        self,
        query: str,
        top_k: int | None = None,
    ):

        if top_k is None:
            top_k = self.config.retrieval.dense_top_k

        query_vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.qdrant.query_points(
            collection_name=(
                self.config.vectorstore.collection_name
            ),
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return results

    # ============================================================
    # RERANK
    # ============================================================

    def rerank(
        self,
        query: str,
        results,
        top_k: int | None = None,
    ):

        if not results:
            return []

        if top_k is None:
            top_k = self.config.retrieval.final_top_k

        documents = [
            point.payload["page_content"]
            for point in results
        ]

        response = self.cohere.rerank(
            model=self.config.retrieval.reranker_model,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        return [
            {
                "score": result.relevance_score,
                "page_content": (
                    results[result.index]
                    .payload["page_content"]
                ),
                "metadata": (
                    results[result.index]
                    .payload.get("metadata", {})
                ),
            }
            for result in response.results
        ]

    # ============================================================
    # FULL RETRIEVAL PIPELINE
    # ============================================================

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        final_top_k: int | None = None,
    ):

        if dense_top_k is None:
            dense_top_k = (
                self.config.retrieval.dense_top_k
            )

        if final_top_k is None:
            final_top_k = (
                self.config.retrieval.final_top_k
            )

        dense_results = self.dense_search(
            query=query,
            top_k=dense_top_k,
        )

        return self.rerank(
            query=query,
            results=dense_results,
            top_k=final_top_k,
        )