from insurance_rag.retrieval.retriever import (
    InsuranceRetriever,
)

from insurance_rag.generation.generator import (
    InsuranceGenerator,
)

from insurance_rag.workflow.state import RAGState


class RAGNodes:

    def __init__(
        self,
        retriever: InsuranceRetriever,
        generator: InsuranceGenerator,
    ):

        self.retriever = retriever
        self.generator = generator

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        state: RAGState,
    ) -> RAGState:

        question = state.get(
            "sanitized_question",
            state["question"],
        )

        results = self.retriever.retrieve(
            query=question
        )

        if results is None:
            results = []

        contexts = []
        retrieval_scores = []

        for result in results:

            if not isinstance(result, dict):
                continue

            page_content = result.get(
                "page_content",
                "",
            )

            if page_content:
                contexts.append(
                    str(page_content)
                )

            score = result.get("score")

            if score is not None:

                try:
                    retrieval_scores.append(
                        float(score)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return {
            **state,

            "retrieval_executed": True,

            "retrieval_results": results,

            "retrieved_documents": results,

            "retrieved_context": contexts,

            "retrieval_scores": retrieval_scores,
        }

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        state: RAGState,
    ) -> RAGState:

        question = state.get(
            "sanitized_question",
            state["question"],
        )

        contexts = state.get(
            "retrieved_context",
            [],
        )

        result = self.generator.generate(
            question=question,
            contexts=contexts,
        )

        return {
            **state,

            "generation_executed": True,

            "answer": result["answer"],

            "model": result["model"],

            "input_tokens": result["input_tokens"],

            "output_tokens": result["output_tokens"],

            "total_tokens": result["total_tokens"],
        }