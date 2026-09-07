from insurance_rag.guardrails.input_guardrail import (
    InputGuardrail,
)

from insurance_rag.guardrails.retrieval_guardrail import (
    RetrievalGuardrail,
)

from insurance_rag.guardrails.output_guardrail import (
    OutputGuardrail,
)

from insurance_rag.workflow.state import RAGState


INPUT_FALLBACK = (
    "I'm sorry, but I can only help with questions "
    "related to the insurance policy."
)

RETRIEVAL_FALLBACK = (
    "I'm sorry, but I couldn't find relevant information "
    "in the policy documents to answer that question."
)

OUTPUT_FALLBACK = (
    "I'm sorry, but I couldn't verify the generated answer "
    "against the insurance policy documents."
)


class GuardrailNodes:

    def __init__(self):

        self.input_guardrail = InputGuardrail()

        self.retrieval_guardrail = RetrievalGuardrail()

        self.output_guardrail = OutputGuardrail()

    # ============================================================
    # INPUT GUARDRAIL
    # ============================================================

    def validate_input(
        self,
        state: RAGState,
    ) -> RAGState:

        question = state["question"]

        result = self.input_guardrail.validate(
            question
        )

        passed = result.get(
            "passed",
            False,
        )

        sanitized_question = result.get(
            "sanitized_question",
            question,
        )

        base_state = {
            **state,

            "input_guardrail_executed": True,

            "question": question,

            "sanitized_question": sanitized_question,

            "input_guardrail_passed": passed,

            "input_guardrail_reason": result.get(
                "reason",
                "blocked" if not passed else "passed",
            ),

            "input_scope_adherent": result.get(
                "scope_adherent",
                False,
            ),

            "input_prompt_injection": result.get(
                "prompt_injection",
                False,
            ),

            "input_pii_detected": result.get(
                "pii_detected",
                False,
            ),
        }

        # =========================================================
        # PASSED
        # =========================================================

        if passed:

            return {
                **base_state,
                "blocked": False,
                "blocked_at": None,
            }

        # =========================================================
        # BLOCKED
        # =========================================================

        return {
            **base_state,

            "blocked": True,

            "blocked_at": "input_guardrail",

            "final_answer": (
                "I'm sorry, but I can only help with "
                "questions related to the insurance policy."
            ),
        }
    # ============================================================
    # RETRIEVAL GUARDRAIL
    # ============================================================

    def validate_retrieval(
        self,
        state: RAGState,
    ) -> RAGState:

        question = state.get(
            "sanitized_question",
            state["question"],
        )

        retrieval_results = state.get(
            "retrieval_results",
            [],
        )

        result = self.retrieval_guardrail.validate(
            question=question,
            retrieval_results=retrieval_results,
        )

        # =========================================================
        # BOOLEAN RESULT
        # =========================================================

        if isinstance(result, bool):

            if result:

                return {
                    **state,

                    "retrieval_guardrail_executed": True,

                    "retrieval_guardrail_passed": True,

                    "retrieval_guardrail_reason": "passed",

                    "retrieval_relevance_passed": True,

                    "retrieved_context_injection": False,

                    "retrieval_guardrail_top_score": None,

                    "retrieval_guardrail_relevance_scores": [],

                    "blocked": False,

                    "blocked_at": None,
                }

            return {
                **state,

                "retrieval_guardrail_executed": True,

                "retrieval_guardrail_passed": False,

                "retrieval_guardrail_reason": "blocked",

                "retrieval_relevance_passed": False,

                "retrieved_context_injection": False,

                "retrieval_guardrail_top_score": None,

                "retrieval_guardrail_relevance_scores": [],

                "blocked": True,

                "blocked_at": "retrieval_guardrail",

                "final_answer": (
                    "I'm sorry, but I couldn't find "
                    "relevant information in the policy "
                    "documents to answer that question."
                ),
            }

        # =========================================================
        # STRUCTURED RESULT
        # =========================================================

        passed = result.get(
            "passed",
            False,
        )

        top_score = result.get(
            "top_score"
        )

        relevance_scores = result.get(
            "relevance_scores",
            [],
        )

        context_injection = result.get(
            "context_injection_detected",
            False,
        )

        update = {
            **state,

            "retrieval_guardrail_executed": True,

            "retrieval_guardrail_passed": passed,

            "retrieval_guardrail_reason": result.get(
                "reason",
                "blocked",
            ),

            "retrieval_relevance_passed": result.get(
                "relevance_passed",
                False,
            ),

            "retrieved_context_injection": context_injection,

            "retrieval_guardrail_top_score": top_score,

            "retrieval_guardrail_relevance_scores": (
                relevance_scores
            ),
        }

        # =========================================================
        # PASSED
        # =========================================================

        if passed:

            return {
                **update,

                "blocked": False,

                "blocked_at": None,
            }

        # =========================================================
        # BLOCKED
        # =========================================================

        return {
            **update,

            "blocked": True,

            "blocked_at": "retrieval_guardrail",

            "final_answer": (
                "I'm sorry, but I couldn't find "
                "relevant information in the policy "
                "documents to answer that question."
            ),
        }
    # ============================================================
    # OUTPUT GUARDRAIL
    # ============================================================

    def validate_output(
        self,
        state: RAGState,
    ) -> RAGState:

        question = state.get(
            "sanitized_question",
            state["question"],
        )

        answer = state.get(
            "answer",
            "",
        )

        contexts = state.get(
            "retrieved_context",
            [],
        )

        result = self.output_guardrail.validate(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        # =========================================================
        # GUARDRAIL EXECUTED
        # =========================================================

        base_state = {
            **state,

            "output_guardrail_executed": True,

            "output_guardrail_reason": result.get(
                "reason",
                "passed",
            ),

            "output_answer_relevant": result.get(
                "answer_relevant",
                False,
            ),

            "output_answer_grounded": result.get(
                "answer_grounded",
                False,
            ),

            "output_unsupported_claims": result.get(
                "unsupported_claims",
                [],
            ),
        }

        # =========================================================
        # PASSED
        # =========================================================

        if result.get("passed", False):

            return {
                **base_state,

                "output_guardrail_passed": True,

                "blocked": False,

                "blocked_at": None,

                "final_answer": answer,
            }

        # =========================================================
        # FAILED
        # =========================================================

        return {
            **base_state,

            "output_guardrail_passed": False,

            "blocked": True,

            "blocked_at": "output_guardrail",

            "final_answer": (
                "I'm sorry, but I couldn't verify "
                "the generated answer against the "
                "insurance policy documents."
            ),
        }