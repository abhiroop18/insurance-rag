# src/insurance_rag/guardrails/output_guardrail.py

from typing import Any, Dict, List

from guardrails import Guard
from guardrails.validators import (
    Validator,
    PassResult,
    FailResult,
    register_validator,
)


# =============================================================================
# OUTPUT RELEVANCE VALIDATOR
# =============================================================================

@register_validator(
    name="insurance-answer-relevant",
    data_type="string",
)
class InsuranceAnswerRelevant(Validator):

    def __init__(
        self,
        on_fail=None,
    ):
        super().__init__(
            on_fail=on_fail,
        )

    def validate(
        self,
        value: str,
        metadata: Dict,
    ):

        question = metadata.get(
            "question",
            "",
        )

        if not question:
            return FailResult(
                error_message=(
                    "Question was not provided "
                    "for answer relevance validation."
                )
            )

        if not value or not value.strip():
            return FailResult(
                error_message=(
                    "Generated answer is empty."
                )
            )

        # -------------------------------------------------------------
        # IMPORTANT
        #
        # We use an LLM to judge whether the answer actually answers
        # the user's question.
        # -------------------------------------------------------------

        from openai import OpenAI

        client = OpenAI()

        prompt = f"""
You are evaluating an insurance RAG system.

Determine whether the answer directly addresses the user's question.

USER QUESTION:
{question}

GENERATED ANSWER:
{value}

Return ONLY one word:

PASS
or
FAIL

PASS means the answer directly addresses the question.

FAIL means the answer is unrelated, avoids the question,
or answers a substantially different question.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = response.choices[0].message.content.strip()

        if result == "PASS":
            return PassResult()

        return FailResult(
            error_message=(
                "Generated answer is not relevant "
                "to the user's question."
            )
        )


# =============================================================================
# OUTPUT GROUNDEDNESS VALIDATOR
# =============================================================================

@register_validator(
    name="insurance-answer-grounded",
    data_type="string",
)
class InsuranceAnswerGrounded(Validator):

    def __init__(
        self,
        on_fail=None,
    ):
        super().__init__(
            on_fail=on_fail,
        )

    def validate(
        self,
        value: str,
        metadata: Dict,
    ):

        question = metadata.get(
            "question",
            "",
        )

        contexts = metadata.get(
            "contexts",
            [],
        )

        if not contexts:
            return FailResult(
                error_message=(
                    "No retrieved context was provided "
                    "for groundedness validation."
                )
            )

        context_text = "\n\n".join(
            str(context)
            for context in contexts
        )

        from openai import OpenAI

        client = OpenAI()

        prompt = f"""
You are evaluating an insurance RAG system.

Determine whether the generated answer is fully supported
by the retrieved insurance policy context.

USER QUESTION:
{question}

RETRIEVED POLICY CONTEXT:
{context_text}

GENERATED ANSWER:
{value}

Return ONLY one word:

PASS
or
FAIL

PASS means every important factual claim in the answer
is supported by the retrieved context.

FAIL means the answer contains one or more factual claims
that are not supported by the retrieved context.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = response.choices[0].message.content.strip()

        if result == "PASS":
            return PassResult()

        return FailResult(
            error_message=(
                "Generated answer contains claims "
                "that are not supported by the "
                "retrieved policy context."
            )
        )


# =============================================================================
# OUTPUT GUARDRAIL
# =============================================================================

class OutputGuardrail:

    def __init__(self):

        # ---------------------------------------------------------------------
        # Guardrails AI Guard
        #
        # Validators are executed in sequence.
        # ---------------------------------------------------------------------

        self.guard = Guard().use(
            InsuranceAnswerRelevant(
                on_fail="exception",
            )
        ).use(
            InsuranceAnswerGrounded(
                on_fail="exception",
            )
        )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    def validate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------------------
        # Basic validation
        # ---------------------------------------------------------------------

        if not answer or not answer.strip():

            return {
                "passed": False,
                "reason": "empty_answer",
                "answer_relevant": False,
                "answer_grounded": False,
                "unsupported_claims": [],
            }

        # ---------------------------------------------------------------------
        # Metadata passed to Guardrails validators
        # ---------------------------------------------------------------------

        metadata = {
            "question": question,
            "contexts": contexts,
        }

        # ---------------------------------------------------------------------
        # Run Guardrails
        # ---------------------------------------------------------------------

        try:

            result = self.guard.validate(
                answer,
                metadata=metadata,
            )

            # -------------------------------------------------------------
            # Guardrails passed
            # -------------------------------------------------------------

            return {
                "passed": True,
                "reason": "passed",
                "answer_relevant": True,
                "answer_grounded": True,
                "unsupported_claims": [],
            }

        except Exception as e:

            error_message = str(e)

            # -------------------------------------------------------------
            # Determine which validator failed
            # -------------------------------------------------------------

            if (
                "not relevant" in error_message.lower()
            ):

                return {
                    "passed": False,
                    "reason": "answer_not_relevant",
                    "answer_relevant": False,
                    "answer_grounded": False,
                    "unsupported_claims": [],
                }

            if (
                "not supported" in error_message.lower()
                or "grounded" in error_message.lower()
            ):

                return {
                    "passed": False,
                    "reason": "answer_not_grounded",
                    "answer_relevant": True,
                    "answer_grounded": False,
                    "unsupported_claims": [
                        error_message
                    ],
                }

            return {
                "passed": False,
                "reason": (
                    "output_validation_error: "
                    + error_message
                ),
                "answer_relevant": False,
                "answer_grounded": False,
                "unsupported_claims": [
                    error_message
                ],
            }