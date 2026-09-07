
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from guardrails import Guard
from guardrails.classes import ValidationOutcome
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from guardrails_ai.guardrails_pii import GuardrailsPII


load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

PII_ENTITIES = [
    "PERSON",
    "LOCATION",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "PHONE_NUMBER",
]


# =============================================================================
# CUSTOM JAILBREAK / PROMPT INJECTION VALIDATOR
# =============================================================================

@register_validator(
    name="insurance_jailbreak",
    data_type="string",
)
class InsuranceJailbreakValidator(Validator):
    """
    Detects common jailbreak and prompt-injection attempts.

    This is intentionally a lightweight rule-based validator.

    It does NOT use Guardrails' DetectJailbreak package, so it does not
    depend on its Hugging Face model.
    """

    def __init__(
        self,
        on_fail: str | None = None,
    ):

        super().__init__(
            on_fail=on_fail,
        )

        # Common prompt-injection patterns.
        self.patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+(all\s+)?prior\s+instructions",
            r"disregard\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(all\s+)?prior\s+instructions",
            r"forget\s+(all\s+)?previous\s+instructions",
            r"forget\s+(all\s+)?prior\s+instructions",

            r"reveal\s+(your\s+)?system\s+prompt",
            r"show\s+(me\s+)?your\s+system\s+prompt",
            r"give\s+me\s+(your\s+)?system\s+prompt",
            r"print\s+(your\s+)?system\s+prompt",

            r"reveal\s+your\s+instructions",
            r"show\s+your\s+instructions",
            r"reveal\s+hidden\s+instructions",
            r"show\s+hidden\s+instructions",

            r"ignore\s+your\s+safety",
            r"bypass\s+(your\s+)?safety",
            r"bypass\s+the\s+guardrail",
            r"bypass\s+guardrails",

            r"jailbreak",
            r"developer\s+message",
            r"system\s+message",

            r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
            r"pretend\s+you\s+have\s+no\s+restrictions",
            r"do\s+anything\s+now",

            r"ignore\s+the\s+rules",
            r"disregard\s+the\s+rules",

            r"tell\s+me\s+the\s+hidden\s+prompt",
            r"tell\s+me\s+your\s+hidden\s+instructions",
        ]

        self.compiled_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in self.patterns
        ]

    # =========================================================================
    # VALIDATE
    # =========================================================================

    def _validate(
        self,
        value: str,
        metadata: dict,
    ) -> ValidationResult:

        if not value or not value.strip():

            return FailResult(
                error_message="Question is empty."
            )

        matched_patterns = []

        for pattern, compiled in zip(
            self.patterns,
            self.compiled_patterns,
        ):

            if compiled.search(value):

                matched_patterns.append(
                    pattern
                )

        # ---------------------------------------------------------------------
        # Jailbreak detected
        # ---------------------------------------------------------------------

        if matched_patterns:

            return FailResult(
                error_message=(
                    "Potential prompt injection detected. "
                    f"Matched patterns: "
                    f"{', '.join(matched_patterns)}"
                )
            )

        # ---------------------------------------------------------------------
        # Safe
        # ---------------------------------------------------------------------

        return PassResult()


# =============================================================================
# CUSTOM INSURANCE TOPIC VALIDATOR
# =============================================================================

@register_validator(
    name="insurance_topic",
    data_type="string",
)
class InsuranceTopicValidator(Validator):
    """
    Checks whether a user question is related to insurance.

    Uses an LLM as the topic classifier.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        on_fail: str | None = None,
    ):

        super().__init__(
            on_fail=on_fail,
            threshold=threshold,
        )

        self.threshold = threshold

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:

            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = "gpt-5-mini"

    # =========================================================================
    # VALIDATE
    # =========================================================================

    def _validate(
        self,
        value: str,
        metadata: dict,
    ) -> ValidationResult:

        if not value or not value.strip():

            return FailResult(
                error_message="Question is empty."
            )

        system_prompt = """
You are an insurance topic classifier.

Determine whether the user's question is related to
insurance or an insurance policy.

ON-TOPIC examples:

- What is the waiting period for cataract surgery?
- Does this policy cover hospitalization?
- What are the exclusions?
- How do I make a claim?
- What is the sum insured?
- Is cataract surgery covered?
- What is the room rent limit?
- Does the policy cover pre-existing diseases?
- What is the premium?
- What is the co-payment?

The question does NOT need to contain the word "insurance".

For example:

"What is the waiting period for cataract surgery?"
=> ON-TOPIC

"Is cataract surgery covered?"
=> ON-TOPIC

"What is the weather in Bangalore?"
=> OFF-TOPIC

"Write a Python program."
=> OFF-TOPIC

"Who won yesterday's cricket match?"
=> OFF-TOPIC

Return ONLY valid JSON:

{
    "is_on_topic": true,
    "score": 0.95,
    "reason": "short explanation"
}

Rules:

- score must be between 0.0 and 1.0.
- 0.8 to 1.0 means strongly insurance-related.
- 0.0 to 0.4 means clearly unrelated.
"""

        user_prompt = f"""
USER QUESTION:
{value}
"""

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

        except Exception as exc:

            return FailResult(
                error_message=(
                    "Insurance topic classification failed: "
                    f"{exc}"
                )
            )

        raw_output = (
            response.choices[0]
            .message
            .content
        )

        if not raw_output:

            return FailResult(
                error_message=(
                    "Insurance topic classifier returned "
                    "an empty response."
                )
            )

        try:

            result = json.loads(
                raw_output
            )

        except json.JSONDecodeError:

            return FailResult(
                error_message=(
                    "Insurance topic classifier returned "
                    "invalid JSON."
                )
            )

        try:

            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        is_on_topic = bool(
            result.get(
                "is_on_topic",
                False,
            )
        )

        reason = result.get(
            "reason",
            "",
        )

        # ---------------------------------------------------------------------
        # PASS
        # ---------------------------------------------------------------------

        if (
            is_on_topic
            and score >= self.threshold
        ):

            return PassResult()

        # ---------------------------------------------------------------------
        # FAIL
        # ---------------------------------------------------------------------

        return FailResult(
            error_message=(
                "Question is outside the scope of "
                "the insurance assistant. "
                f"Score: {score:.2f}. "
                f"Reason: {reason}"
            )
        )


# =============================================================================
# INPUT GUARDRAIL
# =============================================================================

class InputGuardrail:
    """
    Input guardrail for the insurance RAG system.

    Checks:

    1. Empty question
    2. Jailbreak / prompt injection
    3. PII
    4. Insurance topic
    """

    def __init__(self):

        self.guard = Guard().use(

            # -----------------------------------------------------------------
            # 1. Custom jailbreak detection
            # -----------------------------------------------------------------

            InsuranceJailbreakValidator(
                on_fail="exception",
            ),

            # -----------------------------------------------------------------
            # 2. PII detection
            #
            # PII is redacted rather than blocking the user.
            # -----------------------------------------------------------------

            GuardrailsPII(
                entities=PII_ENTITIES,
                on_fail="fix",
            ),

            # -----------------------------------------------------------------
            # 3. Insurance topic
            # -----------------------------------------------------------------

            InsuranceTopicValidator(
                threshold=0.70,
                on_fail="exception",
            ),
        )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    def validate(
        self,
        question: str,
    ) -> dict[str, Any]:

        # ---------------------------------------------------------------------
        # Empty question
        # ---------------------------------------------------------------------

        if not question or not question.strip():

            return {
                "passed": False,
                "reason": "empty_question",
                "scope_adherent": False,
                "prompt_injection": False,
                "pii_detected": False,
                "sanitized_question": "",
            }

        # ---------------------------------------------------------------------
        # Run Guardrails
        # ---------------------------------------------------------------------

        try:

            outcome: ValidationOutcome = (
                self.guard.validate(
                    question
                )
            )

        except Exception as exc:

            error_message = str(exc)

            error_lower = (
                error_message.lower()
            )

            # ---------------------------------------------------------------
            # Jailbreak / prompt injection
            # ---------------------------------------------------------------

            if (
                "prompt injection"
                in error_lower
                or "jailbreak"
                in error_lower
                or "ignore"
                in error_lower
                and "instruction"
                in error_lower
            ):

                return {
                    "passed": False,
                    "reason": "prompt_injection_detected",
                    "scope_adherent": False,
                    "prompt_injection": True,
                    "pii_detected": False,
                    "sanitized_question": question,
                }

            # ---------------------------------------------------------------
            # Insurance topic failure
            # ---------------------------------------------------------------

            if (
                "outside the scope"
                in error_lower
            ):

                return {
                    "passed": False,
                    "reason": "off_topic",
                    "scope_adherent": False,
                    "prompt_injection": False,
                    "pii_detected": False,
                    "sanitized_question": question,
                }

            # ---------------------------------------------------------------
            # Other error
            # ---------------------------------------------------------------

            return {
                "passed": False,
                "reason": (
                    f"input_guardrail_error: {exc}"
                ),
                "scope_adherent": False,
                "prompt_injection": False,
                "pii_detected": False,
                "sanitized_question": question,
            }

        # ---------------------------------------------------------------------
        # Get sanitized output
        # ---------------------------------------------------------------------

        sanitized_question = (
            outcome.validated_output
            if outcome.validated_output is not None
            else question
        )

        pii_detected = (
            sanitized_question != question
        )

        # ---------------------------------------------------------------------
        # Guardrail failed
        # ---------------------------------------------------------------------

        if not outcome.validation_passed:

            error = (
                str(outcome.error)
                if outcome.error
                else "input_validation_failed"
            )

            return {
                "passed": False,
                "reason": error,
                "scope_adherent": False,
                "prompt_injection": False,
                "pii_detected": pii_detected,
                "sanitized_question": sanitized_question,
            }

        # ---------------------------------------------------------------------
        # Guardrail passed
        # ---------------------------------------------------------------------

        return {
            "passed": True,
            "reason": (
                "passed_with_pii_redaction"
                if pii_detected
                else "passed"
            ),
            "scope_adherent": True,
            "prompt_injection": False,
            "pii_detected": pii_detected,
            "sanitized_question": sanitized_question,
        }

