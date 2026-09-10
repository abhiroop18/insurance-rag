from typing import TypedDict, List, Optional, Any


class RetrievedDocument(TypedDict, total=False):
    page_content: str
    score: float
    metadata: dict[str, Any]


class RAGState(TypedDict, total=False):

    # ============================================================
    # USER INPUT
    # ============================================================

    question: str
    sanitized_question: str

    # ============================================================
    # INPUT GUARDRAIL
    # ============================================================

    input_guardrail_executed: bool
    input_guardrail_passed: bool
    input_guardrail_reason: Optional[str]

    input_scope_adherent: bool
    input_prompt_injection: bool
    input_pii_detected: bool

    # ============================================================
    # RETRIEVAL
    # ============================================================

    retrieval_executed: bool

    retrieval_results: List[Any]
    retrieved_documents: List[RetrievedDocument]
    retrieved_context: List[str]

    retrieval_scores: List[float]

    # ============================================================
    # RETRIEVAL GUARDRAIL
    # ============================================================

    retrieval_guardrail_executed: bool
    retrieval_guardrail_passed: bool
    retrieval_guardrail_reason: Optional[str]

    retrieval_relevance_passed: bool
    retrieved_context_injection: bool

    retrieval_guardrail_top_score: Optional[float]
    retrieval_guardrail_relevance_scores: List[float]

    # ============================================================
    # GENERATION
    # ============================================================

    generation_executed: bool

    answer: str

    model: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    # ============================================================
    # OUTPUT GUARDRAIL
    # ============================================================

    output_guardrail_executed: bool
    output_guardrail_passed: bool
    output_guardrail_reason: Optional[str]

    output_answer_relevant: bool
    output_answer_grounded: bool

    output_unsupported_claims: List[str]

    # ============================================================
    # FINAL PIPELINE STATUS
    # ============================================================

    blocked: bool
    blocked_at: Optional[str]

    final_answer: str