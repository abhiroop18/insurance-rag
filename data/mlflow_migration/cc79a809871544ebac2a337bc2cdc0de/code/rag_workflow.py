
import os
from typing import Any

from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from insurance_rag.workflow.state import RAGState

from insurance_rag.workflow.nodes import (
    RAGNodes,
)

from insurance_rag.guardrails.nodes import (
    GuardrailNodes,
)

from insurance_rag.retrieval.retriever import (
    InsuranceRetriever,
)

from insurance_rag.generation.generator import (
    InsuranceGenerator,
)


load_dotenv()


# ============================================================
# FALLBACKS
# ============================================================

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


# ============================================================
# ROUTER: INPUT GUARDRAIL
# ============================================================

def route_after_input_guardrail(
    state: RAGState,
) -> str:

    if state.get(
        "input_guardrail_passed",
        False,
    ):

        return "retrieval"

    return "input_block"


# ============================================================
# INPUT BLOCK
# ============================================================

def input_block_node(
    state: RAGState,
) -> RAGState:

    return {
        **state,

        "blocked": True,

        "blocked_at": "input_guardrail",

        "final_answer": INPUT_FALLBACK,

    }


# ============================================================
# ROUTER: RETRIEVAL GUARDRAIL
# ============================================================

def route_after_retrieval_guardrail(
    state: RAGState,
) -> str:

    if state.get(
        "retrieval_guardrail_passed",
        False,
    ):

        return "generation"

    return "retrieval_block"


# ============================================================
# RETRIEVAL BLOCK
# ============================================================

def retrieval_block_node(
    state: RAGState,
) -> RAGState:

    return {
        **state,

        "blocked": True,

        "blocked_at": "retrieval_guardrail",

        "final_answer": RETRIEVAL_FALLBACK,

    }


# ============================================================
# CREATE GRAPH
# ============================================================

def create_rag_graph():

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    cohere_api_key = os.getenv(
        "COHERE_API_KEY"
    )

    if not cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY is not set."
        )

    retriever = InsuranceRetriever(
        cohere_api_key=cohere_api_key
    )

    generator = InsuranceGenerator()

    # ========================================================
    # NODE CLASSES
    # ========================================================

    rag_nodes = RAGNodes(
        retriever=retriever,
        generator=generator,
    )

    guardrail_nodes = GuardrailNodes()

    # ========================================================
    # GRAPH
    # ========================================================

    workflow = StateGraph(RAGState)

    # ========================================================
    # GUARDRAILS
    # ========================================================

    workflow.add_node(
        "input_guardrail",
        guardrail_nodes.validate_input,
    )

    workflow.add_node(
        "retrieval_guardrail",
        guardrail_nodes.validate_retrieval,
    )

    workflow.add_node(
        "output_guardrail",
        guardrail_nodes.validate_output,
    )

    # ========================================================
    # RAG NODES
    # ========================================================

    workflow.add_node(
        "retrieval",
        rag_nodes.retrieve,
    )

    workflow.add_node(
        "generation",
        rag_nodes.generate,
    )

    # ========================================================
    # BLOCK NODES
    # ========================================================

    workflow.add_node(
        "input_block",
        input_block_node,
    )

    workflow.add_node(
        "retrieval_block",
        retrieval_block_node,
    )

    # ========================================================
    # START
    # ========================================================

    workflow.add_edge(
        START,
        "input_guardrail",
    )

    # ========================================================
    # INPUT GUARDRAIL ROUTING
    # ========================================================

    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "retrieval": "retrieval",
            "input_block": "input_block",
        },
    )

    # ========================================================
    # INPUT BLOCK
    # ========================================================

    workflow.add_edge(
        "input_block",
        END,
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    workflow.add_edge(
        "retrieval",
        "retrieval_guardrail",
    )

    # ========================================================
    # RETRIEVAL GUARDRAIL ROUTING
    # ========================================================

    workflow.add_conditional_edges(
        "retrieval_guardrail",
        route_after_retrieval_guardrail,
        {
            "generation": "generation",
            "retrieval_block": "retrieval_block",
        },
    )

    # ========================================================
    # RETRIEVAL BLOCK
    # ========================================================

    workflow.add_edge(
        "retrieval_block",
        END,
    )

    # ========================================================
    # GENERATION
    # ========================================================

    workflow.add_edge(
        "generation",
        "output_guardrail",
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    workflow.add_edge(
        "output_guardrail",
        END,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    graph = workflow.compile()

    return graph

