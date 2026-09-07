import logging
import time
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from insurance_rag.workflow.rag_workflow import create_rag_graph

from langfuse import get_client
from langfuse.langchain import CallbackHandler


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "insurance_rag_api"
)


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Insurance RAG API",
    description="Insurance policy question-answering API",
    version="1.0.0",
)

langfuse = get_client()
# =============================================================================
# PROMETHEUS
# =============================================================================
#
# IMPORTANT:
# Instrument the application here.
# Do NOT put instrument() inside startup_event().
#
# Prometheus will scrape:
#
#     http://localhost:8000/metrics/prometheus
#
# =============================================================================

instrumentator = Instrumentator(
    excluded_handlers=[
        "/metrics/prometheus",
        "/docs",
        "/openapi.json",
    ],
)

instrumentator.instrument(app).expose(
    app,
    endpoint="/metrics/prometheus",
    include_in_schema=False,
)


# =============================================================================
# RAG PIPELINE
# =============================================================================

rag_pipeline = None


# =============================================================================
# REQUEST MODEL
# =============================================================================

class QuestionRequest(BaseModel):

    question: str


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
def startup_event():

    global rag_pipeline

    logger.info(
        "Initializing RAG pipeline..."
    )

    rag_pipeline = create_rag_graph()

    logger.info(
        "RAG pipeline initialized."
    )


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):

    # -------------------------------------------------------------------------
    # Generate request ID
    # -------------------------------------------------------------------------

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = request_id

    # -------------------------------------------------------------------------
    # Start timer
    # -------------------------------------------------------------------------

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Log request
    # -------------------------------------------------------------------------

    logger.info(
        "Request started | "
        f"request_id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path}"
    )

    # -------------------------------------------------------------------------
    # Execute request
    # -------------------------------------------------------------------------

    try:

        response = await call_next(
            request
        )

    except Exception:

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        logger.exception(
            "Request failed | "
            f"request_id={request_id} | "
            f"latency={elapsed_time:.4f}s"
        )

        raise

    # -------------------------------------------------------------------------
    # Calculate latency
    # -------------------------------------------------------------------------

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    # -------------------------------------------------------------------------
    # Add request ID to response
    # -------------------------------------------------------------------------

    response.headers[
        "X-Request-ID"
    ] = request_id

    # -------------------------------------------------------------------------
    # Log completed request
    # -------------------------------------------------------------------------

    logger.info(
        "Request completed | "
        f"request_id={request_id} | "
        f"status_code={response.status_code} | "
        f"latency={elapsed_time:.4f}s"
    )

    return response


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get(
    "/health",
    tags=["Health"],
)
def health():

    return {
        "status": "ok"
    }


# =============================================================================
# ASK
# =============================================================================

@app.post(
    "/ask",
    tags=["RAG"],
)
def ask(
    request: Request,
    body: QuestionRequest,
):

    if rag_pipeline is None:
        raise RuntimeError(
            "RAG pipeline has not been initialized."
        )

    langfuse_handler = CallbackHandler()

    result = rag_pipeline.invoke(
        {
            "question": body.question,
        },
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "request_id": request.state.request_id,
            },
        },
    )

    # ... your existing metrics code ...

    return {
        "answer": result.get(
            "final_answer",
            "",
        ),
        "blocked": result.get(
            "blocked",
            False,
        ),
        "blocked_at": result.get(
            "blocked_at"
        ),
        "request_id": request.state.request_id,
    }