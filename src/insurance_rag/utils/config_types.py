from pydantic import BaseModel, Field


class VectorstoreConfig(BaseModel):
    embedding_model: str

    collection_name: str

    vector_db_path: str

    chunks_file: str

class RetrievalConfig(BaseModel):

    dense_top_k: int = Field(gt=0)

    final_top_k: int = Field(gt=0)

    reranker_model: str
    guardrail_minimum_score: float = 0.30


class EvaluationConfig(BaseModel):
    model: str

    threshold: float = Field(
        ge=0,
        le=1,
    )

class GenerationConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0, le=2)
    prompt_name: str
    prompt_label: str
    input_cost_per_1m: float
    output_cost_per_1m: float


class DeepEvalConfig(BaseModel):
    max_concurrent: int = Field(gt=0)

    throttle_value: float = Field(ge=0)

    run_async: bool

class DatasetConfig(BaseModel):
    path: str
    safety_golden_path: str


class AppConfig(BaseModel):
    vectorstore: VectorstoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    deepeval: DeepEvalConfig
    dataset: DatasetConfig