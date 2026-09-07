from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Insurance policy question",
    )


class QuestionResponse(BaseModel):

    answer: str