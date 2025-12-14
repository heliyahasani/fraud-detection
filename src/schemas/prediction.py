"""Schema for model predictions."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Model prediction output."""

    is_fraud: int = Field(ge=0, le=1)
    fraud_probability: float = Field(ge=0, le=1)

    model_config = {"extra": "forbid"}
