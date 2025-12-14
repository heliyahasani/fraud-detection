"""Schema for engineered features."""

from pydantic import BaseModel, Field


class Features(BaseModel):
    """Engineered features for model training/inference."""

    # Numerical
    amt: float = Field(gt=0)
    log_amt: float
    city_pop: int = Field(ge=0)
    age: int = Field(ge=0, le=100)
    distance: float = Field(ge=0)

    # Time
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)

    # Binary flags
    is_night: int = Field(ge=0, le=1)
    is_online: int = Field(ge=0, le=1)

    # One-hot encoded risk labels (drop_first=True, so 'high' is baseline)
    cat_risk_low: int = Field(ge=0, le=1, default=0)
    cat_risk_medium: int = Field(ge=0, le=1, default=0)
    state_risk_low: int = Field(ge=0, le=1, default=0)
    state_risk_medium: int = Field(ge=0, le=1, default=0)
    merch_risk_low: int = Field(ge=0, le=1, default=0)
    merch_risk_medium: int = Field(ge=0, le=1, default=0)

    # Demographics
    gender_M: int = Field(ge=0, le=1, default=0)

    model_config = {"extra": "forbid"}

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return list of feature names in order."""
        return [
            "amt",
            "log_amt",
            "city_pop",
            "age",
            "distance",
            "hour",
            "day_of_week",
            "month",
            "is_night",
            "is_online",
            "cat_risk_low",
            "cat_risk_medium",
            "state_risk_low",
            "state_risk_medium",
            "merch_risk_low",
            "merch_risk_medium",
            "gender_M",
        ]
