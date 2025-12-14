"""Schema for raw transaction data."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Raw transaction record from the dataset."""

    trans_date_trans_time: datetime
    cc_num: int
    merchant: str
    category: str
    amt: float = Field(gt=0)
    first: str
    last: str
    gender: str = Field(pattern="^[MF]$")
    street: str
    city: str
    state: str = Field(min_length=2, max_length=2)
    zip: int
    lat: float
    long: float
    city_pop: int = Field(ge=0)
    job: str
    dob: date
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: int = Field(ge=0, le=1)

    model_config = {"extra": "forbid"}
