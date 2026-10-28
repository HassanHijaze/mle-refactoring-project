"""Pydantic contracts for house requests and responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HouseBase(BaseModel):
    """Shared house fields with validation rules."""

    bedrooms: int = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=10)
    sqft_living: int = Field(gt=0)
    grade: int = Field(ge=1, le=13)
    price: float = Field(gt=0)


class HouseCreate(HouseBase):
    """Data required to create a house."""


class HouseUpdate(HouseBase):
    """Data required to update a house."""


class HouseOut(HouseBase):
    """House data returned by the API, including its id."""

    model_config = ConfigDict(from_attributes=True)

    id: int


class HouseFeatures(BaseModel):
    """Raw house data sent to the prediction endpoint."""

    bedrooms: int = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=10)
    sqft_living: int = Field(gt=0)
    sqft_lot: int = Field(gt=0)
    floors: float = Field(ge=1, le=4)
    waterfront: int = Field(ge=0, le=1)
    view: int = Field(ge=0, le=4)
    condition: int = Field(ge=1, le=5)
    grade: int = Field(ge=1, le=13)
    sqft_above: int = Field(gt=0)
    zipcode: int = Field(ge=98000, le=98999)
    lat: float = Field(ge=47.0, le=48.0)
    long: float = Field(ge=-123.0, le=-121.0)
    sqft_living15: int = Field(gt=0)
    sqft_lot15: int = Field(gt=0)
    yr_built: int = Field(ge=1900, le=2026)
    yr_renovated: int = Field(default=0, ge=0, le=2026)


class PricePrediction(BaseModel):
    """Predicted sale price returned by the API."""

    predicted_price: float


class PredictionOut(BaseModel):
    """A saved prediction returned by the API, with its input and time."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    input_features: dict
    predicted_price: float
    created_at: datetime