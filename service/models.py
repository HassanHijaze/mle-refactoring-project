"""Database tables for houses and predictions."""

from datetime import datetime

from sqlalchemy import JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from service.database import Base


class House(Base):
    """One house stored in the houses table."""

    __tablename__ = "houses"

    id: Mapped[int] = mapped_column(primary_key=True)
    bedrooms: Mapped[int]
    bathrooms: Mapped[float]
    sqft_living: Mapped[int]
    grade: Mapped[int]
    price: Mapped[float]


class Prediction(Base):
    """One prediction made by the API, with the input it was based on."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    input_features: Mapped[dict] = mapped_column(JSON)
    predicted_price: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())