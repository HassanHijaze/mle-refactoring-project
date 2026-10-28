"""FastAPI app for King County houses: CRUD, price prediction, and prediction history."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
import skops.io as sio
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from service import models, schemas
from service.database import Base, engine, get_db
from src.king_county.data_preparation import calculate_last_change, sqft_basement
from src.king_county.feature_engineering import add_center_distance, add_water_distance

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.bin")
WATERFRONT_PATH = os.getenv("WATERFRONT_PATH", "model/waterfront_locations.csv")


def load_model(path: str):
    """Load the trained pipeline saved with skops."""
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the tables and load the model once, when the app starts."""
    Base.metadata.create_all(bind=engine)
    app.state.model = load_model(MODEL_PATH)
    app.state.waterfront = pd.read_csv(WATERFRONT_PATH)
    yield


app = FastAPI(
    title="King County Houses API",
    description="CRUD for houses described by 5 features, plus price prediction with a saved history.",
    version="0.1.0",
    lifespan=lifespan,
)

DatabaseSession = Annotated[Session, Depends(get_db)]


def find_house(house_id: int, db: Session) -> models.House:
    """Look up a house by id and answer 404 if it does not exist."""
    house = db.get(models.House, house_id)
    if house is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No house with id {house_id}.",
        )
    return house


def build_model_features(house: schemas.HouseFeatures, waterfront: pd.DataFrame) -> pd.DataFrame:
    """Turn one raw house into the engineered features the model was trained on."""
    features = pd.DataFrame([house.model_dump()])
    features = sqft_basement(features)
    features = calculate_last_change(features)
    features = add_center_distance(features)
    features = add_water_distance(features, waterfront)
    return features


@app.get("/", tags=["status"])
def health_check() -> dict[str, str]:
    """Confirm that the API is up."""
    return {"status": "ok"}


@app.post(
    "/houses",
    response_model=schemas.HouseOut,
    status_code=status.HTTP_201_CREATED,
    tags=["houses"],
)
def create_house(house_data: schemas.HouseCreate, db: DatabaseSession) -> models.House:
    """Save a new house and return it with its new id."""
    house = models.House(**house_data.model_dump())
    db.add(house)
    db.commit()
    db.refresh(house)
    return house


@app.get("/houses", response_model=list[schemas.HouseOut], tags=["houses"])
def list_houses(db: DatabaseSession) -> list[models.House]:
    """Return every stored house, sorted by id."""
    query = select(models.House).order_by(models.House.id)
    return list(db.scalars(query))


@app.get("/houses/{house_id}", response_model=schemas.HouseOut, tags=["houses"])
def read_house(house_id: int, db: DatabaseSession) -> models.House:
    """Return a single house."""
    return find_house(house_id, db)


@app.put("/houses/{house_id}", response_model=schemas.HouseOut, tags=["houses"])
def update_house(
    house_id: int, house_data: schemas.HouseUpdate, db: DatabaseSession
) -> models.House:
    """Overwrite all 5 features of an existing house."""
    house = find_house(house_id, db)
    for field_name, new_value in house_data.model_dump().items():
        setattr(house, field_name, new_value)
    db.commit()
    db.refresh(house)
    return house


@app.delete(
    "/houses/{house_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["houses"]
)
def delete_house(house_id: int, db: DatabaseSession) -> Response:
    """Remove a house from the database."""
    house = find_house(house_id, db)
    db.delete(house)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/predict", response_model=schemas.PricePrediction, tags=["prediction"])
def predict_price(
    house: schemas.HouseFeatures, request: Request, db: DatabaseSession
) -> schemas.PricePrediction:
    """Predict the sale price of one house and save the prediction."""
    model = request.app.state.model
    features = build_model_features(house, request.app.state.waterfront)
    model_input = features[model.feature_names_in_]
    price = round(float(model.predict(model_input)[0]), 2)

    db.add(models.Prediction(input_features=house.model_dump(), predicted_price=price))
    db.commit()

    return schemas.PricePrediction(predicted_price=price)


@app.get("/predictions", response_model=list[schemas.PredictionOut], tags=["prediction"])
def list_predictions(db: DatabaseSession) -> list[models.Prediction]:
    """Return all saved predictions, newest first."""
    query = select(models.Prediction).order_by(models.Prediction.id.desc())
    return list(db.scalars(query))