"""Training workflow: split data, tune an ElasticNet pipeline, evaluate, save."""

import os

import pandas as pd
import skops.io as sio
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.king_county.feature_engineering import get_waterfront_locations
from src.king_county.pipeline import PROCESSED_DATA_PATH, RAW_DATA_PATH, HousingPipeline

TARGET = "price"
COLUMNS_TO_DROP = ["price", "sqft_price", "date", "delta_lat", "delta_long", "id"]
TEST_SIZE = 0.3
RANDOM_STATE = 42
MODEL_PATH = "model/model.bin"
WATERFRONT_PATH = "model/waterfront_locations.csv"

PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}


def split_data(df: pd.DataFrame):
    """Split features and target into train and test sets."""
    X = df.drop(columns=COLUMNS_TO_DROP)
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_pipeline() -> Pipeline:
    """Polynomial features -> scaling -> ElasticNet with a scaled target."""
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(2, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Find the best ElasticNet settings with 5-fold cross-validation."""
    search = GridSearchCV(build_pipeline(), PARAM_GRID, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return R² and mean absolute error on the test set."""
    predictions = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, predictions), 3),
        "mae": round(mean_absolute_error(y_test, predictions)),
    }


def save_model(model: Pipeline, path: str = MODEL_PATH) -> None:
    """Save the trained pipeline with skops."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        sio.dump(model, file)


def save_waterfront_locations(houses: pd.DataFrame, path: str = WATERFRONT_PATH) -> None:
    """Save the waterfront reference points needed to compute water_distance for new houses."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    get_waterfront_locations(houses).to_csv(path, index=False)


if __name__ == "__main__":
    houses = HousingPipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()
    X_train, X_test, y_train, y_test = split_data(houses)
    model = train_model(X_train, y_train)
    print("Test results:", evaluate(model, X_test, y_test))
    save_model(model)
    save_waterfront_locations(houses)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Waterfront locations saved to {WATERFRONT_PATH}")