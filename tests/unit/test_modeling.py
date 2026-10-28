import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.king_county.modeling import (
    COLUMNS_TO_DROP,
    TEST_SIZE,
    build_pipeline,
    evaluate,
    save_waterfront_locations,
    split_data,
)


@pytest.fixture
def processed_houses():
    rng = np.random.default_rng(42)
    n_rows = 20
    houses = pd.DataFrame(
        {
            "id": range(n_rows),
            "date": ["10/13/2014"] * n_rows,
            "price": rng.uniform(200000, 900000, n_rows),
            "sqft_price": rng.uniform(20, 100, n_rows),
            "delta_lat": rng.uniform(0, 1, n_rows),
            "delta_long": rng.uniform(0, 1, n_rows),
            "grade": rng.integers(5, 12, n_rows),
            "sqft_living": rng.integers(800, 4000, n_rows),
        }
    )
    return houses


def test_split_data_sizes(processed_houses):
    X_train, X_test, y_train, y_test = split_data(processed_houses)
    assert len(X_test) == int(len(processed_houses) * TEST_SIZE)
    assert len(X_train) + len(X_test) == len(processed_houses)
    assert len(X_train) == len(y_train)


@pytest.mark.parametrize("column", COLUMNS_TO_DROP)
def test_split_data_drops_leakage_columns(processed_houses, column):
    X_train, X_test, _, _ = split_data(processed_houses)
    assert column not in X_train.columns
    assert column not in X_test.columns


def test_build_pipeline_steps():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["polynomial", "scaler", "model"]


class PerfectModel:
    def predict(self, X):
        return X["true_price"].to_numpy()


def test_evaluate_perfect_model():
    X_test = pd.DataFrame({"true_price": [100.0, 200.0, 300.0]})
    y_test = pd.Series([100.0, 200.0, 300.0])
    assert evaluate(PerfectModel(), X_test, y_test) == {"r2": 1.0, "mae": 0}


def test_save_waterfront_locations(tmp_path):
    houses = pd.DataFrame(
        {"lat": [47.6, 47.7, 47.5], "long": [-122.3, -122.2, -122.1], "waterfront": [1, 0, 1]}
    )
    path = tmp_path / "waterfront_locations.csv"
    save_waterfront_locations(houses, str(path))

    expected_result = pd.DataFrame({"lat": [47.6, 47.5], "long": [-122.3, -122.1]})
    pd.testing.assert_frame_equal(pd.read_csv(path), expected_result)