import os

import numpy as np
import pandas as pd
import pytest
import skops.io as sio

from src.king_county.modeling import evaluate, save_model, split_data, train_model


@pytest.fixture(scope="module")
def processed_houses():
    rng = np.random.default_rng(42)
    n_rows = 60
    grade = rng.integers(5, 12, n_rows)
    sqft_living = rng.integers(800, 4000, n_rows)
    return pd.DataFrame(
        {
            "id": range(n_rows),
            "date": ["10/13/2014"] * n_rows,
            "price": grade * 50000 + sqft_living * 100 + rng.normal(0, 10000, n_rows),
            "sqft_price": rng.uniform(20, 100, n_rows),
            "delta_lat": rng.uniform(0, 1, n_rows),
            "delta_long": rng.uniform(0, 1, n_rows),
            "grade": grade,
            "sqft_living": sqft_living,
        }
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("model").joinpath("model.bin"))


def test_train_save_and_load_model(processed_houses, model_path):
    X_train, X_test, y_train, y_test = split_data(processed_houses)
    model = train_model(X_train, y_train)
    save_model(model, model_path)

    assert os.path.exists(model_path)
    loaded_model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))

    predictions = loaded_model.predict(X_test)
    assert len(predictions) == len(X_test)
    np.testing.assert_allclose(predictions, model.predict(X_test))
    assert evaluate(loaded_model, X_test, y_test)["r2"] > 0.5