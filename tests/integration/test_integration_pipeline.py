import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.king_county.pipeline import HousingPipeline


@pytest.fixture(scope="module")
def input_path(tmp_path_factory):
    raw_path = tmp_path_factory.mktemp("data").joinpath("raw.csv")
    raw_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["10/13/2014", "12/9/2014", "2/25/2015"],
            "price": [221900.0, 538000.0, 640000.0],
            "bedrooms": [3, 3, 25],
            "bathrooms": [1.0, 2.25, 1.75],
            "sqft_living": [1180, 2570, 1620],
            "sqft_lot": [5650, 7242, 6000],
            "floors": [1.0, 2.0, 1.0],
            "waterfront": [np.nan, 1.0, 0.0],
            "view": [0.0, np.nan, 0.0],
            "condition": [3, 3, 5],
            "grade": [7, 7, 7],
            "sqft_above": [1180, 2170, 1040],
            "sqft_basement": ["0.0", "?", "580.0"],
            "yr_built": [1955, 1951, 1947],
            "yr_renovated": [0.0, 1991.0, np.nan],
            "zipcode": [98178, 98125, 98103],
            "lat": [47.5112, 47.7210, 47.6878],
            "long": [-122.257, -122.319, -122.331],
            "sqft_living15": [1340, 1690, 1330],
            "sqft_lot15": [5650, 7639, 4700],
        }
    )
    raw_data.to_csv(raw_path, index=False)
    return str(raw_path)


@pytest.fixture(scope="module")
def output_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("data").joinpath("processed.csv"))


def test_housing_pipeline(input_path, output_path):
    pipeline = HousingPipeline(input_path, output_path)
    processed_data = pipeline.run()

    assert os.path.exists(output_path)
    output_data = pd.read_csv(output_path)

    assert len(processed_data) == 2
    assert processed_data[["view", "waterfront"]].isna().sum().sum() == 0
    assert {"last_known_change", "sqft_price", "center_distance", "water_distance"} <= set(
        processed_data.columns
    )
    assert_frame_equal(processed_data.reset_index(drop=True), output_data)