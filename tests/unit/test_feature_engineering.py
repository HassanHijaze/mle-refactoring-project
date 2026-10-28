import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.king_county.feature_engineering import (
    WEALTH_CENTER_LAT,
    WEALTH_CENTER_LONG,
    add_center_distance,
    add_sqft_price,
    add_water_distance,
    engineer_features,
    get_waterfront_locations,
)


@pytest.fixture
def houses():
    return pd.DataFrame(
        {
            "price": [300000.0, 500000.0],
            "sqft_living": [1000, 2000],
            "sqft_lot": [2000, 3000],
            "lat": [WEALTH_CENTER_LAT, WEALTH_CENTER_LAT + 1],
            "long": [WEALTH_CENTER_LONG, WEALTH_CENTER_LONG],
            "waterfront": [1, 0],
        }
    )


@pytest.mark.parametrize(
    "input_frame, expected_result",
    [
        (
            pd.DataFrame({"price": [300000.0, 500000.0], "sqft_living": [1000, 2000], "sqft_lot": [2000, 3000]}),
            pd.Series([100.0, 100.0], name="sqft_price"),
        ),
        (
            pd.DataFrame({"price": [100000.0], "sqft_living": [1000], "sqft_lot": [2000]}),
            pd.Series([33.33], name="sqft_price"),
        ),
    ],
)
def test_add_sqft_price(input_frame, expected_result):
    assert_series_equal(add_sqft_price(input_frame)["sqft_price"], expected_result)


def test_add_center_distance(houses):
    expected_result = pd.Series([0.0, 111.3], name="center_distance")
    assert_series_equal(
        add_center_distance(houses)["center_distance"], expected_result, atol=0.1
    )


def test_get_waterfront_locations(houses):
    expected_result = pd.DataFrame(
        {"lat": [WEALTH_CENTER_LAT], "long": [WEALTH_CENTER_LONG]}
    )
    assert_frame_equal(get_waterfront_locations(houses), expected_result)


def test_add_water_distance(houses):
    waterfront = get_waterfront_locations(houses)
    expected_result = pd.Series([0.0, 111.3], name="water_distance")
    assert_series_equal(
        add_water_distance(houses, waterfront)["water_distance"], expected_result, atol=0.1
    )


def test_engineer_features_single_house_uses_saved_waterfront(houses):
    waterfront = get_waterfront_locations(houses)
    new_house = houses.iloc[[1]].reset_index(drop=True)
    result = engineer_features(new_house, waterfront)
    expected_result = pd.Series([111.3], name="water_distance")
    assert_series_equal(result["water_distance"], expected_result, atol=0.1)


@pytest.mark.parametrize(
    "feature_step",
    [
        add_sqft_price,
        add_center_distance,
        lambda df: add_water_distance(df, get_waterfront_locations(df)),
    ],
    ids=["add_sqft_price", "add_center_distance", "add_water_distance"],
)
def test_feature_steps_do_not_change_the_input(houses, feature_step):
    original = houses.copy()
    feature_step(houses)
    assert_frame_equal(houses, original)