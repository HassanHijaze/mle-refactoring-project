import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.king_county.data_preparation import (
    calculate_last_change,
    fill_missings_view_wf,
    remove_bedroom_outlier,
    sqft_basement,
)


@pytest.fixture
def houses():
    return pd.DataFrame(
        {
            "bedrooms": [3, 33, 2],
            "sqft_living": [1800, 1620, 300],
            "sqft_above": [1500, 1620, 300],
            "sqft_basement": ["?", "0.0", "0.0"],
            "view": [np.nan, 2.0, 0.0],
            "waterfront": [1.0, np.nan, 0.0],
            "yr_built": [1950, 1960, 1970],
            "yr_renovated": [2000.0, 0.0, np.nan],
        }
    )


def test_remove_bedroom_outlier(houses):
    input_frame = houses[["bedrooms", "sqft_living"]]
    expected_result = pd.DataFrame(
        {"bedrooms": [3, 2], "sqft_living": [1800, 300]},
        index=[0, 2],
    )
    assert_frame_equal(remove_bedroom_outlier(input_frame), expected_result)


@pytest.mark.parametrize(
    "input_frame, expected_result",
    [
        (
            pd.DataFrame({"sqft_living": [2000, 1500], "sqft_above": [1500, 1500], "sqft_basement": ["?", "0.0"]}),
            pd.Series([500, 0], name="sqft_basement"),
        ),
        (
            pd.DataFrame({"sqft_living": [3000], "sqft_above": [1000], "sqft_basement": ["2000.0"]}),
            pd.Series([2000], name="sqft_basement"),
        ),
    ],
)
def test_sqft_basement(input_frame, expected_result):
    assert_series_equal(sqft_basement(input_frame)["sqft_basement"], expected_result)


def test_fill_missings_view_wf(houses):
    input_frame = houses[["view", "waterfront"]]
    expected_result = pd.DataFrame(
        {"view": [0.0, 2.0, 0.0], "waterfront": [1.0, 0.0, 0.0]}
    )
    assert_frame_equal(fill_missings_view_wf(input_frame), expected_result)


@pytest.mark.parametrize(
    "input_frame, expected_result",
    [
        (
            pd.DataFrame({"yr_built": [1950, 1960, 1970], "yr_renovated": [2000.0, 0.0, np.nan]}),
            pd.DataFrame({"last_known_change": [2000, 1960, 1970]}),
        ),
        (
            pd.DataFrame({"yr_built": [1900], "yr_renovated": [1990.0]}),
            pd.DataFrame({"last_known_change": [1990]}),
        ),
    ],
)
def test_calculate_last_change(input_frame, expected_result):
    assert_frame_equal(calculate_last_change(input_frame), expected_result)


@pytest.mark.parametrize(
    "cleaning_step",
    [remove_bedroom_outlier, sqft_basement, fill_missings_view_wf, calculate_last_change],
)
def test_cleaning_steps_do_not_change_the_input(houses, cleaning_step):
    original = houses.copy()
    cleaning_step(houses)
    assert_frame_equal(houses, original)