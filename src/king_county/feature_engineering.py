"""Feature engineering steps for the King County house sales dataset."""

import numpy as np
import pandas as pd

WEALTH_CENTER_LAT = 47.62774
WEALTH_CENTER_LONG = -122.24194
LONG_CORRECTION_LAT = 47.6219
EARTH_RADIUS_KM = 6378
KM_PER_DEGREE = 2 * np.pi * EARTH_RADIUS_KM / 360


def distance_km(long, lat, ref_long, ref_lat):
    """Approximate distance in km between points and a reference location."""
    delta_long_corr = (long - ref_long) * np.cos(np.radians(ref_lat))
    delta_lat = lat - ref_lat
    return np.sqrt(delta_long_corr**2 + delta_lat**2) * KM_PER_DEGREE


def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add price per square foot of living area plus lot."""
    df = df.copy()
    df["sqft_price"] = (df["price"] / (df["sqft_living"] + df["sqft_lot"])).round(2)
    return df


def add_center_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add the distance in km to the wealth center in Medina."""
    df = df.copy()
    df["delta_lat"] = np.absolute(WEALTH_CENTER_LAT - df["lat"])
    df["delta_long"] = np.absolute(WEALTH_CENTER_LONG - df["long"])
    df["center_distance"] = (
        np.sqrt(
            (df["delta_long"] * np.cos(np.radians(LONG_CORRECTION_LAT))) ** 2
            + df["delta_lat"] ** 2
        )
        * KM_PER_DEGREE
    )
    return df


def get_waterfront_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the lat/long of all waterfront houses, used as water reference points."""
    return df.loc[df["waterfront"] == 1, ["lat", "long"]].reset_index(drop=True)


def add_water_distance(df: pd.DataFrame, waterfront: pd.DataFrame) -> pd.DataFrame:
    """Add the distance in km to the nearest house in the waterfront reference set."""
    df = df.copy()
    distances = distance_km(
        df["long"].to_numpy()[:, None],
        df["lat"].to_numpy()[:, None],
        waterfront["long"].to_numpy()[None, :],
        waterfront["lat"].to_numpy()[None, :],
    )
    df["water_distance"] = distances.min(axis=1)
    return df


def engineer_features(
    df: pd.DataFrame, waterfront: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Run all feature engineering steps in the same order as the notebook.

    If no waterfront reference set is given, it is taken from df itself
    (the training case). For new houses, pass the saved reference set.
    """
    if waterfront is None:
        waterfront = get_waterfront_locations(df)
    df = add_sqft_price(df)
    df = add_center_distance(df)
    df = add_water_distance(df, waterfront)
    return df