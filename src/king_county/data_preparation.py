
"""Cleaning steps for the King County house sales dataset."""

import pandas as pd

MIN_SQFT_PER_BEDROOM = 100


def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def remove_bedroom_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """Drop houses whose living area is too small for their number of bedrooms."""
    df = df.copy()
    sqft_per_bedroom = df["sqft_living"] / df["bedrooms"]
    return df.loc[sqft_per_bedroom >= MIN_SQFT_PER_BEDROOM].copy()


def sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild basement size from living and above ground area."""
    df = df.copy()
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df


def fill_missings_view_wf(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing view and waterfront values with 0."""
    df = df.copy()
    df["view"] = df["view"].fillna(0)
    df["waterfront"] = df["waterfront"].fillna(0)
    return df


def calculate_last_change(df: pd.DataFrame) -> pd.DataFrame:
    """Combine yr_renovated and yr_built into last_known_change."""
    df = df.copy()
    renovated = df["yr_renovated"].fillna(0)
    df["last_known_change"] = renovated.where(renovated != 0, df["yr_built"]).astype(int)
    return df.drop(columns=["yr_renovated", "yr_built"])


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in the same order as the notebook."""
    df = remove_bedroom_outlier(df)
    df = sqft_basement(df)
    df = fill_missings_view_wf(df)
    df = calculate_last_change(df)
    return df