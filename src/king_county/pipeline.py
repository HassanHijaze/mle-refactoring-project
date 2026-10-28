"""Reusable pipeline: load raw data, clean it, engineer features, save the result."""

import pandas as pd

from src.king_county.data_preparation import load_data, prepare_data
from src.king_county.feature_engineering import engineer_features

RAW_DATA_PATH = "data/King_County_House_prices_dataset.csv"
PROCESSED_DATA_PATH = "data/King_County_House_prices_processed.csv"


class HousingPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def run(self) -> pd.DataFrame:
        data = self.read_data()
        processed_data = self.process_data(data)
        self.write_data(processed_data)
        return processed_data

    def read_data(self) -> pd.DataFrame:
        return load_data(self.input_path)

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = prepare_data(data)
        return engineer_features(cleaned)

    def write_data(self, processed_data: pd.DataFrame) -> None:
        processed_data.to_csv(self.output_path, index=False)


if __name__ == "__main__":
    result = HousingPipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()
    print(f"Saved {result.shape[0]} rows and {result.shape[1]} columns to {PROCESSED_DATA_PATH}")