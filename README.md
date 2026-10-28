# King County House Prices: From Notebook to Production

This project turns an exploratory Jupyter notebook on King County house sales into clean, tested, reusable Python code, and serves the trained model through a containerized FastAPI service.

What it covers:

- **Data preparation and feature engineering** refactored into small, pure functions
- A **reusable pipeline** (`HousingPipeline`) from raw CSV to model-ready data
- A **training workflow** (ElasticNet with polynomial features, tuned with 5-fold cross-validation)
- **31 unit and integration tests** with pytest
- A **FastAPI** service with CRUD for houses, price prediction from raw house data, and a saved prediction history
- **Docker Compose** setup running the API and PostgreSQL together

## Results

| Metric (test set) | Value |
|---|---|
| R² | 0.844 |
| Mean absolute error | $94,814 |


## Setup

```bash
git clone <repo-url>
cd <repo-name>
uv sync
cp .env.example .env
```

## Usage

**Prepare the data**

```bash
uv run python -m src.king_county.pipeline
```

Writes `data/King_County_House_prices_processed.csv`.

**Train the model**

```bash
uv run python -m src.king_county.modeling
```

Prints the best parameters and test results, and saves `model/model.bin` and `model/waterfront_locations.csv`.

**Run the tests**

```bash
uv run python -m pytest -q tests
```

**Run the API with Docker**

```bash
docker compose up --build
```

Then open http://localhost:8000/docs.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/houses` | Create a house (5 features) |
| `GET` | `/houses` | List all houses |
| `GET` | `/houses/{id}` | Get one house |
| `PUT` | `/houses/{id}` | Update a house |
| `DELETE` | `/houses/{id}` | Delete a house |
| `POST` | `/predict` | Predict a price from raw house data and save the prediction |
| `GET` | `/predictions` | List saved predictions, newest first |

`/predict` takes the raw house fields (for example `yr_built`, `yr_renovated`, `lat`, `long`) and computes `sqft_basement`, `last_known_change`, `center_distance`, and `water_distance` with the same functions used in training.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000,
       "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
       "sqft_above": 1400, "zipcode": 98103, "lat": 47.6, "long": -122.3,
       "sqft_living15": 1700, "sqft_lot15": 5000, "yr_built": 1960, "yr_renovated": 0}'
```


## Project Questions

**What steps did you take to complete the project?**
I reviewed the notebook and separated exploration from stable transformation rules. I turned each cleaning and feature step into a function, moved them into `src/king_county/`, and built `HousingPipeline`. I then refactored the modeling step, wrote unit and integration tests, built the FastAPI CRUD app, containerized it with Docker Compose, and extended it with a prediction endpoint and prediction history.

**What challenges did you face?**
Finding a general rule for the bedroom outlier instead of a hard-coded row. Making `water_distance` work for a single house, which required saving the waterfront reference locations at training time. Keeping the API's feature computation identical to training. 

**What would you do differently with more time?**
Improve the model: predict `log(price)`, treat `zipcode` as categorical, and try tree-based models such as XGBoost or LightGBM. Keep `yr_built` alongside `last_known_change`, since merging them loses information. Add CI to run the tests on every push, and track model performance over time using the saved predictions.

