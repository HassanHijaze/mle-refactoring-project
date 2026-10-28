## Checklist for the Refactoring Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [x] I read and understood the notebook.
- [x] I made a copy of the notebook for refactoring it.
- [x] I removed unnecessary code and only kept relevant code for data cleaning and feature engineering.
- [x] I transformed the steps in the Data Preparation and Feature Engineering sections into functions.
- [x] Functions are in a separate Python file.

**Optional tasks:**
- [x] I refactored the modeling step and added it to the pipeline.
- [x] I built a FastAPI app that allows me to create, read, update and delete houses.
- [x] I created a `Dockerfile` and a `docker-compose.yaml` file for the app.
- [x] I built the container with Docker Compose and I am able to run the app inside of it.
- [x] I created tests for my functions.

**Additional work:**
- [x] Saved the trained model with `skops` together with the waterfront reference locations needed for new predictions.
- [x] Added a `POST /predict` endpoint that accepts raw house data and computes the engineered features with the same code used in training.
- [x] Verified on 10 test houses that the API and the trained model give identical predictions.
- [x] Saved every prediction with its input and timestamp to PostgreSQL and added `GET /predictions` to view the history.