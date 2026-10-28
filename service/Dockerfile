# Image for the King County houses API.
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM python:3.13-slim-bookworm

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first so this layer is reused when only code changes.
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Add the API code, the shared feature code, and the trained model.
COPY src ./src
COPY service ./service
COPY model ./model

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]