FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

COPY artifacts/disaster_tweet_model:v0 artifacts/disaster_tweet_model:v0/

COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn", "src.project.api:app", "--host", "0.0.0.0", "--port", "8000"]
