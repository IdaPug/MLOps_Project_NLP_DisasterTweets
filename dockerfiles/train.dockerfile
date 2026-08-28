FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE

# Install dependencies without installing the project yet
RUN uv sync --locked --no-cache --no-install-project

COPY src src/
COPY data/processed_data data/processed_data

# Install the project
RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "src/project/train.py"]
