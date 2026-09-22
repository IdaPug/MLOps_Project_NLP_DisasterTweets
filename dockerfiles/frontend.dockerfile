FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir uv

RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/project/frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]
