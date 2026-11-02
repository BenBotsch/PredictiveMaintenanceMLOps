# Serving image: the API with one exported bundle, without MLflow or the training data.
# Build after `pmm export`, so that bundle/ exists.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups
COPY bundle ./bundle

RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
CMD ["/app/.venv/bin/pmm", "serve", "--host", "0.0.0.0", "--port", "8000"]
