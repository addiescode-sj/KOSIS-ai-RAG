# Build context is the repo root (see docker-compose.yml) so vendor/phase1 is included.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY vendor/phase1 vendor/phase1
COPY backend backend
ENV PYTHONPATH=/app:/app/vendor/phase1:/app/vendor/phase1/src
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
