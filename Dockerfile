FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" TF_CPP_MIN_LOG_LEVEL=2

# dependências primeiro (cache de camada)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen

EXPOSE 8000
CMD ["uvicorn", "animerec.api:app", "--host", "0.0.0.0", "--port", "8000"]
