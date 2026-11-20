FROM ghcr.io/astral-sh/uv:0.13.9 AS uv
FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 libraqm0 poppler-utils util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --group inpainting --python /usr/local/bin/python
COPY backend/ backend/
COPY frontend/web/ frontend/web/

USER 1000:1000
ENTRYPOINT ["/app/.venv/bin/python", "/app/backend/serve.py"]
