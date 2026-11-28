# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 fonts-comic-neue fonts-dejavu-core libglib2.0-0 libgomp1 libraqm0 poppler-utils util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY pyproject.toml uv.lock ./
RUN --mount=type=bind,from=uv,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev --group inpainting --python /usr/local/bin/python
RUN python -c "import paddle; from PIL import features; assert features.check_feature('raqm')"
COPY deploy/download-models.sh deploy/
RUN HF_HUB_DISABLE_TELEMETRY=1 sh deploy/download-models.sh /opt/tralens/ocr \
    && chmod -R a+rX /opt/tralens/ocr \
    && rm -rf /root/.cache/huggingface
COPY deploy/ deploy/
RUN python deploy/font-manifest.py \
    && dpkg-query -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' > /app/debian-sources.tsv
COPY README.md LICENSE THIRD_PARTY_NOTICES.txt PRIVACY.txt ./
COPY backend/ backend/
COPY frontend/web/ frontend/web/

USER 1000:1000
ENTRYPOINT ["/app/.venv/bin/python", "/app/backend/serve.py"]
