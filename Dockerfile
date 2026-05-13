FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES="" \
    FORM_PARSER_OUTPUT_DIR=/app/output \
    FORM_PARSER_UPLOAD_DIR=/app/output/uploads \
    FORM_PARSER_RUNS_DIR=/app/output/runs \
    FORM_PARSER_EASYOCR_MODEL_DIR=/app/output/easyocr-models \
    FORM_PARSER_EASYOCR_DOWNLOAD_ENABLED=true \
    FORM_PARSER_OCR_THREADS=1 \
    FORM_PARSER_OCR_BATCH_SIZE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir \
        --index-url "${PYTORCH_CPU_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        --requirement requirements.txt

COPY src ./src

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p \
        /app/output/uploads \
        /app/output/runs \
        /app/output/easyocr-models/user_network \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
