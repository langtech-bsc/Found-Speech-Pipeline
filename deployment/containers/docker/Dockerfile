# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl gnupg ca-certificates lsb-release \
        ffmpeg libsndfile1 sudo unzip

# Apertium nightly repo + language packages
RUN wget -qO- https://apertium.projectjj.com/apt/install-nightly.sh | bash && \
    apt-get update && apt-get install -y --no-install-recommends \
        apertium-all-dev apertium-eng apertium-en-es apertium-cat-spa apertium-spa apertium-cat apertium-cat-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Runtime directories and language-ID model (required by generate_final_data.py)
RUN mkdir -p ingestion merged utils/models/nemo utils/models/huggingface

# Model cache directories – mount a host folder at /app/utils/models for offline use
ENV HF_HOME=/app/utils/models/huggingface
ENV NEMO_CACHE_DIR=/app/utils/models/nemo

# Download lid.176.bin from B2Drop (same source as README).
# If the link fails, the build will error with a helpful message.
RUN curl -sL "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" -o /tmp/lid_dl \
    && (unzip -t /tmp/lid_dl >/dev/null 2>&1 \
        && unzip -j -o /tmp/lid_dl '*lid.176.bin' -d utils/models \
        || cp /tmp/lid_dl utils/models/lid.176.bin) \
    && rm -f /tmp/lid_dl \
    && if [ ! -s utils/models/lid.176.bin ] || [ "$(stat -c%s utils/models/lid.176.bin)" -lt 1000000 ]; then \
        echo "ERROR: lid.176.bin download failed or file is too small." && \
        echo "Please ensure the download link is active or mount the model at runtime:" && \
        echo "  -v /path/to/lid.176.bin:/app/utils/models/lid.176.bin" && \
        exit 1; \
    fi

# Set default command to show help or run batch mode if ingestion is populated
CMD ["python", "pipeline_service.py", "--help"]
