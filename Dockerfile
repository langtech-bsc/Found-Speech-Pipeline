# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl gnupg ca-certificates lsb-release \
        ffmpeg sudo unzip

# Apertium nightly repo + language packages
RUN wget -qO- https://apertium.projectjj.com/apt/install-nightly.sh | bash && \
    apt-get update && apt-get install -y --no-install-recommends \
        apertium-all-dev apertium-eng apertium-en-es apertium-cat-spa apertium-spa apertium-cat apertium-cat-eng

WORKDIR /app
COPY . .

# Python environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel \
 && pip install "Cython>=0.29" \
 && pip install --no-build-isolation youtokentome==1.0.6 \
 && PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt

# Ensure setuptools provides pkg_resources (required by lightning_utilities → NeMo).
# Pin to <70: setuptools 70+ changed pkg_resources and breaks lightning_utilities.
RUN pip install "setuptools>=65,<70"

# Runtime directories and language-ID model (required by generate_final_data.py)
RUN mkdir -p ingestion merged utils/models/nemo utils/models/huggingface

# Model cache directories – mount a host folder at /app/utils/models for offline use
ENV HF_HOME=/app/utils/models/huggingface
ENV NEMO_CACHE_DIR=/app/utils/models/nemo

# Download lid.176.bin from B2Drop (same source as README). 
# If the link fails, the build will error with a helpful message.
RUN curl -sL "https://b2drop.bsc.es/index.php/s/x5kXGjTX7mYZFEN/download" -o /tmp/lid_dl \
    && (file /tmp/lid_dl | grep -qi "zip" \
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
