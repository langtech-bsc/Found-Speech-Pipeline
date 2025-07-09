# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl gnupg ca-certificates lsb-release \
        ffmpeg sudo

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
 && PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt

# Runtime directories
RUN mkdir -p ingestion merged
ENV SUDO_PWD="your_sudo_password"

CMD ["python", "pipeline_service.py", "https://www.youtube.com/watch?v=-f2OsxyRLlQ"]
