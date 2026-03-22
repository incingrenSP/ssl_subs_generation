# ─────────────────────────────────────────────────────────────────────────────
# Nepali ASR — CPU-only deployment image
# Python 3.11 on Debian slim to match Miniconda env
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential + cmake        : compile KenLM Python bindings from source
# libboost-all-dev + libeigen3   : KenLM runtime dependencies
# zlib / bz2 / lzma              : KenLM compression support
# ffmpeg                         : audio format conversion (mp4, m4a, opus …)
# git                            : clone KenLM repo during build
# libsndfile1                    : soundfile backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer cache is reused when only
# source code changes — expensive pip installs only rerun when
# requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── KenLM Python bindings ─────────────────────────────────────────────────────
# Built from source inside the container — no Windows native build needed.
# MAX_ORDER=6 covers any LM up to 6-gram; your 3-gram binary works fine.
RUN git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm
RUN MAX_ORDER=6 pip install --no-cache-dir /tmp/kenlm && \
    rm -rf /tmp/kenlm

# ── Application source ────────────────────────────────────────────────────────
COPY src/          /app/src/
COPY templates/    /app/templates/
# COPY static/       /app/static/
COPY app.py        /app/app.py

# ── Model files ───────────────────────────────────────────────────────────────
# These are large — copied last so code changes above don't bust this cache layer.
COPY checkpoints/ssl_model.pt     /app/checkpoints/ssl_model.pt
COPY vocab/tokenizer.json         /app/vocab/tokenizer.json
COPY lm/nepali_kenlm_3g.bin          /app/lm/nepali_kenlm_3g.bin

# ── Runtime directories ───────────────────────────────────────────────────────
# Created at build time so the app doesn't need write permission errors at start.
RUN mkdir -p /app/uploads /app/outputs /app/static/audio

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
