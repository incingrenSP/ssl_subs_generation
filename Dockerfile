# ─────────────────────────────────────────────────────────────────────────────
# Nepali ASR — CPU-only deployment image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
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

# ── 1. KenLM FIRST ───────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm && \
    MAX_ORDER=6 pip install --no-cache-dir /tmp/kenlm && \
    rm -rf /tmp/kenlm

# ── 2. flashlight-text SECOND (requires kenlm to be importable) ──────────────
RUN pip install --no-cache-dir flashlight-text --no-binary flashlight-text

# ── 3. Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/        /app/src/
COPY index.html  /app/index.html
COPY main.py     /app/main.py

# ── Model files ───────────────────────────────────────────────────────────────
COPY deployment/asr_model_prototype.pt  /app/deployment/asr_model_prototype.pt
COPY deployment/nepali_kenlm_3g.bin     /app/deployment/nepali_kenlm_3g.bin
COPY data/tokenizer.json                /app/data/tokenizer.json
COPY data/metadata_normal.tsv           /app/data/metadata_normal.tsv

# Runtime directories
RUN mkdir -p /app/asr_jobs

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Port
EXPOSE 8000

# Entrypoint
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]