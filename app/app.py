"""
app.py  –  Nepali ASR Web Application
======================================
Flask backend for Nepali speech-to-text using the SSL-based NepaliASR model.

Preprocessing pipeline — grounded in the actual source code
────────────────────────────────────────────────────────────
FACT 1  (ssl_model.py  ConvBlock.__init__)
    First Conv1d block uses nn.Identity() — no normalisation.
    Raw waveform amplitude hits kernel weights directly.
    → Per-chunk RMS normalisation to TARGET_RMS (0.03) matches training dist.

FACT 2  (dataset.py  _load_audio_sf)
    sf.read(..., dtype='float32') → float32 in [-1, 1].
    No global amplitude normalisation in the training pipeline.
    → Peak normalisation removed; per-chunk RMS scaling is sufficient.

FACT 3  (dataset.py  collate_fn_asr)
    lengths must be RAW SAMPLE COUNT of real audio (not padded tensor size).
    frame_lengths = (lengths / T_wav * T_cnn).long().clamp(max=T_cnn)
    → actual_len = LEAD_SAMPLES + real_samples_in_chunk (no end-padding).

FACT 4  (asr_model.py  NepaliASR.forward)
    Model returns (log_probs, frame_lengths) directly.
    log_probs already log-softmaxed — feed straight to greedy decode.
    → No separate CTC decode reimplementation needed; argmax + collapse inline.

Sliding window design
─────────────────────
    MAX_DURATION = 4.0s  — hard cap matching ASR fine-tuning max_length
    WINDOW_SIZE  = 3.5s  — analysis window per chunk
    OVERLAP      = 1.5s  — overlap between consecutive windows
    STEP         = 2.0s  — non-overlapping advance (WINDOW - OVERLAP)
    LEADING_SILENCE = 0.3s  — silence prepended to compensate CTC onset skip
"""

import os
import sys
import uuid
import json
import traceback
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from flask import Flask, request, jsonify, send_file, render_template

# ─────────────────────────────────────────────────────────────────────────────
# App & directory setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
AUDIO_DIR  = BASE_DIR / "static" / "audio"

for _d in (UPLOAD_DIR, OUTPUT_DIR, AUDIO_DIR, BASE_DIR / "templates"):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = str(BASE_DIR / "checkpoints" / "ssl_model.pt")
TOKENIZER_PATH  = str(BASE_DIR / "vocab"       / "tokenizer.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants — from ssl_model.py NepaliSSLConfig.__post_init__
#   cnn_strides = [5, 2, 2, 2, 2, 2, 2]  →  total stride = 5 × 2^6 = 320
#   at 16 000 Hz → one frame every 20 ms
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CNN_STRIDE   = 320                              # samples per CNN output frame

# ─────────────────────────────────────────────────────────────────────────────
# Sliding window parameters
# ─────────────────────────────────────────────────────────────────────────────
MAX_DURATION    = 4.0                           # seconds — must match training max_length
WINDOW_SIZE     = 3.5                           # seconds — analysis window
OVERLAP         = 1.5                           # seconds — window overlap
LEADING_SILENCE = 0.3                           # seconds — prepended silence for CTC onset
TARGET_RMS      = 0.03                          # per-chunk RMS target (matches training)

# Derived sample counts
WINDOW_SAMPLES  = int(WINDOW_SIZE     * SAMPLE_RATE)   # 56 000
OVERLAP_SAMPLES = int(OVERLAP         * SAMPLE_RATE)   # 24 000
STEP_SAMPLES    = WINDOW_SAMPLES - OVERLAP_SAMPLES     # 32 000
LEAD_SAMPLES    = int(LEADING_SILENCE * SAMPLE_RATE)   # 4 800
CHUNK_SAMPLES   = LEAD_SAMPLES + WINDOW_SAMPLES        # 60 800 (< 64 000 MAX ✓)
MIN_SAMPLES     = CNN_STRIDE * 4                       # 1 280 — shortest viable chunk

SILENCE_RMS_THRESHOLD = 1e-4                    # skip chunks below this RMS

# ─────────────────────────────────────────────────────────────────────────────
# Supported file types
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts"}

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model + tokenizer — loaded once on first request
# ─────────────────────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    sys.path.insert(0, str(BASE_DIR))
    from src.tokenizer import CharTokenizer, GraphemeTokenizer
    from src.asr_model  import NepaliASR
    from src.ssl_model  import NepaliSSL, NepaliSSLConfig

    # Auto-detect tokenizer type
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_meta = json.load(f)
    tok_type = tok_meta.get("tokenizer_type", "char")

    print(f"[ASR] Loading tokenizer  ← {TOKENIZER_PATH}  (type={tok_type})")
    _tokenizer = (
        GraphemeTokenizer.load(TOKENIZER_PATH)
        if tok_type == "grapheme"
        else CharTokenizer.load(TOKENIZER_PATH)
    )

    print(f"[ASR] Loading checkpoint ← {CHECKPOINT_PATH}  (device={DEVICE})")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    raw_sd     = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Detect checkpoint type by presence of CTC head weights
    is_asr_ckpt = any(k.startswith("ctc_head") for k in raw_sd)

    # Build SSL backbone — weights overwritten by load_state_dict below
    cfg_data = checkpoint.get("config", None)
    cfg = (
        NepaliSSLConfig(**cfg_data) if isinstance(cfg_data, dict)
        else NepaliSSLConfig()
    )
    ssl_model = NepaliSSL(cfg)

    if is_asr_ckpt:
        vocab_size = raw_sd["ctc_head.weight"].shape[0]
        print(f"[ASR] ASR checkpoint detected  (vocab_size={vocab_size})")
        asr_model = NepaliASR(ssl_model, vocab_size=vocab_size, freeze_encoder=False)
        asr_model.load_state_dict(raw_sd, strict=True)
    else:
        print("[ASR] SSL-only checkpoint — wrapping in NepaliASR (CTC head random)")
        ssl_model.load_state_dict(raw_sd, strict=True)
        ssl_model.eval()
        vocab_size = _tokenizer.get_vocab_size()
        asr_model  = NepaliASR(ssl_model, vocab_size=vocab_size, freeze_encoder=True)

    _model = asr_model.to(DEVICE)
    _model.eval()

    total     = sum(p.numel() for p in _model.parameters())
    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print(f"[ASR] Total params:     {total:,}")
    print(f"[ASR] Trainable params: {trainable:,}")
    print("[ASR] Ready.")
    return _model, _tokenizer


# ═════════════════════════════════════════════════════════════════════════════
#   AUDIO PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def extract_to_wav(src: Path, dst: Path) -> None:
    """Convert any audio/video to 16 kHz mono pcm_s16le WAV via FFmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-2000:]}")


def load_audio(wav_path: Path) -> np.ndarray:
    """
    Load WAV → float32 mono array at SAMPLE_RATE.
    Mirrors dataset.py _load_audio_sf exactly (Fact 2).
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        g    = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data.astype(np.float32)


def apply_rms_normalisation(chunk: np.ndarray, actual_len: int) -> np.ndarray:
    """
    Scale real audio portion to TARGET_RMS (0.03).

    Only [LEAD_SAMPLES:actual_len] is scaled — leading silence and
    end-padding zeros are left untouched. Clipped to [-1, 1] to keep
    values in the range the CNN's first conv was trained on (Fact 1).
    Replaces the previous global peak normalisation which had no measurable
    effect on this model.
    """
    real = chunk[LEAD_SAMPLES:actual_len]
    rms  = float(np.sqrt(np.mean(real ** 2)))
    if rms < 1e-8:
        return chunk                            # silent — don't amplify noise
    chunk = chunk.copy()
    chunk[LEAD_SAMPLES:actual_len] = np.clip(real * (TARGET_RMS / rms), -1.0, 1.0)
    return chunk


def is_silent(chunk: np.ndarray, actual_len: int) -> bool:
    """
    Return True if real audio [LEAD_SAMPLES:actual_len] is below threshold.
    Excludes leading silence and end-padding from the RMS calculation.
    """
    real = chunk[LEAD_SAMPLES:actual_len]
    return len(real) == 0 or float(np.sqrt(np.mean(real ** 2))) < SILENCE_RMS_THRESHOLD


def sliding_window_chunks(waveform: np.ndarray):
    """
    Yield (chunk, actual_len, start_sec, end_sec) tuples.

    chunk layout:
        [zeros × LEAD_SAMPLES] [real_audio] [zeros × end_padding]

    actual_len = LEAD_SAMPLES + len(real_audio)   — excludes end-padding.
    This is the value passed as `lengths` to NepaliASR.forward (Fact 3).

    Wall-clock timestamps are positions in the original waveform before
    leading silence was added, so SRT timestamps are correct.
    """
    silence = np.zeros(LEAD_SAMPLES, dtype=np.float32)
    total   = len(waveform)
    pos     = 0

    while pos < total:
        end_pos   = min(pos + WINDOW_SAMPLES, total)
        real_part = waveform[pos:end_pos]
        real_len  = len(real_part)

        if real_len < MIN_SAMPLES:
            pos += STEP_SAMPLES
            continue

        chunk      = np.concatenate([silence, real_part])
        actual_len = LEAD_SAMPLES + real_len           # Fact 3

        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.concatenate(
                [chunk, np.zeros(CHUNK_SAMPLES - len(chunk), dtype=np.float32)]
            )

        yield chunk, actual_len, pos / SAMPLE_RATE, end_pos / SAMPLE_RATE
        pos += STEP_SAMPLES


# ═════════════════════════════════════════════════════════════════════════════
#   INFERENCE
#   NepaliASR.forward → (log_probs, frame_lengths)  [Fact 4]
#   Greedy CTC: argmax → collapse consecutive duplicates → strip blanks.
#   Done inline — no separate decode function needed.
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def transcribe_file(wav_path: Path):
    """
    Full pipeline: load → sliding window → model → greedy CTC → entries.

    Returns
    -------
    entries  : list of {"start": float, "end": float, "text": str}
    duration : total audio duration in seconds
    """
    model, tokenizer = get_model_and_tokenizer()
    blank_id = 0

    waveform = load_audio(wav_path)
    duration = len(waveform) / SAMPLE_RATE
    entries  = []

    for chunk, actual_len, start_s, end_s in sliding_window_chunks(waveform):
        if is_silent(chunk, actual_len):
            continue

        chunk = apply_rms_normalisation(chunk, actual_len)

        wav_t = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE)
        len_t = torch.tensor([actual_len], dtype=torch.long).to(DEVICE)

        # NepaliASR.forward — Fact 4
        log_probs, frame_lengths = model(wav_t, len_t)

        # Greedy CTC decode — inline, no reimplementation of model internals
        ids      = log_probs[0, :frame_lengths[0].item()].argmax(dim=-1).tolist()
        out, prev = [], -1
        for idx in ids:
            if idx != blank_id and idx != prev:
                out.append(idx)
            prev = idx

        text = tokenizer.decode(out).strip()
        if text:
            entries.append({"start": round(start_s, 3), "end": round(end_s, 3), "text": text})

    return entries, duration


# ═════════════════════════════════════════════════════════════════════════════
#   SRT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def to_srt_ts(s: float) -> str:
    s  = max(0.0, s)
    hh = int(s // 3600);  s -= hh * 3600
    mm = int(s // 60);    s -= mm * 60
    ss = int(s);          ms = round((s - ss) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def build_srt(entries: list) -> str:
    """
    Build SRT with OS-appropriate line endings and encoding.
    Windows → CRLF + UTF-8-BOM  (VLC needs BOM for Devanagari)
    Linux/Mac → LF + UTF-8      (BOM causes garbage char on Unix players)
    """
    eol    = "\r\n" if os.name == "nt" else "\n"
    blocks = []
    for i, e in enumerate(entries, 1):
        blocks.append(
            f"{i}{eol}"
            f"{to_srt_ts(e['start'])} --> {to_srt_ts(e['end'])}{eol}"
            f"{e['text']}{eol}"
            f"{eol}"
        )
    return "".join(blocks)


# ═════════════════════════════════════════════════════════════════════════════
#   FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f    = request.files["file"]
    name = f.filename or "upload"
    ext  = Path(name).suffix.lower()

    if ext not in AUDIO_EXTS | VIDEO_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    job_id        = str(uuid.uuid4())[:8]
    original_stem = Path(name).stem

    raw_in   = UPLOAD_DIR / f"{job_id}_raw{ext}"
    wav_out  = AUDIO_DIR  / f"{job_id}.wav"
    srt_out  = OUTPUT_DIR / f"{job_id}.srt"
    meta_out = OUTPUT_DIR / f"{job_id}.json"

    f.save(str(raw_in))

    try:
        extract_to_wav(raw_in, wav_out)
    except Exception as e:
        raw_in.unlink(missing_ok=True)
        return jsonify({"error": f"Audio extraction failed: {e}"}), 500
    finally:
        raw_in.unlink(missing_ok=True)

    try:
        entries, duration = transcribe_file(wav_out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Transcription failed: {e}"}), 500

    encoding = "utf-8-sig" if os.name == "nt" else "utf-8"
    with open(str(srt_out), "w", encoding=encoding, newline="") as fh:
        fh.write(build_srt(entries))

    meta_out.write_text(json.dumps({"original_stem": original_stem}), encoding="utf-8")

    return jsonify({
        "audio_url":         f"/static/audio/{job_id}.wav",
        "duration":          round(duration, 3),
        "entries":           entries,
        "job_id":            job_id,
        "chunks_processed":  len(entries),
        "original_filename": original_stem,
    })


@app.route("/download_srt/<job_id>")
def download_srt(job_id: str):
    srt_path  = OUTPUT_DIR / f"{job_id}.srt"
    meta_path = OUTPUT_DIR / f"{job_id}.json"

    if not srt_path.exists():
        return jsonify({"error": "SRT not found"}), 404

    try:
        meta          = json.loads(meta_path.read_text(encoding="utf-8"))
        download_name = f"{meta['original_stem']}.srt"
    except Exception:
        download_name = f"subtitles_{job_id}.srt"

    return send_file(
        str(srt_path), as_attachment=True,
        download_name=download_name,
        mimetype="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    print(f"[ASR] Device: {DEVICE}")
    app.run(debug=True, host="0.0.0.0", port=5000)