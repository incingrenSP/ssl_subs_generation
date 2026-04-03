"""
py -3.9 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

"""

import json
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import soundfile as sf
import imageio_ffmpeg
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# CONFIG
CHECKPOINT     = "deployment/asr_model_prototype.pt"
TOKENIZER_PATH = "data/tokenizer.json"
METADATA_PATH  = "data/metadata_normal.tsv"
KENLM_PATH     = "deployment/nepali_kenlm_3g.bin"
ALPHA          = 0.5
BETA           = 0.1
BEAM_WIDTH     = 100
SAMPLE_RATE    = 16_000
WINDOW_S       = 4.0
OVERLAP_S      = 0.0
FRAME_STRIDE   = 320

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

JOBS_DIR = Path("asr_jobs")
JOBS_DIR.mkdir(exist_ok=True)

# Model globals
_model      = None
_lm_decoder = None


def load_models():
    global _model, _lm_decoder

    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        tok_data = json.load(f)
    id2char  = tok_data["id2char"]
    vocab = []
    for i in range(len(id2char)):
        ch = id2char.get(str(i), "<unk>")
        if ch == "<blank>":   vocab.append("")
        elif ch == "<space>": vocab.append(" ")
        else:                 vocab.append(ch)

    from src.ssl_model import NepaliSSL, NepaliSSLConfig
    from src.asr_model import NepaliASR
    from src.tokenizer import GraphemeTokenizer
    from pyctcdecode import build_ctcdecoder

    tokenizer   = GraphemeTokenizer.load(TOKENIZER_PATH)
    ckpt        = torch.load(CHECKPOINT, map_location="cpu")
    cfg_data    = ckpt.get("config", None)
    cfg         = NepaliSSLConfig(**cfg_data) if isinstance(cfg_data, dict) else NepaliSSLConfig()
    ssl_model   = NepaliSSL(cfg)
    model_state = ckpt["model"]
    asr_model   = NepaliASR(ssl_model, vocab_size=tokenizer.get_vocab_size(), freeze_encoder=False)
    asr_model.load_state_dict(model_state)
    asr_model.eval()

    unigrams = set()
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                for word in parts[1].split():
                    unigrams.add(word)
    print(f"Unigrams: {len(unigrams):,}")

    lm_decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=KENLM_PATH,
        unigrams=list(unigrams),
        alpha=ALPHA,
        beta=BETA,
        unk_score_offset=-10.0,
        lm_score_boundary=True,
    )

    _model      = asr_model
    _lm_decoder = lm_decoder
    print("Model ready.")


# Inference helpers

def _infer_chunk(waveform):
    with torch.no_grad():
        x       = waveform.unsqueeze(0)
        lengths = torch.tensor([waveform.shape[0]], dtype=torch.long)
        log_probs, frame_lengths = _model(x, lengths)
    T_valid  = int(frame_lengths[0].item())
    lp_slice = log_probs[0, :T_valid].float().cpu()
    return _lm_decoder.decode(lp_slice.numpy(), beam_width=BEAM_WIDTH)


def _dedupe_overlap(prev_words, curr_words, overlap_n=4):
    if not prev_words or not curr_words:
        return curr_words
    for n in range(min(overlap_n, len(prev_words), len(curr_words)), 0, -1):
        if prev_words[-n:] == curr_words[:n]:
            return curr_words[n:]
    return curr_words


def run_transcription(audio_path: Path):
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    data     = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)

    duration_s      = waveform.shape[0] / SAMPLE_RATE
    window_samples  = int(WINDOW_S  * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_S * SAMPLE_RATE)
    step_samples    = window_samples - overlap_samples

    entries    = []
    prev_words = []
    pos        = 0
    chunks_processed = 0

    while pos < waveform.shape[0]:
        end   = min(pos + window_samples, waveform.shape[0])
        chunk = waveform[pos:end]
        if chunk.shape[0] < FRAME_STRIDE:
            break

        start_s = pos / SAMPLE_RATE
        end_s   = end / SAMPLE_RATE
        text    = _infer_chunk(chunk).strip()
        curr_words = text.split()
        if OVERLAP_S > 0:
            curr_words = _dedupe_overlap(prev_words, curr_words, overlap_n=4)

        if curr_words:
            entries.append({
                "start": round(start_s, 2),
                "end":   round(end_s,   2),
                "text":  " ".join(curr_words),
            })

        prev_words = text.split()
        chunks_processed += 1

        if end >= waveform.shape[0]:
            break
        pos += step_samples

    return entries, duration_s, chunks_processed


def build_srt(entries):
    def fmt(s):
        s = max(0.0, s)
        h = int(s // 3600); s -= h * 3600
        m = int(s // 60);   s -= m * 60
        sec = int(s);       ms = int((s - sec) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{fmt(e['start'])} --> {fmt(e['end'])}")
        lines.append(e["text"])
        lines.append("")
    return "\n".join(lines)


def extract_audio(input_path: Path, output_path: Path):
    import subprocess
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", str(input_path),
         "-ar", "16000", "-ac", "1", str(output_path)],
        check=True, capture_output=True
    )


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()   # startup
    yield
    # shutdown logic


# FastAPI app

app = FastAPI(title="Nepali ASR", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    html = Path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    suffix   = Path(file.filename).suffix.lower()
    tmp      = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    wav_path = None

    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts"}
        AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".m4a", ".aac"}

        if suffix in VIDEO_EXTS:
            # extract audio track from video
            wav_path   = tmp_path.with_suffix(".wav")
            extract_audio(tmp_path, wav_path)
            audio_path = wav_path
        elif suffix in AUDIO_EXTS:
            # convert non-WAV audio to 16kHz mono WAV
            wav_path   = tmp_path.with_suffix(".wav")
            extract_audio(tmp_path, wav_path)
            audio_path = wav_path
        else:
            # assume it's already a WAV
            audio_path = tmp_path

        # run transcription
        entries, duration, chunks = run_transcription(audio_path)

        # save job files
        job_id  = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir()

        # copy audio for playback
        audio_out = job_dir / "audio.wav"
        if audio_path.suffix == ".wav":
            shutil.copy(audio_path, audio_out)
        else:
            extract_audio(audio_path, audio_out)

        # save srt
        srt_path = job_dir / "subtitles.srt"
        srt_path.write_text(build_srt(entries), encoding="utf-8")

        return JSONResponse({
            "job_id":           job_id,
            "duration":         round(duration, 2),
            "chunks_processed": chunks,
            "entries":          entries,
            "audio_url":        f"/audio/{job_id}",
        })

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try: tmp_path.unlink()
        except: pass
        if wav_path is not None:
            try: wav_path.unlink()
            except: pass


@app.get("/audio/{job_id}")
def get_audio(job_id: str):
    path = JOBS_DIR / job_id / "audio.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/download_srt/{job_id}")
def download_srt(job_id: str):
    path = JOBS_DIR / job_id / "subtitles.srt"
    if not path.exists():
        raise HTTPException(404, "SRT not found")
    return FileResponse(
        str(path),
        media_type="text/plain",
        filename="subtitles.srt",
    )