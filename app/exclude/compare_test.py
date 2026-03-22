"""
compare_test.py
===============
Run this with the SAME audio file you uploaded to the web app.
It tests three variants side by side so you can see exactly
what difference (if any) preprocessing makes.

Usage:
    python compare_test.py path/to/your_audio.wav
"""
import sys, torch, numpy as np, soundfile as sf
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.tokenizer import GraphemeTokenizer, CharTokenizer
from src.asr_model import NepaliASR
from src.ssl_model import NepaliSSL, NepaliSSLConfig

TOKENIZER_PATH = 'vocab/tokenizer.json'
CHECKPOINT     = 'checkpoints/ssl_model.pt'
DEVICE         = 'cpu'
SAMPLE_RATE    = 16_000

# ── Load model once ──────────────────────────────────────────────────────────
import json
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)
tok_type = meta.get('tokenizer_type', 'char')
tok = (GraphemeTokenizer if tok_type == 'grapheme' else CharTokenizer).load(TOKENIZER_PATH)
print(f"Tokenizer: {tok_type}, {tok.get_vocab_size()} tokens")
print(f"blank_id={tok.blank_id}  pad_id={tok.pad_id}  "
      f"space_id={tok.char2id.get('<space>', 'NOT FOUND')}")

ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
raw_sd = ckpt['model'] if 'model' in ckpt else ckpt
cfg    = NepaliSSLConfig()
ssl    = NepaliSSL(cfg)
model  = NepaliASR(ssl, vocab_size=tok.get_vocab_size(), freeze_encoder=False)
model.load_state_dict(raw_sd)
model.eval()

# ── CTC greedy decode ─────────────────────────────────────────────────────────
def greedy(log_probs_2d, blank_id=0):
    ids = log_probs_2d.argmax(dim=-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != blank_id and i != prev:
            out.append(i)
        prev = i
    return out

# ── Run one variant ───────────────────────────────────────────────────────────
def run(label, waveform):
    print(f"\n{'─'*60}")
    print(f"VARIANT: {label}")
    print(f"  samples={len(waveform)}  min={waveform.min():.4f}  "
          f"max={waveform.max():.4f}  rms={np.sqrt(np.mean(waveform**2)):.4f}")

    wav_t = torch.from_numpy(waveform).unsqueeze(0)
    len_t = torch.tensor([len(waveform)], dtype=torch.long)

    with torch.no_grad():
        log_probs, frame_lengths = model(wav_t, len_t)

    n   = frame_lengths[0].item()
    lp  = log_probs[0, :n]

    raw = lp.argmax(dim=-1).tolist()
    space_id = tok.char2id.get('<space>', 3)
    nb = raw.count(tok.blank_id)
    ns = raw.count(space_id)
    nc = len(raw) - nb - ns
    print(f"  frames={n}  blank={nb}({100*nb//max(n,1)}%)  "
          f"space={ns}({100*ns//max(n,1)}%)  char={nc}({100*nc//max(n,1)}%)")
    print(f"  raw argmax (first 50): {raw[:50]}")

    collapsed = greedy(lp)
    print(f"  collapsed ids ({len(collapsed)}): {collapsed}")

    text = tok.decode(collapsed).strip()
    print(f"  DECODED: '{text}'")
    return text

# ── Load audio ────────────────────────────────────────────────────────────────
audio_path = sys.argv[1] if len(sys.argv) > 1 else None
if not audio_path:
    print("Usage: python compare_test.py path/to/audio.wav")
    sys.exit(1)

data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
data = data.mean(axis=1)
if sr != SAMPLE_RATE:
    from scipy.signal import resample_poly
    common = gcd(sr, SAMPLE_RATE)
    data   = resample_poly(data, SAMPLE_RATE // common, sr // common).astype(np.float32)

print(f"\nAudio file : {audio_path}")
print(f"Samples    : {len(data)}  ({len(data)/SAMPLE_RATE:.2f}s)")
print(f"Native SR  : {sr} Hz  →  resampled to {SAMPLE_RATE} Hz")

# Variant A — raw soundfile output (exactly like training)
run("A: raw (no normalisation) — matches training exactly", data.copy())

# Variant B — with DC removal + peak normalisation (what web app does)
normed = data.copy() - data.mean()
peak   = abs(normed).max()
if peak > 1e-6:
    normed = normed / peak
run("B: DC removed + peak normalised (what web app does)", normed)

# Variant C — only DC removal, no peak normalisation
dc_only = data.copy() - data.mean()
run("C: DC removed only", dc_only)

print("\n" + "="*60)
print("CONCLUSION:")
print("  If A is best → remove normalize_amplitude() from app.py")
print("  If B is best → keep normalize_amplitude() as-is")
print("  If A == B    → normalisation has no effect on this model")
print("="*60)