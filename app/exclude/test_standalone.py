# test_standalone.py  — run this from your project root
import sys, torch, numpy as np, soundfile as sf
sys.path.insert(0, '.')

from src.tokenizer import GraphemeTokenizer
from src.asr_model import NepaliASR
from src.ssl_model import NepaliSSL, NepaliSSLConfig

TOKENIZER_PATH = 'vocab/tokenizer.json'
CHECKPOINT     = 'checkpoints/ssl_model.pt'
AUDIO_FILE     = 'nep_0258_0119737288.wav'   # ← same file you uploaded
DEVICE         = 'cpu'

# Load tokenizer
tok = GraphemeTokenizer.load(TOKENIZER_PATH)

# Load model
ckpt   = torch.load(CHECKPOINT, map_location=DEVICE)
raw_sd = ckpt['model'] if 'model' in ckpt else ckpt
cfg    = NepaliSSLConfig()
ssl    = NepaliSSL(cfg)
model  = NepaliASR(ssl, vocab_size=tok.get_vocab_size(), freeze_encoder=False)
model.load_state_dict(raw_sd)
model.eval()

# Load audio — NO normalisation, raw soundfile output
data, sr = sf.read(AUDIO_FILE, dtype='float32', always_2d=True)
data = data.mean(axis=1)
print(f"Audio: {len(data)} samples, {len(data)/16000:.2f}s, min={data.min():.4f} max={data.max():.4f}")

wav_t = torch.from_numpy(data).unsqueeze(0)
len_t = torch.tensor([len(data)], dtype=torch.long)

with torch.no_grad():
    log_probs, frame_lengths = model(wav_t, len_t)

n = frame_lengths[0].item()
ids = log_probs[0, :n].argmax(dim=-1).tolist()
out, prev = [], -1
for i in ids:
    if i != 0 and i != prev:
        out.append(i)
    prev = i

print(f"Raw argmax (first 40): {ids[:40]}")
print(f"Collapsed ids: {out}")
print(f"Decoded: '{tok.decode(out)}'")