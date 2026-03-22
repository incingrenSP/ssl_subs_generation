"""
Complex ASR inference script with KenLM beam decoder.

Usage: python complex_inference.py
"""

import json
import random
import unicodedata
import torch
import soundfile as sf
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT     = "deployment/asr_model_prototype.pt"
TOKENIZER_PATH = "data/tokenizer.json"
METADATA_PATH  = "data/metadata_normal.tsv"
KENLM_PATH     = "deployment/nepali_kenlm.bin"
NUM_SAMPLES    = 1000  # how many samples to test
OUTPUT_FILE    = "results.txt"
RANDOM_SEED    = 42   # set to None for different samples each run

# KenLM decoder params — tune these after first run
ALPHA          = 0.5   # LM weight (higher = trust LM more)
BETA           = 1.0   # word insertion bonus
BEAM_WIDTH     = 100   # beam size (higher = better but slower)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16_000
FRAME_STRIDE = 320


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)
    return waveform


# ── Metrics ───────────────────────────────────────────────────────────────────

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(ref, hyp):
    ref_w = ref.strip().split()
    hyp_w = hyp.strip().split()
    errors = edit_distance(ref_w, hyp_w)
    return errors, len(ref_w)


def compute_cer(ref, hyp):
    ref = unicodedata.normalize("NFC", ref.strip()).replace(" ", "")
    hyp = unicodedata.normalize("NFC", hyp.strip()).replace(" ", "")
    errors = edit_distance(list(ref), list(hyp))
    return errors, len(ref)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load tokenizer
    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        tok_data = json.load(f)
    id2char  = tok_data["id2char"]

    # Build vocab list for pyctcdecode
    # Must be ordered by token id, blank mapped to ""
    vocab_size = len(id2char)
    vocab = []
    for i in range(vocab_size):
        ch = id2char.get(str(i), "<unk>")
        if ch == "<blank>":
            vocab.append("")        # pyctcdecode expects blank as empty string
        elif ch == "<space>":
            vocab.append(" ")       # word boundary signal
        elif ch == "<pad>":
            vocab.append("<pad>")   # keep unique label
        elif ch == "<unk>":
            vocab.append("<unk>")   # keep unique label
        else:
            vocab.append(ch)

    # Load unigrams from metadata transcripts
    print("Loading unigrams from metadata...")
    unigrams = set()
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                for word in parts[1].split():
                    unigrams.add(word.strip())
    unigrams = list(unigrams)
    print(f"  {len(unigrams):,} unigrams loaded")

    # Load KenLM decoder
    print("Loading KenLM decoder...")
    from pyctcdecode import build_ctcdecoder
    lm_decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=KENLM_PATH,
        unigrams=unigrams,
        alpha=ALPHA,
        beta=BETA,
        unk_score_offset=-10.0,   # penalize unknown tokens heavily
        lm_score_boundary=True,   # only apply LM scores at word boundaries
    )
    print(f"KenLM decoder ready  (alpha={ALPHA}, beta={BETA}, beam={BEAM_WIDTH})\n")

    # Load ASR model
    from src.ssl_model import NepaliSSL, NepaliSSLConfig
    from src.asr_model import NepaliASR
    from src.tokenizer import GraphemeTokenizer

    print("Loading ASR model...")
    tokenizer   = GraphemeTokenizer.load(TOKENIZER_PATH)
    ckpt        = torch.load(CHECKPOINT, map_location="cpu")
    cfg_data    = ckpt.get("config", None)
    cfg         = NepaliSSLConfig(**cfg_data) if isinstance(cfg_data, dict) else NepaliSSLConfig()
    ssl_model   = NepaliSSL(cfg)
    model_state = ckpt["model"]
    asr_model   = NepaliASR(ssl_model, vocab_size=tokenizer.get_vocab_size(), freeze_encoder=False)
    asr_model.load_state_dict(model_state)
    asr_model.eval()
    print("Model ready.\n")

    # Load and shuffle metadata
    all_samples = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("path"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                all_samples.append((parts[0], parts[1]))

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    samples = all_samples[:NUM_SAMPLES]
    print(f"Running on {len(samples)} random samples...\n")

    # Accumulators
    wer_errors = cer_errors = total_words = total_chars = 0

    lines = []
    sep   = "=" * 70

    lines.append(sep)
    lines.append(f"  alpha={ALPHA}  beta={BETA}  beam={BEAM_WIDTH}  samples={NUM_SAMPLES}")
    lines.append(sep)

    for i, (audio_path, reference) in enumerate(samples, 1):
        try:
            waveform = load_audio(audio_path)
        except Exception as e:
            lines.append(f"[{i}] SKIP — {audio_path}: {e}")
            continue

        duration = waveform.shape[0] / SAMPLE_RATE

        with torch.no_grad():
            x       = waveform.unsqueeze(0)
            lengths = torch.tensor([waveform.shape[0]], dtype=torch.long)
            log_probs, frame_lengths = asr_model(x, lengths)

        T_valid  = int(frame_lengths[0].item())
        lp_slice = log_probs[0, :T_valid].float().cpu()

        # KenLM beam decode
        lm_text = lm_decoder.decode(lp_slice.numpy(), beam_width=BEAM_WIDTH)

        # Metrics
        w_e, w_n = compute_wer(reference, lm_text)
        c_e, c_n = compute_cer(reference, lm_text)
        wer_errors += w_e; total_words += w_n
        cer_errors += c_e; total_chars += c_n

        lines.append(f"[{i}] {audio_path}  ({duration:.1f}s)")
        lines.append(f"  REF : {reference}")
        lines.append(f"  HYP : {lm_text}")
        lines.append(f"  WER : {w_e/max(w_n,1)*100:.1f}%  ({w_e}/{w_n} word errors)")
        lines.append(f"  CER : {c_e/max(c_n,1)*100:.1f}%  ({c_e}/{c_n} char errors)")
        lines.append("")

    # Summary
    if total_words > 0:
        lines.append(sep)
        lines.append("SUMMARY")
        lines.append(f"  Samples : {len(samples)}")
        lines.append(f"  WER     : {wer_errors/total_words*100:.1f}%  ({wer_errors}/{total_words})")
        lines.append(f"  CER     : {cer_errors/total_chars*100:.1f}%  ({cer_errors}/{total_chars})")
        lines.append(f"  alpha={ALPHA}  beta={BETA}  beam={BEAM_WIDTH}")
        lines.append(sep)

    # Print + save
    for line in lines:
        print(line)

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()