"""
Nepali ASR — Deployment-Ready Inference Module
================================================
CPU-optimised inference with sliding window support for long audio.

Outputs:
  - Plain transcript string
  - Word-level timestamps (seconds)

Usage:
    from inference import NepaliASRInference

    asr = NepaliASRInference(
        checkpoint   = "checkpoints/asr/checkpoint_0058000_best.pt",
        tokenizer    = "data/tokenizer.json",
        ssl_config   = None,   # loaded from checkpoint automatically
    )

    # Short clip (<= max_duration_s)
    result = asr.transcribe("audio.wav")
    print(result.text)
    print(result.words)   # [WordResult(word, start_s, end_s, confidence), ...]

    # Long audio — automatic sliding window
    result = asr.transcribe("long_broadcast.wav")

    # Benchmark
    asr.benchmark("audio.wav", runs=20)

Requirements:
    pip install torch torchaudio numpy
    (optional, for resampling non-16kHz audio)
"""

import time
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import sys, os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

log = logging.getLogger(__name__)

SAMPLE_RATE    = 16_000
FRAME_STRIDE   = 320          # CNN total stride — 20ms per frame at 16kHz
FRAME_RATE     = SAMPLE_RATE / FRAME_STRIDE   # 50 Hz


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class WordResult:
    word:       str
    start_s:    float
    end_s:      float
    confidence: float   # mean frame probability of the emitted tokens, 0–1

    def __repr__(self):
        return f"WordResult({self.word!r}, {self.start_s:.2f}s–{self.end_s:.2f}s, conf={self.confidence:.2f})"


@dataclass
class TranscriptResult:
    text:          str
    words:         List[WordResult]
    duration_s:    float
    rtf:           float    # real-time factor: inference_time / audio_duration
    num_chunks:    int      # 1 for short audio, >1 for sliding window

    def __repr__(self):
        return (
            f"TranscriptResult(\n"
            f"  text={self.text!r}\n"
            f"  words={len(self.words)} words\n"
            f"  duration={self.duration_s:.2f}s  rtf={self.rtf:.3f}  chunks={self.num_chunks}\n"
            f")"
        )


@dataclass
class BenchmarkResult:
    mean_rtf:     float
    std_rtf:      float
    min_rtf:      float
    max_rtf:      float
    mean_ms:      float
    audio_duration_s: float
    runs:         int

    def __repr__(self):
        return (
            f"BenchmarkResult over {self.runs} runs on {self.audio_duration_s:.2f}s audio:\n"
            f"  RTF  mean={self.mean_rtf:.4f}  std={self.std_rtf:.4f}  "
            f"min={self.min_rtf:.4f}  max={self.max_rtf:.4f}\n"
            f"  Latency  mean={self.mean_ms:.1f}ms"
        )


# ── CTC greedy decoder with timestamps ───────────────────────────────────────

def ctc_decode_with_timestamps(
    log_probs:  torch.Tensor,    # (T, vocab_size)  — float32, CPU
    id2char:    dict,
    blank_id:   int,
    space_id:   int,
    frame_rate: float = FRAME_RATE,
    offset_s:   float = 0.0,
) -> Tuple[str, List[WordResult]]:
    """
    Greedy CTC decode with per-word timestamp and confidence estimation.

    Timestamps are derived from frame positions of the first and last
    non-blank, non-repeated emission of each token within a word.
    Confidence is the mean softmax probability of each emitted token
    across its span of dominant frames.
    """
    T, V = log_probs.shape
    probs = log_probs.exp()    # (T, V)

    # ── Greedy collapse ───────────────────────────────────────────────────────
    # Build list of (token_id, first_frame, last_frame, mean_prob)
    tokens_with_frames = []
    prev_id = -1
    span_start = 0

    for t in range(T):
        tok = int(log_probs[t].argmax().item())
        prob = float(probs[t, tok].item())

        if tok == blank_id:
            prev_id = -1
            continue

        if tok != prev_id:
            tokens_with_frames.append([tok, t, t, prob, 1])
            prev_id = tok
        else:
            # Extend current span
            tokens_with_frames[-1][2] = t
            tokens_with_frames[-1][3] += prob
            tokens_with_frames[-1][4] += 1

    # Finalise mean confidence per token
    decoded_tokens = []
    for tok_id, f_start, f_end, prob_sum, count in tokens_with_frames:
        if tok_id == blank_id:
            continue
        decoded_tokens.append((tok_id, f_start, f_end, prob_sum / count))

    if not decoded_tokens:
        return "", []

    # ── Group into words by space token ──────────────────────────────────────
    words: List[WordResult] = []
    current_chars  = []
    current_frames = []
    current_confs  = []

    def flush_word():
        if not current_chars:
            return
        text = "".join(
            " " if id2char.get(c, "") == "<space>" else id2char.get(c, "?")
            for c in current_chars
        ).strip()
        if not text:
            return
        start_s = offset_s + current_frames[0]  / frame_rate
        end_s   = offset_s + current_frames[-1] / frame_rate
        conf    = float(np.mean(current_confs))
        words.append(WordResult(text, start_s, end_s, conf))
        current_chars.clear()
        current_frames.clear()
        current_confs.clear()

    for tok_id, f_start, f_end, conf in decoded_tokens:
        if tok_id == space_id:
            flush_word()
        else:
            current_chars.append(tok_id)
            current_frames.extend([f_start, f_end])
            current_confs.append(conf)

    flush_word()

    text = " ".join(w.word for w in words)
    return text, words


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    """Load ASR model and tokenizer from disk."""
    from src.ssl_model import NepaliSSL, NepaliSSLConfig
    from src.model import NepaliASR
    from src.tokenizer import GraphemeTokenizer

    tokenizer = GraphemeTokenizer.load(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    cfg_data = ckpt.get("config", None)
    if isinstance(cfg_data, dict):
        cfg = NepaliSSLConfig(**cfg_data)
    else:
        cfg = NepaliSSLConfig()

    ssl_model = NepaliSSL(cfg)

    # Strip ASR head weights to get just the encoder
    model_state = ckpt["model"]
    ssl_keys = {k: v for k, v in model_state.items()
                if k.startswith("encoder.") or k.startswith("context.")}

    # Build a temporary SSL model to hold encoder weights
    ssl_model.load_state_dict(
        {k.replace("encoder.", "cnn.").replace("context.", "context."): v
         for k, v in ssl_keys.items()},
        strict=False
    )

    asr_model = NepaliASR(ssl_model, vocab_size=tokenizer.get_vocab_size(), freeze_encoder=False)
    asr_model.load_state_dict(model_state)
    asr_model.eval()

    return asr_model, tokenizer


# ── Audio loading and preprocessing ──────────────────────────────────────────

def load_audio(path: str) -> Tuple[torch.Tensor, float]:
    """
    Load audio file, resample to 16kHz mono if necessary.
    Returns (waveform_1d, duration_seconds).
    """
    waveform, sr = torchaudio.load(path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)

    waveform    = waveform.squeeze(0)   # (T_samples,)
    duration_s  = waveform.shape[0] / SAMPLE_RATE
    return waveform, duration_s


# ── Main inference class ──────────────────────────────────────────────────────

class NepaliASRInference:
    """
    Deployment-ready Nepali ASR inference.

    Parameters
    ----------
    checkpoint    : path to checkpoint_XXXXXXX_best.pt
    tokenizer     : path to tokenizer.json
    ssl_config    : NepaliSSLConfig instance or None (auto from checkpoint)
    max_duration_s: maximum audio length before switching to sliding window (default 20s)
    window_s      : sliding window length in seconds (default 18s)
    overlap_s     : overlap between adjacent windows in seconds (default 2s)
    num_threads   : number of CPU threads for PyTorch (default: all available)
    """

    def __init__(
        self,
        checkpoint:     str,
        tokenizer:      str,
        ssl_config      = None,
        max_duration_s: float = 20.0,
        window_s:       float = 18.0,
        overlap_s:      float = 2.0,
        num_threads:    Optional[int] = None,
    ):
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        log.info(f"Loading checkpoint: {checkpoint}")
        self.model, self.tokenizer = _load_model_and_tokenizer(checkpoint, tokenizer)
        self.model.eval()

        # Optimise for CPU inference
        self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model)) \
            if False else self.model   # TorchScript optional — disable if model has dynamic control flow

        self.max_duration_s = max_duration_s
        self.window_s       = window_s
        self.overlap_s      = overlap_s

        self.blank_id = self.tokenizer.blank_id
        self.space_id = self.tokenizer.char2id[self.tokenizer.SPACE_TOKEN]
        self.id2char  = self.tokenizer.id2char

        log.info(
            f"Model ready — vocab={self.tokenizer.get_vocab_size()} "
            f"blank={self.blank_id} space={self.space_id} "
            f"threads={torch.get_num_threads()}"
        )

    # ── Internal: single chunk inference ─────────────────────────────────────

    @torch.no_grad()
    def _infer_chunk(
        self,
        waveform: torch.Tensor,   # (T_samples,) float32
        offset_s: float = 0.0,
    ) -> Tuple[str, List[WordResult]]:
        """Run model on a single waveform chunk, return text and word list."""
        x       = waveform.unsqueeze(0)   # (1, T)
        lengths = torch.tensor([waveform.shape[0]], dtype=torch.long)

        log_probs, frame_lengths = self.model(x, lengths)
        # log_probs: (1, T_frames, vocab)

        T_valid   = int(frame_lengths[0].item())
        lp_slice  = log_probs[0, :T_valid].float().cpu()   # (T_valid, vocab)

        text, words = ctc_decode_with_timestamps(
            lp_slice, self.id2char, self.blank_id, self.space_id,
            frame_rate=FRAME_RATE, offset_s=offset_s,
        )
        return text, words

    # ── Internal: sliding window ──────────────────────────────────────────────

    def _infer_sliding(
        self,
        waveform:   torch.Tensor,
        duration_s: float,
    ) -> Tuple[str, List[WordResult], int]:
        """
        Chunk long audio into overlapping windows, decode each, stitch results.
        Returns (full_text, all_words, num_chunks).
        """
        window_samples  = int(self.window_s  * SAMPLE_RATE)
        overlap_samples = int(self.overlap_s * SAMPLE_RATE)
        step_samples    = window_samples - overlap_samples

        all_words: List[WordResult] = []
        chunk_idx = 0
        pos = 0

        while pos < waveform.shape[0]:
            end = min(pos + window_samples, waveform.shape[0])
            chunk = waveform[pos:end]

            # Pad short final chunk to at least 1 frame
            if chunk.shape[0] < FRAME_STRIDE:
                break

            offset_s = pos / SAMPLE_RATE
            _, words = self._infer_chunk(chunk, offset_s=offset_s)

            # De-duplicate words in the overlap zone from the previous chunk.
            # Keep words whose start time is beyond (previous_end - overlap_s/2).
            if chunk_idx > 0 and all_words:
                prev_end_s  = (pos + overlap_samples) / SAMPLE_RATE
                cutoff_s    = prev_end_s - self.overlap_s / 2
                words = [w for w in words if w.start_s >= cutoff_s]

            all_words.extend(words)
            chunk_idx += 1

            if end >= waveform.shape[0]:
                break
            pos += step_samples

        full_text = " ".join(w.word for w in all_words)
        return full_text, all_words, chunk_idx

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """
        Transcribe a WAV or MP3 file.

        Automatically uses sliding window for audio longer than max_duration_s.

        Parameters
        ----------
        audio_path : path to audio file (WAV, MP3, FLAC, etc.)

        Returns
        -------
        TranscriptResult with .text (str) and .words (List[WordResult])
        """
        waveform, duration_s = load_audio(audio_path)

        t_start = time.perf_counter()

        if duration_s <= self.max_duration_s:
            text, words = self._infer_chunk(waveform, offset_s=0.0)
            num_chunks  = 1
        else:
            log.info(
                f"Audio {duration_s:.1f}s > {self.max_duration_s}s — "
                f"using sliding window ({self.window_s}s / {self.overlap_s}s overlap)"
            )
            text, words, num_chunks = self._infer_sliding(waveform, duration_s)

        elapsed_s = time.perf_counter() - t_start
        rtf       = elapsed_s / max(duration_s, 1e-6)

        return TranscriptResult(
            text       = text,
            words      = words,
            duration_s = duration_s,
            rtf        = rtf,
            num_chunks = num_chunks,
        )

    def transcribe_waveform(
        self,
        waveform:   torch.Tensor,   # (T_samples,) float32, already 16kHz
        duration_s: Optional[float] = None,
    ) -> TranscriptResult:
        """
        Transcribe a pre-loaded waveform tensor directly.
        Useful for streaming or when audio is already in memory.
        """
        if duration_s is None:
            duration_s = waveform.shape[0] / SAMPLE_RATE

        t_start = time.perf_counter()

        if duration_s <= self.max_duration_s:
            text, words = self._infer_chunk(waveform, offset_s=0.0)
            num_chunks  = 1
        else:
            text, words, num_chunks = self._infer_sliding(waveform, duration_s)

        elapsed_s = time.perf_counter() - t_start
        rtf       = elapsed_s / max(duration_s, 1e-6)

        return TranscriptResult(
            text       = text,
            words      = words,
            duration_s = duration_s,
            rtf        = rtf,
            num_chunks = num_chunks,
        )

    def benchmark(self, audio_path: str, runs: int = 20, warmup: int = 3) -> BenchmarkResult:
        """
        Measure inference latency and real-time factor over multiple runs.

        Parameters
        ----------
        audio_path : path to audio file to benchmark on
        runs       : number of timed runs
        warmup     : number of untimed warmup runs before measurement

        Returns
        -------
        BenchmarkResult with mean/std/min/max RTF and latency in ms
        """
        waveform, duration_s = load_audio(audio_path)

        log.info(f"Benchmarking on {duration_s:.2f}s audio — {warmup} warmup + {runs} timed runs")

        # Warmup
        for _ in range(warmup):
            self.transcribe_waveform(waveform, duration_s)

        # Timed runs
        times = []
        for i in range(runs):
            t0 = time.perf_counter()
            self.transcribe_waveform(waveform, duration_s)
            times.append(time.perf_counter() - t0)

        times    = np.array(times)
        rtfs     = times / duration_s

        result = BenchmarkResult(
            mean_rtf         = float(rtfs.mean()),
            std_rtf          = float(rtfs.std()),
            min_rtf          = float(rtfs.min()),
            max_rtf          = float(rtfs.max()),
            mean_ms          = float(times.mean() * 1000),
            audio_duration_s = duration_s,
            runs             = runs,
        )
        print(result)
        return result

    def profile_components(self, audio_path: str) -> dict:
        """
        Break down inference time into CNN encoder, transformer, and CTC head.
        Useful for identifying the bottleneck before optimisation.
        """
        waveform, duration_s = load_audio(audio_path)
        x       = waveform.unsqueeze(0)
        lengths = torch.tensor([waveform.shape[0]], dtype=torch.long)

        timings = {}

        with torch.no_grad():
            # CNN encoder
            t0 = time.perf_counter()
            features = self.model.encoder(x)
            timings["cnn_ms"] = (time.perf_counter() - t0) * 1000

            T = features.shape[1]
            frame_lengths = (lengths.float() / x.shape[1] * T).long().clamp(max=T)
            pad_mask = (
                torch.arange(T).unsqueeze(0) >= frame_lengths.unsqueeze(1)
            )

            # Transformer
            t0 = time.perf_counter()
            context = self.model.context(features, padding_mask=pad_mask)
            timings["transformer_ms"] = (time.perf_counter() - t0) * 1000

            # CTC head + softmax
            t0 = time.perf_counter()
            context_normed = self.model.pre_head_norm(context)
            logits    = self.model.ctc_head(context_normed)
            log_probs = F.log_softmax(logits, dim=-1)
            timings["ctc_head_ms"] = (time.perf_counter() - t0) * 1000

            # Decode
            T_valid  = int(frame_lengths[0].item())
            lp_slice = log_probs[0, :T_valid].float().cpu()
            t0 = time.perf_counter()
            ctc_decode_with_timestamps(
                lp_slice, self.id2char, self.blank_id, self.space_id
            )
            timings["decode_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"]      = sum(timings.values())
        timings["audio_duration_s"] = duration_s
        timings["rtf"]           = timings["total_ms"] / 1000 / duration_s

        print(f"\nComponent breakdown ({duration_s:.2f}s audio):")
        for k, v in timings.items():
            if k.endswith("_ms"):
                pct = v / timings["total_ms"] * 100
                print(f"  {k:<20s} {v:7.2f} ms  ({pct:.1f}%)")
        print(f"  {'rtf':<20s} {timings['rtf']:.4f}")

        return timings


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # parser = argparse.ArgumentParser(description="Nepali ASR inference")
    # parser.add_argument("audio",        help="Path to audio file")
    # parser.add_argument("--checkpoint", required=True, help="Path to ASR checkpoint .pt")
    # parser.add_argument("--tokenizer",  required=True, help="Path to tokenizer.json")
    # parser.add_argument("--benchmark",  type=int, default=0, metavar="N",
    #                     help="Run N benchmark iterations instead of single transcription")
    # parser.add_argument("--profile",    action="store_true",
    #                     help="Profile time breakdown by component")
    # parser.add_argument("--max-duration", type=float, default=20.0,
    #                     help="Max audio duration before sliding window (default: 20s)")
    # parser.add_argument("--window",     type=float, default=18.0,
    #                     help="Sliding window size in seconds (default: 18s)")
    # parser.add_argument("--overlap",    type=float, default=2.0,
    #                     help="Window overlap in seconds (default: 2s)")
    # parser.add_argument("--threads",    type=int, default=None,
    #                     help="Number of CPU threads (default: all available)")
    # parser.add_argument("--json",       action="store_true",
    #                     help="Output result as JSON")
    # args = parser.parse_args()

    audio = os.path.join("test.flac")
    checkpoint = os.path.join("deployment", "asr_model_prototype.pt")
    tokenizer = os.path.join("data", "tokenizer.json")
    max_duration = 20.0
    window = 18.0
    overlap = 2.0
    threads = None
    profile = True
    benchmark = 20
    store_json = True

    asr = NepaliASRInference(
        checkpoint     = checkpoint,
        tokenizer      = tokenizer,
        max_duration_s = max_duration,
        window_s       = window,
        overlap_s      = overlap,
        num_threads    = threads,
    )

    if profile:
        asr.profile_components(audio)

    elif args.benchmark > 0:
        asr.benchmark(audio, runs=benchmark)

    else:
        result = asr.transcribe(audio)

        if store_json:
            out = {
                "text":       result.text,
                "duration_s": result.duration_s,
                "rtf":        result.rtf,
                "num_chunks": result.num_chunks,
                "words": [
                    {
                        "word":       w.word,
                        "start_s":    round(w.start_s, 3),
                        "end_s":      round(w.end_s, 3),
                        "confidence": round(w.confidence, 4),
                    }
                    for w in result.words
                ],
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"\nTranscript:\n{result.text}\n")
            print(f"Duration: {result.duration_s:.2f}s  |  RTF: {result.rtf:.4f}  |  Chunks: {result.num_chunks}")
            print(f"\nWord timestamps:")
            for w in result.words:
                print(f"  [{w.start_s:6.2f}s – {w.end_s:6.2f}s]  {w.word:<30s}  conf={w.confidence:.2f}")
