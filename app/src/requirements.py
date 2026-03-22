"""
src/requirements.py  — inference-safe shim
==========================================
The original requirements.py contains:
    from torchaudio.models.decoder import ctc_decoder   # line 12

torchaudio's ctc_decoder requires the 'flashlight-text' C++ package.
That package is only needed for beam-search decoding, which the web app
does NOT use (we use greedy CTC decode instead).

This shim re-exports everything the rest of the src.* modules need but
replaces the ctc_decoder import with a stub that raises a clear error only
if someone actually tries to call it — not at import time.

HOW TO USE:
  Place this file at  <project_root>/src/requirements.py
  It will shadow the original file when Python resolves  'from src.requirements import *'
  Your original requirements.py is NOT modified.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import os
import glob
import random
import copy
import math
import json
import pickle
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, LinearLR, SequentialLR,
)

# ── torchaudio (safe subset — no ctc_decoder) ────────────────────────────────
import torchaudio
import torchaudio.transforms as T

# ctc_decoder stub — raises only if actually called, not at import time
class _CTCDecoderStub:
    def __init__(self, *a, **kw):
        raise RuntimeError(
            "ctc_decoder (flashlight-text) is not available in this environment. "
            "The web app uses greedy CTC decoding instead — you should not be "
            "hitting this code path. Check that asr_metrics.py beam functions "
            "are not being called from app.py."
        )

ctc_decoder = _CTCDecoderStub

# ── Scientific / data ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import soundfile as sf

# librosa import is optional — used only for training utilities, not inference
try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

# matplotlib / tqdm — optional at inference
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore

# ── Profiler ──────────────────────────────────────────────────────────────────
try:
    from torch.profiler import profile, ProfilerActivity
except ImportError:
    profile = None          # type: ignore
    ProfilerActivity = None # type: ignore

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
TOP_DB = 30

# ── Constants copied verbatim from original requirements.py ──────────────────
DIGIT_MAP = {
    # Devanagari
    '०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9',
    # Arabic-Indic (Persian/Urdu)
    '۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','۶':'6','۷':'7','८':'8','९':'9',
    # Fullwidth
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    # Bengali
    '০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','९':'9',
    # Khmer
    '០':'0','១':'1','២':'2','៣':'3','៤':'4','៥':'5','៦':'6','៧':'7','៨':'8','៩':'9',
}

TYPOGRAPHIC_REMOVE = set([
    '¹','²','³',
    '❶','❷','❸','❹','❺',
    '①','②','③','④','⑤',
    '⑴','⑵','⑶','⑷','⑸',
])
