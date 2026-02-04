import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.rnn as rnn_utils
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR

import torchaudio
import torchaudio.transforms as T
from torchaudio.models.decoder import ctc_decoder

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
import glob
import random
import unicodedata
import re
from collections import Counter
import json
import random
import copy
import math
import jiwer
import pickle
from pathlib import Path
import time

import pandas as pd
import numpy as np

import soundfile as sf
import librosa

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
TOP_DB = 30

# A complete map based on the junk found in your vocab
DIGIT_MAP = {
    # Devanagari
    '०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9',
    # Arabic-Indic (Persian/Urdu)
    '۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','۶':'6','۷':'7','८':'8','९':'9',
    # Fullwidth
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    # Bengali (as seen in your IDs 118, 125, 134)
    '০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','९':'9',
    # Khmer (as seen in your IDs 111, 126, 120)
    '០':'0','១':'1','២':'2','៣':'3','៤':'4','៥':'5','៦':'6','៧':'7','៨':'8','៩':'9'
}

TYPOGRAPHIC_REMOVE = set([
    '¹','²','³',
    '❶','❷','❸','❹','❺',
    '①','②','③','④','⑤',
    '⑴','⑵','⑶','⑷','⑸'
])
