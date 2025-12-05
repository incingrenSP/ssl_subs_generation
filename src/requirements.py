import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.rnn as rnn_utils
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import torchaudio
import torchaudio.transforms as T

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
import glob
import random
import regex
import unicodedata
import re
from collections import Counter
import json
import random

import pandas as pd
import numpy as np

import soundfile as sf
import librosa

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
TOP_DB = 30