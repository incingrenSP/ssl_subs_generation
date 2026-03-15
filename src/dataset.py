from src.requirements import *
from typing import Optional, Dict, List
import hashlib

def _load_audio_sf(path: str, target_sr: int) -> torch.Tensor:
    # always_2d=True → data is always (frames, channels)
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)

    # Mix down to mono: (frames, channels) → (frames,)
    data = data.mean(axis=1)

    # Resample only if needed
    if sr != target_sr:
        common = gcd(sr, target_sr)
        up     = target_sr // common
        down   = sr        // common
        data   = resample_poly(data, up, down).astype(np.float32)

    return torch.from_numpy(data)


def _audio_duration_sf(path: str) -> float:
    info = sf.info(str(path))
    return info.duration   # sndfile computes frames / samplerate internally


# Dataset
class NepaliSpeechDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        audio_base_dir: str,
        use_memory_map: bool = False,
        cache_dir: Optional[str] = None,
        use_spectrogram: bool = False,
        target_sr: int = 16000,
        max_length: Optional[float] = None,
        for_ssl: bool = False,
        n_fft: int = 400,
        n_mels: int = 80,
        hop_length: int = 160,
    ):
        super().__init__()

        self.metadata_path   = Path(metadata_path)
        self.audio_base_dir  = Path(audio_base_dir)
        self.use_memory_map  = use_memory_map
        self.cache_dir       = Path(cache_dir) if cache_dir else self.audio_base_dir / "cache"
        self.use_spectrogram = use_spectrogram
        self.target_sr       = target_sr
        self.max_length      = max_length
        self.for_ssl         = for_ssl
        self.n_fft           = n_fft
        self.n_mels          = n_mels
        self.hop_length      = hop_length

        self.metadata = pd.read_csv(
            metadata_path,
            sep='\t',
            names=['path', 'transcript'],
            header=0 if self._has_header(metadata_path) else None,
        )
        print(f"Loaded {len(self.metadata)} samples from {metadata_path}")

        if max_length is not None:
            self._filter_by_length()

        if use_memory_map:
            self._setup_memory_map()

        if use_spectrogram:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0.0,
                f_max=target_sr / 2.0,
            )

    def _has_header(self, path: str) -> bool:
        with open(path, 'r', encoding='utf-8') as f:
            first = f.readline().strip()
        return ('path' in first.lower()) and ('transcript' in first.lower())

    def _reconstruct_path(self, filename: str) -> Path:
        filename = Path(str(filename).strip()).name   # drop any accidental dir prefix
        subdir   = filename[:2]                       # e.g. "00", "01"
        return self.audio_base_dir / "audio" / subdir / filename

    def _filter_by_length(self):
        print(f"Filtering: keeping samples ≤ {self.max_length} s …")
        valid = []
        for idx, row in self.metadata.iterrows():
            audio_path = self._reconstruct_path(row['path'])
            try:
                duration = _audio_duration_sf(str(audio_path))
                if duration <= self.max_length:
                    valid.append(idx)
            except Exception as exc:
                print(f"  Warning – cannot inspect {audio_path}: {exc}")
        self.metadata = self.metadata.iloc[valid].reset_index(drop=True)
        print(f"  Kept {len(self.metadata)} samples after length filtering.")

    def _get_cache_path(self) -> Path:
        cfg  = f"{self.metadata_path}_{self.target_sr}_{self.use_spectrogram}"
        h    = hashlib.md5(cfg.encode()).hexdigest()[:8]
        return self.cache_dir / f"cache_{h}.npz"

    def _setup_memory_map(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Loading waveform cache from {cache_path} …")
            try:
                with np.load(str(cache_path), allow_pickle=False) as data:
                    self.cache_data = {}
                    for key in data.files:
                        idx = int(key.split('_')[1])
                        arr = data[key]
                        # Empty array = invalid sample
                        self.cache_data[idx] = arr if arr.size > 0 else None
                print(f"  Cache contains {len(self.cache_data)} entries.")
                return
            except Exception as exc:
                print(f"  Warning: cache file corrupted or unreadable ({exc})")
                print(f"  Rebuilding cache from scratch …")
                # Fall through to rebuild

        print("Building waveform cache (one-time, may take a while) …")
        self.cache_data: Dict[int, Optional[np.ndarray]] = {}

        for idx, row in self.metadata.iterrows():
            audio_path = self._reconstruct_path(row['path'])
            try:
                waveform = _load_audio_sf(str(audio_path), self.target_sr)
                self.cache_data[idx] = waveform.numpy()
            except Exception as exc:
                print(f"  Error caching {audio_path}: {exc}")
                # Store as empty array so the key exists
                self.cache_data[idx] = None

            if (idx + 1) % 100 == 0:
                print(f"  Cached {idx + 1} / {len(self.metadata)} …")

        # Save as .npz (compressed, robust)
        print(f"Saving cache → {cache_path}")
        arrays_to_save = {}
        for idx, arr in self.cache_data.items():
            # Store valid arrays as-is; invalid as empty float32 array
            arrays_to_save[f"audio_{idx}"] = (
                arr if arr is not None else np.array([], dtype=np.float32)
            )

        np.savez_compressed(str(cache_path), **arrays_to_save)
        print("  Cache saved successfully.")


    def _load_audio(self, idx: int) -> torch.Tensor:
        if self.use_memory_map:
            arr = self.cache_data.get(idx)
            if arr is None:
                raise ValueError(f"Cache entry {idx} is invalid / missing.")
            return torch.from_numpy(arr).float()

        # On-the-fly: soundfile decode + scipy resample
        filename   = self.metadata.iloc[idx]['path']
        audio_path = self._reconstruct_path(filename)
        return _load_audio_sf(str(audio_path), self.target_sr)

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.use_spectrogram:
            return waveform                                    # (T,)

        # MelSpectrogram expects (channel, T) → add + remove channel dim
        mel  = self.mel_transform(waveform.unsqueeze(0))      # (1, n_mels, T')
        logm = torch.log(mel + 1e-9)
        return logm.squeeze(0)                                 # (n_mels, T')


    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        try:
            waveform = self._load_audio(idx)
            features = self._extract_features(waveform)

            result = {
                'audio':        features,
                'audio_length': features.shape[-1],
                'filename':     self.metadata.iloc[idx]['path'],
            }
            if not self.for_ssl:
                result['transcript'] = self.metadata.iloc[idx]['transcript']
            return result

        except Exception as exc:
            print(f"Error loading sample {idx}: {exc}")
            dummy_len = 1000
            dummy = (
                torch.zeros(self.n_mels, dummy_len)
                if self.use_spectrogram
                else torch.zeros(dummy_len)
            )
            return {
                'audio':        dummy,
                'audio_length': dummy_len,
                'transcript':   '' if not self.for_ssl else None,
                'filename':     'dummy',
            }

def collate_fn_ssl(batch: List[Dict]) -> Dict:
    batch   = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
    audios  = [item['audio']        for item in batch]
    lengths = torch.tensor([item['audio_length'] for item in batch])
    names   = [item['filename']     for item in batch]

    if audios[0].dim() == 1:
        # raw waveform path → simple pad_sequence
        padded = torch.nn.utils.rnn.pad_sequence(
            audios, batch_first=True, padding_value=0.0
        )                                                       # (B, T_max)
    else:
        # spectrogram path → manual zero-pad along time axis
        n_mels = audios[0].shape[0]
        T_max  = max(a.shape[-1] for a in audios)
        padded = torch.zeros(len(audios), n_mels, T_max)
        for i, a in enumerate(audios):
            padded[i, :, :a.shape[-1]] = a                     # (B, n_mels, T_max)

    return {'waveform': padded, 'lengths': lengths, 'filenames': names}


def collate_fn_asr(batch: List[Dict], tokenizer) -> Dict:
    batch         = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
    audios        = [item['audio']       for item in batch]
    audio_lengths = torch.tensor([item['audio_length'] for item in batch])
    transcripts   = [item['transcript']  for item in batch]
    names         = [item['filename']    for item in batch]

    if audios[0].dim() == 1:
        padded_audio = torch.nn.utils.rnn.pad_sequence(
            audios, batch_first=True, padding_value=0.0
        )
    else:
        n_mels = audios[0].shape[0]
        T_max  = max(a.shape[-1] for a in audios)
        padded_audio = torch.zeros(len(audios), n_mels, T_max)
        for i, a in enumerate(audios):
            padded_audio[i, :, :a.shape[-1]] = a

    encoded        = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded])
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        encoded, batch_first=True, padding_value=tokenizer.pad_id
    )

    return {
        'waveforms':      padded_audio,
        'audio_lengths':  audio_lengths,
        'targets':        padded_targets,
        'target_lengths': target_lengths,
        'filenames':      names,
        'transcripts':    transcripts,
    }
