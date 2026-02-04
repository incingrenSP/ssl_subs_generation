from src.requirements import *

class AudioDataset(Dataset):
    def __init__(self, metadata_path, cache_dir='data/cache_mmap', top_db=20):
        super().__init__()
        self.df = pd.read_csv(metadata_path, sep="\t")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_db = top_db
        
        dataset_name = Path(metadata_path).stem
        meta_file = self.cache_dir / f"{dataset_name}_meta.npz"
        audio_file = self.cache_dir / f"{dataset_name}_audio.dat"
        
        if not meta_file.exists():
            print("Preprocessing audio (first time only)...")
            self._preprocess_all(audio_file, meta_file)
        else:
            print(f"Loading metadata from cache...")
        
        meta = np.load(meta_file, allow_pickle=True)
        self.audio_shapes = meta['shapes']
        self.audio_offsets = meta['offsets']
        self.total_size = meta['total_size'].item()
        
        self.audio_mmap = np.memmap(audio_file, dtype='float32', mode='r', shape=(self.total_size,))
        
        print(f"  Cache ready! {len(self.df)} samples")
        print(f"  Total audio size: {self.total_size * 4 / (1024**3):.2f} GB")
    
    def _preprocess_all(self, audio_file, meta_file):
        import time
        start = time.time()
        
        print("Processing audio files...")
        
        all_audio_data = []
        all_shapes = []
        
        for idx in tqdm(range(len(self.df)), desc="Loading & preprocessing"):
            path = self.df.iloc[idx]['path']
            
            waveform, sr = sf.read(path, always_2d=True)
            waveform = np.array(waveform.T, dtype=np.float32)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform[0]
            
            # Trim
            trimmed, _ = librosa.effects.trim(waveform, top_db=self.top_db)
            
            max_val = np.abs(trimmed).max()
            if max_val > 0:
                trimmed = trimmed / max_val
            
            all_audio_data.append(trimmed.astype(np.float32))
            all_shapes.append(len(trimmed))
        
        total_size = sum(all_shapes)
        print(f"Total samples: {total_size:,} ({total_size * 4 / (1024**3):.2f} GB)")
        
        offsets = np.zeros(len(all_shapes) + 1, dtype=np.int64)
        np.cumsum(all_shapes, out=offsets[1:])
        
        print("Writing to memory-mapped file...")
        mmap = np.memmap(audio_file, dtype='float32', mode='w+', shape=(total_size,))
        
        for i in tqdm(range(len(all_audio_data)), desc="Writing"):
            start_pos = offsets[i]
            end_pos = offsets[i + 1]
            mmap[start_pos:end_pos] = all_audio_data[i]
        
        mmap.flush()
        del mmap
        
        np.savez(meta_file,
                 shapes=np.array(all_shapes, dtype=np.int64),
                 offsets=offsets,
                 total_size=np.array(total_size, dtype=np.int64))
        
        print(f"Done in {time.time() - start:.1f}s")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        start = self.audio_offsets[idx]
        end = self.audio_offsets[idx + 1]
        return torch.from_numpy(self.audio_mmap[start:end].copy())


class ASRDataset(Dataset):    
    def __init__(self, metadata_path, tokenizer, cache_dir='data/cache_mmap_asr', top_db=20):
        super().__init__()
        self.df = pd.read_csv(metadata_path, sep="\t")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.top_db = top_db
        
        dataset_name = Path(metadata_path).stem
        meta_file = self.cache_dir / f"{dataset_name}_meta.npz"
        audio_file = self.cache_dir / f"{dataset_name}_audio.dat"
        
        if not meta_file.exists():
            print("Preprocessing audio (first time only)...")
            self._preprocess_all(audio_file, meta_file)
        else:
            print(f"Loading metadata from cache...")
        
        meta = np.load(meta_file, allow_pickle=True)
        self.audio_shapes = meta['shapes']
        self.audio_offsets = meta['offsets']
        self.total_size = meta['total_size'].item()
        
        self.audio_mmap = np.memmap(audio_file, dtype='float32', mode='r', shape=(self.total_size,))
        
        print(f"  Cache ready! {len(self.df)} samples")
        print(f"  Total audio size: {self.total_size * 4 / (1024**3):.2f} GB")
        
        print("Encoding transcripts...")
        self.encode_transcripts = []
        for t in tqdm(self.df['transcript'].tolist(), desc="Encoding"):
            encoded = tokenizer.encode(t)
            encoded = [i if i >= 0 else 0 for i in encoded]
            self.encode_transcripts.append(torch.tensor(encoded, dtype=torch.long))
    
    def _preprocess_all(self, audio_file, meta_file):
        import time
        start = time.time()
        
        print("Processing audio files...")
        
        all_audio_data = []
        all_shapes = []
        
        for idx in tqdm(range(len(self.df)), desc="Loading & preprocessing"):
            path = self.df.iloc[idx]['path']
            
            waveform, sr = sf.read(path, always_2d=True)
            waveform = np.array(waveform.T, dtype=np.float32)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform[0]
            
            trimmed, _ = librosa.effects.trim(waveform, top_db=self.top_db)
            
            max_val = np.abs(trimmed).max()
            if max_val > 1:
                trimmed = trimmed / max_val
            
            all_audio_data.append(trimmed.astype(np.float32))
            all_shapes.append(len(trimmed))
        
        total_size = sum(all_shapes)
        print(f"Total samples: {total_size:,} ({total_size * 4 / (1024**3):.2f} GB)")
        
        offsets = np.zeros(len(all_shapes) + 1, dtype=np.int64)
        np.cumsum(all_shapes, out=offsets[1:])
        
        print("Writing to memory-mapped file...")
        mmap = np.memmap(audio_file, dtype='float32', mode='w+', shape=(total_size,))
        
        for i in tqdm(range(len(all_audio_data)), desc="Writing"):
            start_pos = offsets[i]
            end_pos = offsets[i + 1]
            mmap[start_pos:end_pos] = all_audio_data[i]
        
        mmap.flush()
        del mmap
        
        np.savez(meta_file,
                 shapes=np.array(all_shapes, dtype=np.int64),
                 offsets=offsets,
                 total_size=np.array(total_size, dtype=np.int64))
        
        print(f"Done in {time.time() - start:.1f}s")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        start = self.audio_offsets[idx]
        end = self.audio_offsets[idx + 1]
        waveform = torch.from_numpy(self.audio_mmap[start:end].copy())
        target = self.encode_transcripts[idx]
        return waveform, target


def collate_padding(batch):
    if len(batch) == 0:
        return torch.empty(0, 1, 0)
    
    batch = [torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b for b in batch]
    batch = [b.flatten() for b in batch]
    padded_batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0.0)
    padded_batch = padded_batch.unsqueeze(1)
    
    return padded_batch

def collate_padding_asr(batch):
    waveforms, targets = zip(*batch)
    waveforms = [w.flatten() if isinstance(w, torch.Tensor) else torch.as_tensor(w).flatten() 
                 for w in waveforms]
    
    raw_waveform_len = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    
    waveforms_padded = rnn_utils.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    waveforms_padded = waveforms_padded.unsqueeze(1)  # (batch, 1, max_len)
    
    DOWNSAMPLING_FACTOR = 320
    input_lengths = torch.clamp(
        torch.div(raw_waveform_len, DOWNSAMPLING_FACTOR, rounding_mode='floor'),
        min=1
    )
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    
    return waveforms_padded, targets, input_lengths, target_lengths

    

def is_valid_char(ch):
    code = ord(ch)

    # 1. Devanagari block (U+0900 to U+097F)
    if 0x0900 <= code <= 0x097F:
        return True

    # 2. Standard Latin Digits (0-9) 
    if '0' <= ch <= '9':
        return True

    # 3. Basic Punctuation & Whitespace
    if ch in " \n\t.,?!-()\"'।॥":
        return True

    return False

def normalize_text(text):
    out = []
    for ch in text:
        # 1. Convert any weird numeral to standard '0-9'
        char_to_check = DIGIT_MAP.get(ch, ch)
        
        # 2. Only keep it if it's in our allowed list
        if is_valid_char(char_to_check):
            out.append(char_to_check)
            
    return "".join(out)

def load_text(text_path):
    all_chunks = []
    for file in tqdm(glob.glob(text_path + "/**/*.txt", recursive=True)):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            text = normalize_text(text)
            filtered_text = "".join([ch for ch in text if is_valid_char(ch)])
            all_chunks.append(filtered_text)
    
    return "\n".join(all_chunks)