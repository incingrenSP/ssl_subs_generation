from src.requirements import *
        
class AudioDataset(Dataset):
    def __init__(self, metadata_path):
        super().__init__()
        self.df = pd.read_csv(metadata_path, sep="\t")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        waveform, sr = sf.read(path, always_2d=True)
        waveform = torch.tensor(waveform.T, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        wave_np = waveform.squeeze(0).numpy()
        trimmed, _ = librosa.effects.trim(wave_np, top_db=TOP_DB)
        waveform = torch.tensor(trimmed, dtype=torch.float32).unsqueeze(0)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return waveform.squeeze(0)

class ASRDataset(Dataset):
    def __init__(self, metadata_path, tokenizer):
        super().__init__()
        self.df = pd.read_csv(metadata_path, sep="\t")
        self.tokenizer = tokenizer
        self.audio_paths = self.df['path'].tolist()
        self.transcripts = self.df['transcript'].tolist()

        self.encode_transcripts = []
        for t in self.transcripts:
            encoded = tokenizer.encode(t)
            encoded = [i if i >= 0 else 0 for i in encoded]
            self.encode_transcripts.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(row['path'], always_2d=True)
        waveform = torch.tensor(waveform.T, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        wave_np = waveform.squeeze(0).numpy()
        trimmed, _ = librosa.effects.trim(wave_np, top_db=TOP_DB)
        waveform = torch.tensor(trimmed, dtype=torch.float32).unsqueeze(0)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 1:
            waveform = waveform / max_val
        
        target = self.encode_transcripts[idx]
        
        return waveform.squeeze(0), target

def collate_padding(batch):
    padded_batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    padded_batch = padded_batch.unsqueeze(1)
    return padded_batch

def collate_padding_asr(batch):
    waveforms, targets = zip(*batch)
    raw_waveform_len = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    
    waveforms = rnn_utils.pad_sequence(waveforms, batch_first=True, padding_value=0)
    waveforms = waveforms.unsqueeze(1)
    
    target_len = torch.tensor([len(target) for target in targets], dtype=torch.long)
    
    DOWNSAMPLING_FACTOR = 320 
    input_len = torch.div(raw_waveform_len, DOWNSAMPLING_FACTOR, rounding_mode='floor')
    input_len[input_len == 0] = 1
    
    return waveforms, targets, input_len, target_len

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