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

class ABXDataset(Dataset):
    def __init__(self, metadata_path, segment_len=32000): # e.g., 2 seconds at 16k
        self.df = pd.read_csv(metadata_path, sep="\t")
        self.segment_len = segment_len

    def __len__(self):
        return len(self.df)

    def _get_segment(self, path):
        waveform, _ = sf.read(path, always_2d=True)
        waveform = torch.tensor(waveform.T, dtype=torch.float32).mean(dim=0)
        
        # Ensure it's long enough, else pad
        if waveform.shape[0] < self.segment_len:
            waveform = F.pad(waveform, (0, self.segment_len - waveform.shape[0]))
        
        # Random crop
        start = torch.randint(0, waveform.shape[0] - self.segment_len + 1, (1,)).item()
        return waveform[start : start + self.segment_len]

    def __getitem__(self, idx):
        path_a = self.df.iloc[idx]['path']
        
        # A and X are two different crops/augments of the same file
        anchor = self._get_segment(path_a)
        positive = self._get_segment(path_a) 
        
        # B is a random different file
        random_idx = torch.randint(0, len(self.df), (1,)).item()
        while random_idx == idx:
            random_idx = torch.randint(0, len(self.df), (1,)).item()
        
        negative = self._get_segment(self.df.iloc[random_idx]['path'])
        
        return anchor, positive, negative

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

    # targets = rnn_utils.pad_sequence(targets, batch_first=True, padding_value=0)
    target_len = torch.tensor([len(target) for target in targets], dtype=torch.long)

    DOWNSAMPLING_FACTOR = 320 
    input_len = torch.div(raw_waveform_len, DOWNSAMPLING_FACTOR, rounding_mode='floor')
    
    input_len[input_len == 0] = 1

    return waveforms, targets, input_len, target_len

def load_text(text_path):
    all_text = ""
    for file in tqdm(glob.glob(text_path + "/**/*.txt", recursive=True)):
        with open(file, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    return all_text

@torch.no_grad()
def run_abx_val(model, abx_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for a, p, n in tqdm(abx_loader):
        a, p, n = a.to(device).unsqueeze(1), p.to(device).unsqueeze(1), n.to(device).unsqueeze(1)
        
        feat_a = model.extract_features(a).mean(dim=1)
        feat_p = model.extract_features(p).mean(dim=1)
        feat_n = model.extract_features(n).mean(dim=1)
        
        sim_pos = F.cosine_similarity(feat_a, feat_p)
        sim_neg = F.cosine_similarity(feat_a, feat_n)
        
        correct += (sim_pos > sim_neg).sum().item()
        total += a.size(0)
    
    accuracy = correct / total
    print(f"ABX Discrimination Accuracy: {accuracy:.2%}")
    return accuracy