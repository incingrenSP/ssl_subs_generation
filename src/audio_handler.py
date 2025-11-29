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

        waveform = waveform / torch.max(torch.abs(waveform))

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

        waveform = waveform / torch.max(torch.abs(waveform))
        target = self.encode_transcripts[idx]
        
        return waveform.squeeze(0), target

def collate_padding(batch):
    batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    batch = batch.unsqueeze(1)
    return batch

def collate_padding_asr(batch):
    waveforms, targets = zip(*batch)
    
    waveforms = rnn_utils.pad_sequence(waveforms, batch_first=True, padding_value=0)    
    waveforms = waveforms.unsqueeze(1)
    targets = rnn_utils.pad_sequence(targets, batch_first=True, padding_value=0)

    input_len = torch.tensor([wave.shape[-1] for wave in waveforms], dtype=torch.long)
    target_len = torch.tensor([len(target) for target in targets], dtype=torch.long)

    return waveforms, targets, input_len, target_len

def load_text(text_path):
    all_text = ""
    for file in tqdm(glob.glob(text_path + "/**/*.txt", recursive=True)):
        with open(file, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    return all_text

def flatten_targets(targets, target_lengths):
    flattened = []
    for i in range(targets.size(0)):
        seq = targets[i, :target_lengths[i]]
        flattened.append(seq)
    return torch.cat(flattened)