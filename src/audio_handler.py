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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(row['path'], always_2d=True)
            
        waveform = torch.tensor(waveform.T, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform / torch.max(torch.abs(waveform))
            
        target = self.tokenizer.encode(row['transcript'])
        
        return waveform.squeeze(0), torch.tensor(target, dtype=torch.long)

class Tokenizer:
    def __init__(self, corpus_path, add_blank=True):        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [unicodedata.normalize('NFC', l.strip()) for l in f]

        tokens = []
        for line in lines:
            tokens.extend(self.tokenize(line))

        counter = Counter(tokens)
        self.vocab = sorted(counter.keys())

        if add_blank:
            self.vocab = ['<blank>'] + self.vocab

        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def tokenize(self, text):
        text = unicodedata.normalize('NFC', text)
        return regex.findall(r'\X', text)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids):
        return ''.join([self.id_to_token[i] for i in ids if i in self.id_to_token])

def collate_padding(batch):
    batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    batch = batch.unsqueeze(1)
    return batch

def collate_padding_asr(batch):
    waveforms, targets = zip(*batch)
    
    waveforms = rnn_utils.pad_sequence(waveforms, batch_first=True, padding_value=0)    
    waveforms = waveforms.unsqueeze(1)
    
    target_len = torch.tensor([target.size(0) for target in targets], dtype=torch.long)

    # padding value -1 to not get nan cuz ctc is a bitch
    targets = rnn_utils.pad_sequence(targets, batch_first=True, padding_value=-1)
    
    return waveforms, targets, target_len

def load_text(text_path):
    all_text = ""
    for file in tqdm(glob.glob(text_path + "/**/*.txt", recursive=True)):
        with open(file, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    return all_text