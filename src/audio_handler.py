from src.requirements import *
        
class AudioDataset(Dataset):
    def __init__(self, audio_path, device='cpu', target_sr=16000):
        self.audio_path = audio_path
        self.device = device
        self.target_sr = target_sr
        self.data = []
        self.load_data()
        
    def load_data(self):
        for dir1 in os.listdir(self.audio_path):
            for dir2 in tqdm(os.listdir(os.path.join(self.audio_path, dir1))):
                if os.path.splitext(dir2)[1] == '.tsv':
                    continue
                waveform, sr = torchaudio.load(os.path.join(self.audio_path, dir1, dir2))
                waveform, sr = self.audio_preprocess(waveform, sr, self.target_sr)
                self.data.append(waveform.squeeze(0))

    def audio_preprocess(self, waveform, sr, target_sr):
        # resample
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
        # stereo -> mono
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform_mono
    
        # normalize
        waveform = waveform / waveform.abs().max()
    
        return waveform, target_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_padding(batch):
    batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    batch = batch.unsqueeze(1)
    return batch