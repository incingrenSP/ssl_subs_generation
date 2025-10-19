from src.requirements import *
        
class AudioDataset(Dataset):
    def __init__(self, audio_path, target_sr=16000, transform=None):
        self.audio_path = audio_path
        self.target_sr = target_sr
        self.transform = transform
        self.file_list = []
        self._gather_files()

    def _gather_files(self):
        for dir1 in os.listdir(self.audio_path):
            if dir1.endswith('.tsv'):
                continue
            subdir = os.path.join(self.audio_path, dir1)
            for dir2 in tqdm(os.listdir(subdir)):
                if dir2.endswith('.tsv'):
                    continue
                path = os.path.join(subdir, dir2)
                self.file_list.append(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        waveform, sr = sf.read(path, always_2d=True)
        waveform = torch.Tensor(waveform.T)

        if sr != self.target_sr:
            resampler = T.Resampler(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform / (waveform.abs().max() + 1e-8)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0)

def collate_padding(batch):
    batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    batch = batch.unsqueeze(1)
    return batch