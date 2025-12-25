from src.requirements import *
from src.ssl_model import *
from src.asr_model import *
from src.tokenizer import *

class InferenceModel(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, tokenClass, device="cpu"):
        super().__init__()
        self.device = device

        self.tokenizer = tokenClass.load(tokenizer)
        vocab_size = len(self.tokenizer.vocab)
        
        dummy_ssl = SSLModel()
        self.asr_model = ASRModel(dummy_ssl, vocab_size-1, True)

        asr_checkpoint = torch.load(decoder)
        self.asr_model.load_state_dict(asr_checkpoint['model_state_dict'])
        self.asr_model.eval()

    def greedy_decode(self, log_probs):
        prediction_ids = torch.argmax(log_probs, dim=-1)
        results = []

        prev = None
        blank_id = 0

        for p in prediction_ids:
            p = p.item()
            
            if p != blank_id and p!= prev:
                results.append(p)
                
            prev = p

        return results

    def forward(self, waveform, sr):
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 2:
            waveform = waveform.T
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        wave_np = waveform.squeeze(0).numpy()
        trimmed, _ = librosa.effects.trim(wave_np, top_db=TOP_DB)
        waveform = torch.tensor(trimmed, dtype=torch.float32).unsqueeze(0)
            
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)

        waveform = waveform.unsqueeze(0)
                
        with torch.no_grad():
            logits = self.asr_model(waveform)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs[0]

        ids = self.greedy_decode(log_probs)
        text = self.tokenizer.decode(ids)

        return text
