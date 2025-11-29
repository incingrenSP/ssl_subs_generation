from src.requirements import *

class InferenceModel(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, encoderClass, decoderClass, tokenClass, device="cpu"):
        self.encoder = encoderClass()
        self.encoder.load_state_dict(torch.load(encoder))
        self.encoder.eval()
        
        self.decoder = decoderClass()
        self.decoder.load_state_dict(torch.load(decoder))
        self.decoder.eval()
        
        self.tokenizer = tokenClass.load(tokenizer)
        vocab_size = len(self.tokenizer.vocab)

    def greedy_decode(self, log_probs):
        prediction_ids = torch.argmax(log_probs, dim=-1)
        results = []

        prev = None
        blank_id = 0

        for p in prediction_ids:
            p = p.items()
            if p != blank_id and p!= prev:
                results.append(p)
            prev = p

        return results

    def transcribe(self, waveform, sr):
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)

        waveform = waveform.to(device)

        with torch.no_grad():
            log_probs = self.decoder(waveform.squeeze(0))
            log_probs = log_probs[0]

        ids = self.greedy_decode(lob_probs)
        text = self.tokenizer.decode(ids)

        return text