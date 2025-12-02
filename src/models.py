from src.requirements import *

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=5, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()            
        )
        
    def forward(self, x):
        return self.encoder(x)

class ContextModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, z):
        # natural autoregressive context representation
        output, _ = self.gru(z)
        return output

class ContrastivePredictor(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.project = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        return self.project(x)

class SSLModel(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.context = ContextModule(feat_dim, feat_dim)
        self.predictor = ContrastivePredictor(feat_dim, proj_dim)
        self.target_proj = nn.Linear(feat_dim, proj_dim)

    def forward(self, x, mask=None, mask_prob=0.065, mask_length=10):
        # z -> true latent features
        z = self.encoder(x)
        z = z.transpose(1, 2)
        B, T, F = z.shape

        # masking
        z_masked = z.clone()
        
        if mask is None:
            mask = compute_mask_indices(B, T, mask_prob, mask_length, device=z.device)
        
        # z_masked[torch.arange(z.size(0)).unsqueeze(1), mask_indices] = 0
        z_masked[mask.unsqueeze(-1).expand_as(z_masked)] = 0

        # c -> context
        c = self.context(z_masked)
        
        # q -> predicted queries
        q = self.predictor(c)

        # projection
        z_proj = self.target_proj(z)

        return z_proj, q, mask

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, freeze_ssl=True):
        super().__init__()
        self.model = ssl_model
        
        if freeze_ssl:
             for params in self.model.encoder.parameters():
                 params.requires_grad = False

        self.fc = nn.Linear(128, vocab_size+1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        z = self.model.encoder(x)
        z = z.transpose(1, 2)

        c = self.model.context(z)

        logits = self.fc(c)
        log_probs = self.log_softmax(logits)

        return log_probs

class InferenceModel(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, tokenClass, device="cpu"):
        super().__init__()
        self.device = device

        self.tokenizer = tokenClass.load(tokenizer)
        vocab_size = len(self.tokenizer.vocab)
        
        dummy_ssl = SSLModel()
        self.asr_model = ASRModel(dummy_ssl, vocab_size).to(device)

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
        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)

        with torch.no_grad():
            log_probs = self.decoder(waveform)
            log_probs = log_probs[0]

        ids = self.greedy_decode(log_probs)
        text = self.tokenizer.decode(ids)

        return text

def compute_mask_indices(B, T, mask_prob, mask_length, device="cpu"):
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    num_masked_spans = int((T * mask_prob) / mask_length)
    
    for b in range(B):
        starts = torch.randint(0, T - mask_length, (num_masked_spans,))
        
        for s in starts:
            mask[b, s : s + mask_length] = True
            
    return mask

def flatten_targets(targets, target_lengths):
    flattened = []
    
    for i in range(targets.size(0)):
        seq = targets[i, :target_lengths[i]]
        flattened.append(seq)
        
    return torch.cat(flattened)