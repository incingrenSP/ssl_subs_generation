from src.requirements import *

def compute_mask_indices(B, T, mask_prob, mask_length, device="cpu"):

    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    num_masked_spans = int((T * mask_prob) / mask_length)
    for b in range(B):
        starts = torch.randint(0, T - mask_length, (num_masked_spans,))
        for s in starts:
            mask[b, s : s + mask_length] = True
    return mask

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)

class AutoregressiveContext(nn.Module):
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

class SSLAutoregressiveModel(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.context = AutoregressiveContext(feat_dim, feat_dim)
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