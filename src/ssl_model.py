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
        output, _ = self.gru(z)
        return output

class ContrastivePredictor(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.project(x)

class SSLModel(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=128, m=0.999):
        super().__init__()

        # online networks
        self.encoder = FeatureEncoder()
        self.context = ContextModule(feat_dim, feat_dim)
        self.predictor = ContrastivePredictor(feat_dim, proj_dim)
        
        self.target_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )

        # momentum networks (exponential moving average)
        self.momentum = m
        self.encoder_m = FeatureEncoder()
        self.context_m = ContextModule(feat_dim, feat_dim)
        self.target_proj_m = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )

        self._init_momentum_encoder()

    def _init_momentum_encoder(self):
        for online, target in [(self.encoder, self.encoder_m),
                               (self.context, self.context_m),
                               (self.target_proj, self.target_proj_m)]:
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.copy_(p_o.data)
                p_t.requires_grad = False

    @torch.no_grad()
    def update_momentum(self):
        m = self.momentum
        for online, target in [(self.encoder, self.encoder_m),
                               (self.context, self.context_m),
                               (self.target_proj, self.target_proj_m)]:
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data = p_t.data * m + p_o.data * (1.0 - m)

    def extract_features(self, x):
        z = self.encoder(x)
        z = z.transpose(1, 2)
        return self.context(z)

    def forward(self, x, mask=None, mask_prob=0.5, mask_length=10):
        B = x.size(0)

        with torch.no_grad():
            z_t = self.encoder_m(x)
            z_t = z_t.transpose(1, 2)
            # z_t = self._encode(x, self.encoder_m, self.context_m)
            t_proj = self.target_proj_m(z_t)

        B, T, _ = t_proj.shape
        mask = compute_mask_indices(B, T, mask_prob, mask_length, x.device)

        z_online = self.encoder(x)
        z_online = z_online.transpose(1, 2)

        z_masked = z_online.clone()
        z_masked[mask.unsqueeze(-1).expand_as(z_masked)] = 0
        # z_masked[mask] = 0

        c = self.context(z_masked)
        q = self.predictor(c)

        z_proj_online = self.target_proj(z_online)
        t_proj = t_proj.detach()

        return z_proj_online, q, t_proj, mask

    # def forward(self, x, mask=None, mask_prob=0.5, mask_length=10):

    #     z = self.encoder(x)
    #     z = z.transpose(1, 2)
    #     B, T, D = z.shape
        
    #     z_masked = z.clone()
        
    #     if mask is None:
    #         mask = compute_mask_indices(B, T, mask_prob, mask_length, device=z.device)
            
    #     # z_masked[mask.unsqueeze(-1).expand_as(z_masked)] = 0
    #     z_masked[mask] = 0

    #     c = self.context(z_masked)
    #     q = self.predictor(c)
    #     q = F.normalize(q, dim=-1)
        
    #     z_proj = self.target_proj(z).detach()
    #     z_proj = F.normalize(z_proj, dim=-1)

    #     print(f"z requires_grad: {z_proj.requires_grad}")
    #     print(f"q requires_grad: {q.requires_grad}")

    #     return z_proj, q, mask

def compute_mask_indices(B, T, mask_prob=0.5, mask_length=10, device="cpu"):
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    num_masked_steps = int(mask_prob * T)
    num_spans = max(1, num_masked_steps // mask_length)

    for b in range(B):
        possible_starts = torch.arange(T - mask_length, device=device)
        perm = torch.randperm(len(possible_starts), device=device)
        span_starts = possible_starts[perm[:num_spans]]

        for s in span_starts:
            mask[b, s: s + mask_length] = True

    return mask