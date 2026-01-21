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
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            curr_input = input_dim if i == 0 else hidden_dim
            self.layers.append(
                nn.GRU(curr_input, hidden_dim, batch_first=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for gru, norm in zip(self.layers, self.norms):
            residual = x
            x, _ = gru(x)
            x = norm(x)
            x = self.dropout(x)
            
            if residual.shape == x.shape:
                x = x + residual
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.project(x)

class SSLModel(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=128, m=0.99):
        super().__init__()

        # online networks
        self.encoder = FeatureEncoder()
        self.context = ContextModule(feat_dim, feat_dim)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.predictor = Predictor(proj_dim, proj_dim)

        # momentum networks (EMA)
        self.momentum = m
        self.encoder_m = copy.deepcopy(self.encoder)
        self.context_m = copy.deepcopy(self.context)
        self.projector_m = copy.deepcopy(self.projector)

        self._init_momentum_encoder()

    def _init_momentum_encoder(self):
        for p in self.encoder_m.parameters():
            p.requires_grad = False
        for p in self.context_m.parameters():
            p.requires_grad = False
        for p in self.projector_m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_momentum(self):
        m = self.momentum
        for online, target in [
            (self.encoder, self.encoder_m),
            (self.context, self.context_m),
            (self.projector, self.projector_m),
        ]:
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def extract_features(self, x):
        z = self.encoder(x).transpose(1, 2)
        return self.context(z)

    def forward(self, x, mask_prob=0.065, mask_length=10):
        z1 = self.encoder(x).transpose(1, 2)
        B, T, C = z1.shape

        mask1 = compute_mask_indices(B, T, mask_prob, mask_length, z1.device)
        z1_masked = apply_mask(z1, mask1) 
        
        c1 = self.context(z1_masked)
        c1_sub = c1[:, ::2]
        p1 = self.predictor(self.projector(c1_sub))
        
        with torch.no_grad():
            z2 = self.encoder_m(x).transpose(1, 2)
            c2 = self.context_m(z2)
            h2 = self.projector_m(c2[:, ::2])
        
        mask_sub = mask1[:, ::2]
        
        p1_masked = p1[mask_sub]
        h2_masked = h2[mask_sub]
        
        loss = byol_loss(p1_masked, h2_masked) + 1.0 * variance_loss(c1_sub)
        return loss

def byol_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


def compute_mask_indices(B, T, mask_prob=0.05, mask_length=10, device="cpu"):
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

def variance_loss(z, eps=1e-4):
    z = z.reshape(-1, z.size(-1))
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(1.0 - std))


def apply_mask(z, mask):
    z = z.clone()
    z[mask] = 0.0
    return z

def masked_mean(c, mask):
    valid = (~mask).unsqueeze(-1).float()
    return (c * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
