from src.requirements import *

class DeepFeatureEncoder(nn.Module):    
    def __init__(self, in_channels=1, hidden_dims=[256, 384, 512, 512]):
        super().__init__()
        
        layers = []
        in_dim = in_channels
        
        # First layer: Large receptive field
        layers.append(nn.Conv1d(in_dim, hidden_dims[0], kernel_size=10, stride=5, padding=3))
        layers.append(nn.GroupNorm(num_groups=16, num_channels=hidden_dims[0]))
        layers.append(nn.GELU())
        
        # Layer 2
        layers.append(nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=4, padding=1))
        layers.append(nn.GroupNorm(num_groups=24, num_channels=hidden_dims[1]))
        layers.append(nn.GELU())
        
        # Layer 3
        layers.append(nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=4, padding=1))
        layers.append(nn.GroupNorm(num_groups=32, num_channels=hidden_dims[2]))
        layers.append(nn.GELU())
        
        # Layer 4
        layers.append(nn.Conv1d(hidden_dims[2], hidden_dims[3], kernel_size=3, stride=2, padding=1))
        layers.append(nn.GroupNorm(num_groups=32, num_channels=hidden_dims[3]))
        layers.append(nn.GELU())
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class DeepContextModule(nn.Module):    
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,  # 2048
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x


class PositionalEncoding(nn.Module):    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LargeSSLModel(nn.Module):
    def __init__(self, feat_dim=512, proj_dim=512, m=0.996):
        super().__init__()
        
        # Online networks
        self.encoder = DeepFeatureEncoder(
            in_channels=1,
            hidden_dims=[256, 384, 512, 512]
        )
        
        self.context = DeepContextModule(
            input_dim=512,
            hidden_dim=512,
            num_layers=4,
            nhead=8,
            dropout=0.1
        )
        
        # Projector (larger)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        
        # Momentum networks
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
        # Encode
        z1 = self.encoder(x).transpose(1, 2)
        B, T, C = z1.shape
        
        # Masking
        mask1 = compute_mask_indices(B, T, mask_prob, mask_length, z1.device)
        z1_masked = apply_mask(z1, mask1)
        
        # Online path
        c1 = self.context(z1_masked)
        c1_sub = c1[:, ::2]
        p1 = self.predictor(self.projector(c1_sub))
        
        # Momentum path
        with torch.no_grad():
            z2 = self.encoder_m(x).transpose(1, 2)
            c2 = self.context_m(z2)
            h2 = self.projector_m(c2[:, ::2])
        
        # Loss
        mask_sub = mask1[:, ::2]
        p1_masked = p1[mask_sub]
        h2_masked = h2[mask_sub]
        
        # BYOL loss (no variance loss - model is large enough)
        loss = byol_loss(p1_masked, h2_masked)
        
        return loss


# Helper functions
def byol_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


def compute_mask_indices(B, T, mask_prob=0.065, mask_length=10, device="cpu"):
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


def apply_mask(z, mask):
    z = z.clone()
    z[mask] = 0.0
    return z