from src.requirements import *
from src.tokenizer import *
from src.ssl_model import *

def create_padding_mask(input_lengths, batch_size, max_len, device):
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    lengths_expanded = input_lengths.unsqueeze(1)
    mask = range_tensor >= lengths_expanded
    return mask

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        
        # Different temporal scales
        self.local = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1, groups=1)
        self.medium = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3, groups=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_fc = nn.Linear(input_dim, output_dim // 4)
        
        # Learnable combination
        self.combine = nn.Sequential(
            nn.Linear(output_dim // 4 * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # Transpose for conv1d: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Multi-scale features
        local_feat = self.local(x_t).transpose(1, 2)  # (batch, seq_len, dim//4)
        medium_feat = self.medium(x_t).transpose(1, 2)  # (batch, seq_len, dim//4)
        
        # Global context
        global_feat = self.global_pool(x_t).squeeze(-1)  # (batch, dim)
        global_feat = self.global_fc(global_feat)  # (batch, dim//4)
        global_feat = global_feat.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, dim//4)
        
        # Concatenate
        combined = torch.cat([local_feat, medium_feat, global_feat], dim=-1)
        
        # Mix and project
        output = self.combine(combined)
        
        # Residual connection
        output = output + self.residual_proj(x)
        
        return output


class PositionalEncoding(nn.Module):    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.encoder = ssl_model.encoder
        self.context = ssl_model.context

        self.multiscale = MultiScaleFeatureExtractor(128, 128)

        self.diversify = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(128, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.symbol = nn.Linear(128, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, input_lengths=None):
        with torch.no_grad():
            z = self.encoder(x).transpose(1, 2)
            z = self.context(z)

        z = self.multiscale(z)
        z = z + self.diversify(z)
        z = self.pos_encoder(z)
        
        padding_mask = None
        if input_lengths is not None:
            batch_size, max_len, _ = z.shape
            padding_mask = create_padding_mask(input_lengths, batch_size, max_len, z.device)
        
        z = self.transformer(z, src_key_padding_mask=padding_mask)
        
        logits = self.symbol(z) 
        log_probs = logits.transpose(0, 1).log_softmax(dim=2)
        
        return log_probs
    
    def freeze_ssl(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.context.parameters():
            param.requires_grad = False
        
        self.encoder.eval()
        self.context.eval()
    
    def unfreeze_ssl(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.context.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }