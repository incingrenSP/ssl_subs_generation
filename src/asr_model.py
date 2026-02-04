from src.requirements import *
from src.tokenizer import *
from src.ssl_model import *

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
        

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        
        # SSL components (frozen)
        self.encoder = ssl_model.encoder
        self.context = ssl_model.context
        
        # Multi-scale feature extraction
        self.multiscale = MultiScaleFeatureExtractor(128, 128)
        
        # Diversification layer
        self.diversify = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Dropout(dropout)
        )
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bi-directional output back to single dimension
        self.projection = nn.Linear(hidden_dim * 2, 128)
        
        # Output layer
        self.symbol = nn.Linear(128, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.symbol.weight)
        nn.init.zeros_(self.symbol.bias)
    
    def forward(self, x, input_lengths=None):
        """
        Forward pass.
        
        Args:
            x: (batch, channels, samples)
            input_lengths: (batch,) - sequence lengths after downsampling
        
        Returns:
            log_probs: (seq_len, batch, vocab_size) for CTC
        """
        # Extract SSL features (frozen)
        with torch.no_grad():
            z = self.encoder(x).transpose(1, 2)  # (batch, time, features)
            z = self.context(z)
        
        # Multi-scale feature extraction
        z = self.multiscale(z)
        z = z + self.diversify(z)  # Residual connection
        
        # Pack sequences if lengths provided (for efficiency)
        if input_lengths is not None:
            z = nn.utils.rnn.pack_padded_sequence(
                z, 
                input_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
        
        # Bi-LSTM
        z, _ = self.lstm(z)
        
        # Unpack sequences
        if input_lengths is not None:
            z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True)
        
        # Project back to 128 dimensions
        z = self.projection(z)
        
        # Output layer
        logits = self.symbol(z)
        log_probs = logits.transpose(0, 1).log_softmax(dim=2)
        
        return log_probs
    
    def freeze_ssl(self):
        """Freeze SSL encoder and context."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.context.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.context.eval()
    
    def unfreeze_ssl(self):
        """Unfreeze SSL components."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.context.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}