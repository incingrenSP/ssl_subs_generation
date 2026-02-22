from src.requirements import *
from src.tokenizer import *
from src.ssl_large import *


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Different temporal scales
        self.local = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.medium = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        
        # Global context
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
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # Transpose for conv1d: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Multi-scale features
        local_feat = self.local(x_t).transpose(1, 2)  # (batch, seq_len, output_dim//4)
        medium_feat = self.medium(x_t).transpose(1, 2)  # (batch, seq_len, output_dim//4)
        
        # Global context
        global_feat = self.global_pool(x_t).squeeze(-1)  # (batch, input_dim)
        global_feat = self.global_fc(global_feat)  # (batch, output_dim//4)
        global_feat = global_feat.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, output_dim//4)
        
        # Concatenate all scales
        combined = torch.cat([local_feat, medium_feat, global_feat], dim=-1)
        
        # Mix and project
        output = self.combine(combined)
        
        # Residual connection
        output = output + self.residual_proj(x)
        
        return output


class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Get SSL feature dimension (flexible for different model sizes)
        self.ssl_feat_dim = self._get_ssl_feat_dim(ssl_model)
        
        # SSL components (will be frozen)
        self.encoder = ssl_model.encoder
        self.context = ssl_model.context
        
        # Multi-scale feature extraction (adapts to SSL dimension)
        self.multiscale = MultiScaleFeatureExtractor(
            input_dim=self.ssl_feat_dim,
            output_dim=self.ssl_feat_dim
        )
        
        # Diversification layer
        self.diversify = nn.Sequential(
            nn.Linear(self.ssl_feat_dim, self.ssl_feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ssl_feat_dim * 2, self.ssl_feat_dim),
            nn.Dropout(dropout)
        )
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.ssl_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bi-directional output
        self.projection = nn.Linear(hidden_dim * 2, self.ssl_feat_dim)
        
        # Output layer
        self.symbol = nn.Linear(self.ssl_feat_dim, vocab_size)
        
        self._init_weights()
    
    def _get_ssl_feat_dim(self, ssl_model):
        # Try to get from context module
        if hasattr(ssl_model.context, 'input_proj'):
            return ssl_model.context.input_proj.out_features
        elif hasattr(ssl_model.context, 'norm'):
            return ssl_model.context.norm.normalized_shape[0]
        else:
            # Fallback: assume 256
            return 256
    
    def _init_weights(self):
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
        # Extract SSL features (frozen)
        with torch.no_grad():
            z = self.encoder(x).transpose(1, 2)  # (batch, time, ssl_feat_dim)
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
        
        # Project back to SSL dimension
        z = self.projection(z)
        
        # Output layer
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
        print("✓ SSL components frozen")
    
    def unfreeze_ssl(self, lr_scale=0.1):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.context.parameters():
            param.requires_grad = True
        self.encoder.train()
        self.context.train()
        print(f"✓ SSL components unfrozen (recommended LR scale: {lr_scale}x)")
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        # Breakdown by component
        ssl_params = sum(p.numel() for p in self.encoder.parameters()) + \
                    sum(p.numel() for p in self.context.parameters())
        asr_params = total - ssl_params
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'ssl': ssl_params,
            'asr': asr_params
        }
    
    def print_model_info(self):
        params = self.get_num_params()
        print(f"\n{'='*60}")
        print("ASR Model Information")
        print(f"{'='*60}")
        print(f"SSL feature dimension: {self.ssl_feat_dim}")
        print(f"Vocabulary size: {self.symbol.out_features}")
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Frozen: {params['frozen']:,}")
        print(f"  SSL: {params['ssl']:,}")
        print(f"  ASR head: {params['asr']:,}")
        print(f"{'='*60}\n")


def create_asr_model(ssl_checkpoint_path, tokenizer, device='cuda', 
                     hidden_dim=256, num_layers=4, dropout=0.1):
    
    print(f"Loading SSL model from {ssl_checkpoint_path}...")
    
    # Create SSL model (adjust size based on your training)
    ssl_model = LargeSSLModel(
        feat_dim=256,
        proj_dim=256,
        m=0.996
    )
    
    # Load checkpoint
    checkpoint = torch.load(ssl_checkpoint_path, map_location=device)
    ssl_model.load_state_dict(checkpoint['model_state_dict'])
    ssl_model.to(device)
    ssl_model.eval()
    
    num_updates = checkpoint.get('num_updates', 'unknown')
    print(f"✓ SSL model loaded (trained for {num_updates} updates)")
    
    # Create ASR model
    vocab_size = len(tokenizer)
    asr_model = ASRModel(
        ssl_model=ssl_model,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Freeze SSL by default
    asr_model.freeze_ssl()
    asr_model.to(device)
    
    # Print info
    asr_model.print_model_info()
    
    return asr_model