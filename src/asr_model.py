import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class WhiteningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.01):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_cov", torch.eye(dim))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (B, T, dim)
        if self.training:
            B, T, D = x.shape
            x_flat = x.reshape(-1, D)
            mean = x_flat.mean(0)
            centered = x_flat - mean
            cov = (centered.T @ centered) / (x_flat.shape[0] - 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_cov  = (1 - self.momentum) * self.running_cov  + self.momentum * cov
        else:
            mean = self.running_mean
            cov  = self.running_cov

        # Whiten
        U, S, _ = torch.linalg.svd(cov + self.eps * torch.eye(cov.shape[0], device=cov.device))
        W = U @ torch.diag(1.0 / S.sqrt()) @ U.T
        x_centered = x - mean
        x_white = x_centered @ W.T
        return x_white * self.weight + self.bias

class NepaliASR(nn.Module):
    def __init__(
        self,
        ssl_model,
        vocab_size: int,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Pull encoder components from SSL model
        self.encoder  = ssl_model.cnn      # CNNEncoder
        self.context  = ssl_model.context  # TransformerEncoder
        hidden_dim    = ssl_model.cfg.hidden_dim  # 256
        
        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        # self.pre_head_norm = WhiteningNorm(hidden_dim)

        # CTC projection head
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.ctc_head.weight, mean=0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)

        self.vocab_size = vocab_size

        if freeze_encoder:
            self.freeze_encoder()

    # Freeze / unfreeze helpers
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.context.parameters():
            p.requires_grad = False

    def unfreeze_transformer(self):
        for p in self.context.parameters():
            p.requires_grad = True

    def unfreeze_cnn(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        self.unfreeze_transformer()
        self.unfreeze_cnn()

    # Forward
    def forward(
        self,
        waveform: torch.Tensor,         # (B, T_samples)
        lengths:  Optional[torch.Tensor] = None,  # (B,) sample lengths
        spec_aug = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T_wav = waveform.shape
        device = waveform.device

        if lengths is None:
            lengths = torch.full((B,), T_wav, device=device, dtype=torch.long)

        # CNN feature
        features = self.encoder(waveform)               # (B, T, 256)
        T = features.size(1)
        
        # Check Spectral Augmentation
        if spec_aug is not None:
            features = spec_aug(features)

        # Compute valid frame lengths from sample lengths
        frame_lengths = (
            lengths.float() / T_wav * T
        ).long().clamp(max=T)                           # (B,)

        # Padding mask for transformer (True = padding)
        pad_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= frame_lengths.unsqueeze(1)
        )                                               # (B, T)

        # Transformer context
        context = self.context(features, padding_mask=pad_mask)  # (B, T, 256)
        context = self.pre_head_norm(context)

        # CTC head
        logits    = self.ctc_head(context)              # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)       # (B, T, vocab_size)

        return log_probs, frame_lengths

    # Representation geometry

    @torch.no_grad()
    def get_representations(
        self,
        waveform: torch.Tensor,
        lengths:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_wav = waveform.shape
        device = waveform.device

        if lengths is None:
            lengths = torch.full((B,), T_wav, device=device, dtype=torch.long)

        features = self.encoder(waveform)
        T = features.size(1)
        frame_lengths = (lengths.float() / T_wav * T).long().clamp(max=T)
        pad_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            >= frame_lengths.unsqueeze(1)
        )
        return self.context(features, padding_mask=pad_mask)


def load_asr_model(
    ssl_checkpoint_path: str,
    vocab_size: int,
    device: str = "cuda",
    freeze_encoder: bool = True,
) -> NepaliASR:
    from src.ssl_model import NepaliSSL, NepaliSSLConfig

    checkpoint = torch.load(ssl_checkpoint_path, map_location=device)

    cfg_data = checkpoint.get("config", None)
    if isinstance(cfg_data, dict):
        cfg = NepaliSSLConfig(**cfg_data)
    elif cfg_data is None:
        cfg = NepaliSSLConfig()
    else:
        cfg = cfg_data
    ssl_model = NepaliSSL(cfg)
    ssl_model.load_state_dict(checkpoint["model"])
    ssl_model.eval()

    asr_model = NepaliASR(ssl_model, vocab_size=vocab_size, freeze_encoder=freeze_encoder)
    asr_model = asr_model.to(device)

    total     = sum(p.numel() for p in asr_model.parameters())
    trainable = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")

    return asr_model
