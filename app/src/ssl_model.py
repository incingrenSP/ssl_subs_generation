import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# Config
@dataclass
class NepaliSSLConfig:
    # CNN encoder
    cnn_channels: list = None          # per-layer output channels
    cnn_kernels: list = None           # kernel sizes
    cnn_strides: list = None           # strides (controls downsampling)

    # Transformer
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_positions: int = 2048          # for relative/fixed PE

    # Quantizer
    use_quantizer: bool = True
    num_codebooks: int = 2
    codebook_size: int = 320
    codebook_dim: int = 128
    
    # Gumbel temperature annealing
    codebook_temp_start:        float = 2.0
    codebook_temp_end:          float = 0.5
    codebook_temp_anneal_steps: int   = 100_000
    
    # Local contrastive loss
    num_negatives: int   = 100   # negatives per masked frame, from same utterance
    logit_temp:    float = 0.1   # temperature on cosine logits

    # Loss weights
    diversity_loss_weight:  float = 1.0    # maximize codebook usage (-perplexity)
    commitment_loss_weight: float = 0.25   # push encoder output toward codebook

    # Masking
    mask_prob:   float = 0.065
    mask_length: int   = 10

    def __post_init__(self):
        if self.cnn_channels is None:
            # 7-layer CNN like wav2vec2-base but slightly smaller
            self.cnn_channels = [512, 512, 512, 512, 512, 512, 512]
        if self.cnn_kernels is None:
            self.cnn_kernels  = [10,   3,   3,   3,   3,   2,   2]
        if self.cnn_strides is None:
            self.cnn_strides  = [5,    2,   2,   2,   2,   2,   2]
        # total stride = 5*2*2*2*2*2*2 = 320 -> 20ms frames at 16kHz


# CNN Feature Encoder
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, first=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)
        self.norm = nn.GroupNorm(1, out_ch) if not first else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNEncoder(nn.Module):
    def __init__(self, cfg: NepaliSSLConfig):
        super().__init__()
        layers = []
        in_ch = 1
        for i, (out_ch, k, s) in enumerate(zip(cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides)):
            layers.append(ConvBlock(in_ch, out_ch, k, s, first=(i == 0)))
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.proj   = nn.Linear(cfg.cnn_channels[-1], cfg.hidden_dim)
        self.norm   = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_samples)
        x = x.unsqueeze(1)              # (B, 1, T)
        x = self.layers(x)              # (B, C, T')
        x = x.transpose(1, 2)          # (B, T', C)
        return self.norm(self.proj(x))  # (B, T', hidden_dim)


# Positional Encoding
class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Transformer Context Network
class TransformerEncoder(nn.Module):
    def __init__(self, cfg: NepaliSSLConfig):
        super().__init__()
        self.pe = SinusoidalPE(cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers,
                                              enable_nested_tensor=False)

    def forward(self, x, padding_mask=None):
        x = self.pe(x)
        return self.encoder(x, src_key_padding_mask=padding_mask)


# Product Quantizer
class ProductQuantizer(nn.Module):
    def __init__(self, cfg: NepaliSSLConfig):
        super().__init__()
        self.G = int(cfg.num_codebooks)
        self.V = int(cfg.codebook_size)
        self.D = int(cfg.codebook_dim)

        self.input_proj  = nn.Linear(cfg.hidden_dim, self.G * self.V)
        self.codebook    = nn.Parameter(torch.FloatTensor(self.G, self.V, self.D))
        self.output_proj = nn.Linear(self.G * self.D, cfg.hidden_dim)
        nn.init.uniform_(self.codebook)

        # Usage counter — reset every check interval in trainer
        self.register_buffer(
            "usage_count",
            torch.zeros(self.G, self.V, dtype=torch.long)
        )

    # ── forward ──────────────────────────────

    def forward(
        self,
        x: torch.Tensor,          # (B, T, hidden_dim)  — detached CNN features
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T, _ = x.shape

        logits = self.input_proj(x)                         # (B, T, G*V)
        logits = logits.view(B, T, self.G, self.V)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            idx   = logits.argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        # ── Perplexity ────────────────────────
        avg_probs  = probs.mean(dim=[0, 1])                 # (G, V)
        perplexity = torch.exp(
            -(avg_probs * (avg_probs + 1e-9).log()).sum(dim=-1)
        ).mean()                                            # scalar, max = V

        # ── Update usage counts ───────────────
        with torch.no_grad():
            hard_idx = logits.argmax(dim=-1)               # (B, T, G)
            for g in range(self.G):
                codes, counts = hard_idx[:, :, g].unique(return_counts=True)
                self.usage_count[g].scatter_add_(
                    0, codes, counts.to(self.usage_count.dtype)
                )

        # ── Weighted codebook lookup ──────────
        q = torch.einsum("btgv,gvd->btgd", probs, self.codebook)  # (B, T, G, D)
        q = self.output_proj(q.reshape(B, T, self.G * self.D))     # (B, T, hidden_dim)

        # ── Commitment loss ───────────────────
        commit_loss = F.mse_loss(x, q.detach())

        return q, perplexity, commit_loss

    # ── Dead code reset ───────────────────────

    @torch.no_grad()
    def reset_dead_codes(
        self,
        encoder_outputs: torch.Tensor,   # (N, hidden_dim) — recent CNN frame features
        threshold: int = 2,
    ) -> int:
        
        n_reset = 0
        N = encoder_outputs.size(0)

        candidates = encoder_outputs[torch.randperm(N, device=encoder_outputs.device)]

        for g in range(self.G):
            dead_mask = self.usage_count[g] < threshold      # (V,)
            n_dead    = int(dead_mask.sum().item())

            if n_dead == 0:
                continue

            # replace_idx = torch.randperm(N, device=encoder_outputs.device)[:n_dead]
            # new_vectors = candidates[replace_idx]            # (n_dead, hidden_dim)

            live_mask  = ~dead_mask
            live_codes = self.codebook[g][live_mask]         # (n_live, D)

            if live_codes.size(0) == 0:
                # Entire codebook dead — re-init from scratch
                nn.init.uniform_(self.codebook[g])
            else:
                live_sample_idx = torch.randint(
                    0, live_codes.size(0), (n_dead,), device=self.codebook.device
                )
                noise = torch.randn(n_dead, self.D, device=self.codebook.device) * 0.1
                self.codebook[g][dead_mask] = live_codes[live_sample_idx] + noise

            self.usage_count[g][dead_mask] = 0
            n_reset += n_dead

        # Reset usage counts for next interval
        self.usage_count.zero_()
        return n_reset


# Masking
def compute_mask(lengths: torch.Tensor, mask_prob: float, mask_length: int,
                 device, max_len: int = None) -> torch.Tensor:
    B = lengths.size(0)
    T = max_len if max_len is not None else int(lengths.max().item())
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i in range(B):
        t = int(lengths[i].item())
        num_spans = max(1, int(mask_prob * t / mask_length))
        starts = torch.randint(0, max(1, t - mask_length), (num_spans,))
        for s in starts:
            mask[i, s: s + mask_length] = True
    return mask

def local_contrastive_loss(
    context:       torch.Tensor,   # (B, T, H)
    targets:       torch.Tensor,   # (B, T, H)
    mask:          torch.Tensor,   # (B, T) bool
    frame_lengths: torch.Tensor,   # (B,)
    num_negatives: int,
    temperature:   float,
) -> torch.Tensor:
    B, T, H = context.shape
    device  = context.device

    # ── Gather all masked positions across the batch at once ──────────
    # mask: (B, T) → flat indices into (B*T)
    flat_mask    = mask.reshape(-1)                          # (B*T,)
    flat_context = context.reshape(-1, H)                    # (B*T, H)
    flat_targets = targets.reshape(-1, H)                    # (B*T, H)

    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Which batch item does each masked position belong to?
    batch_idx = torch.arange(B, device=device).unsqueeze(1) \
                     .expand(B, T).reshape(-1)[flat_mask]   # (N_masked,)

    queries   = flat_context[flat_mask]                      # (N_masked, H)
    positives = flat_targets[flat_mask]                      # (N_masked, H)
    N_masked  = queries.size(0)

    # ── Sample negatives — one row per masked frame ───────────────────
    frame_starts = (batch_idx * T)                           # (N_masked,) — utterance offsets

    neg_offsets  = torch.randint(
        0, T, (N_masked, num_negatives), device=device
    )                                                        # (N_masked, num_neg)

    # Clamp to valid frame range per utterance using frame_lengths
    valid_len    = frame_lengths[batch_idx].unsqueeze(1)     # (N_masked, 1)
    neg_offsets  = neg_offsets % valid_len                   # wrap into valid range

    neg_flat_idx = frame_starts.unsqueeze(1) + neg_offsets   # (N_masked, num_neg)
    negatives    = flat_targets[neg_flat_idx.view(-1)] \
                               .view(N_masked, num_negatives, H)  # (N_masked, num_neg, H)

    # ── Cosine similarity logits ──────────────────────────────────────
    candidates = torch.cat(
        [positives.unsqueeze(1), negatives], dim=1
    )                                                        # (N_masked, 1+num_neg, H)

    q_norm = F.normalize(queries,    dim=-1)                 # (N_masked, H)
    c_norm = F.normalize(candidates, dim=-1)                 # (N_masked, 1+num_neg, H)

    logits = torch.bmm(
        q_norm.unsqueeze(1),
        c_norm.transpose(1, 2)
    ).squeeze(1) / temperature                               # (N_masked, 1+num_neg)

    labels = torch.zeros(N_masked, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


# Main Model
class NepaliSSL(nn.Module):
    def __init__(self, cfg: NepaliSSLConfig = None):
        super().__init__()
        if cfg is None:
            cfg = NepaliSSLConfig()
        self.cfg = cfg

        self.cnn       = CNNEncoder(cfg)
        self.context   = TransformerEncoder(cfg)
        self.quantizer = ProductQuantizer(cfg)
        self.mask_emb  = nn.Parameter(torch.FloatTensor(cfg.hidden_dim).uniform_())

    def get_tau(self, step: int) -> float:
        cfg      = self.cfg
        progress = min(1.0, step / max(1, cfg.codebook_temp_anneal_steps))
        return cfg.codebook_temp_end + (
            cfg.codebook_temp_start - cfg.codebook_temp_end
        ) * (1.0 - progress)

    def forward(
        self,
        waveform: torch.Tensor,
        lengths:  Optional[torch.Tensor] = None,
        step:     int = 0,
    ) -> Dict:
        B, T_wav = waveform.shape
        device   = waveform.device

        if lengths is None:
            lengths = torch.full((B,), T_wav, device=device, dtype=torch.long)

        # 1. CNN features
        features      = self.cnn(waveform)
        T             = features.size(1)
        frame_lengths = (lengths.float() / T_wav * T).long().clamp(max=T)
        pad_mask      = (
            torch.arange(T, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1)
        )

        # 2. Quantize -> targets + losses
        tau = self.get_tau(step)

        if self.cfg.use_quantizer:
            q_targets, perplexity, commit_loss = self.quantizer(features.detach(), tau=tau)
            blend_steps = 20_000
            blend_start = 40_000
            alpha = min(1.0, max(0.0, (step - blend_start) / blend_steps))
            targets = (1 - alpha) * features.detach().clone() + alpha * q_targets
        else:
            targets = features.detach().clone()
            perplexity  = torch.tensor(0.0, device=device)
            commit_loss = torch.tensor(0.0, device=device)

        # 3. Mask
        mask = compute_mask(frame_lengths, self.cfg.mask_prob,
                    self.cfg.mask_length, device,
                    max_len=features.size(1))     
        masked_features = features.clone()
        masked_features[mask] = self.mask_emb

        # 4. Context network
        context = self.context(masked_features, padding_mask=pad_mask)
        norm_penalty = (context.norm(dim=-1).mean() - 1.0).pow(2) * 0.001

        flat_context = context.reshape(-1, context.size(-1))
        dim_var = flat_context.var(dim=0).mean()   # variance across batch per dimension
        var_penalty = (dim_var - 1.0).pow(2) * 0.01

        # 5. Local contrastive loss
        loss_contrast = local_contrastive_loss(
            context       = context,
            targets       = targets,
            mask          = mask,
            frame_lengths = frame_lengths,
            num_negatives = self.cfg.num_negatives,
            temperature   = self.cfg.logit_temp,
        )

        # 6. Total loss
        # loss = loss_contrast + norm_penalty + var_penalty
        loss_diversity = -perplexity
        loss = (
            loss_contrast
            + self.cfg.diversity_loss_weight  * loss_diversity
            + self.cfg.commitment_loss_weight * commit_loss
            + norm_penalty
            + var_penalty
        )

        return {
            "loss":            loss,
            "loss_contrast":   loss_contrast.item(),
            "loss_diversity":  loss_diversity.item(),
            "loss_commitment": commit_loss.item(),
            "perplexity":      perplexity.item(),
            "tau":             tau,
            "num_masked":      int(mask.sum().item()),
            "alpha":           alpha,
            "_features":       features.detach(),  # for dead code reset
        }

    @torch.no_grad()
    def extract_features(
        self,
        waveform: torch.Tensor,
        lengths:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_wav = waveform.shape
        device   = waveform.device
        if lengths is None:
            lengths = torch.full((B,), T_wav, device=device, dtype=torch.long)
        features      = self.cnn(waveform)
        T             = features.size(1)
        frame_lengths = (lengths.float() / T_wav * T).long().clamp(max=T)
        pad_mask      = (
            torch.arange(T, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1)
        )
        return self.context(features, padding_mask=pad_mask)


def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    return total, trainable