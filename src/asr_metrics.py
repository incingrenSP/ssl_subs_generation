import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict
from src.tokenizer import CharTokenizer

# Greedy Decoding

def greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[int]:

    indices = log_probs.argmax(dim=-1).tolist()
    decoded = []
    prev = -1
    for idx in indices:
        if idx != blank and idx != prev:
            decoded.append(idx)
        prev = idx
        
    return decoded


def decode_batch(
    log_probs: torch.Tensor,       # (B, T, vocab_size)
    frame_lengths: torch.Tensor,   # (B,)
    tokenizer: CharTokenizer,
) -> List[str]:

    results = []
    for i in range(log_probs.size(0)):
        T = frame_lengths[i].item()
        ids  = greedy_decode(log_probs[i, :T])
        text = tokenizer.decode(ids)
        results.append(text)
    return results

# Beam Decoding

def build_beam_decoder(
    lexicon_path: str,
    tokenizer,
    beam_size:       int   = 50,
    beam_threshold:  float = 50.0,
):
    from torchaudio.models.decoder import ctc_decoder

    # Build tokens list — must match tokenizer exactly
    # index 0 = blank, rest = id2char order
    tokens = []
    for i in range(tokenizer.get_vocab_size()):
        if i == tokenizer.blank_id:
            tokens.append(tokenizer.BLANK_TOKEN)   # e.g. "<blank>"
        elif i in tokenizer.id2char:
            tokens.append(tokenizer.id2char[i])
        else:
            tokens.append(f"<unk{i}>")

    # decoder = ctc_decoder(
    #     lexicon        = lexicon_path,
    #     tokens         = tokens,
    #     lm             = None,           # no LM for now
    #     nbest          = 1,
    #     beam_size      = beam_size,
    #     beam_size_token= 30,
    #     beam_threshold = beam_threshold,
    #     sil_token      = tokenizer.SPACE_TOKEN,   # e.g. "<space>"
    #     blank_token    = tokenizer.BLANK_TOKEN,   # e.g. "<blank>"
    # )
    
    decoder = ctc_decoder(
        lexicon=None,
        tokens=tokens,
        lm=None,
        nbest=1,
        beam_size=50,
        beam_size_token=10,
        beam_threshold=50,
        word_score=0,
        unk_score=0,           # was float('-inf') — killed everything
        sil_score=0,
        log_add=False,
        blank_token='<blank>',
        sil_token='<space>',
    )
    return decoder

def beam_decode_batch(
    log_probs:     torch.Tensor,
    frame_lengths: torch.Tensor,
    decoder,
    tokenizer,
) -> List[str]:
    
    results = decoder(log_probs.cpu(), frame_lengths.cpu())
    decoded = []
    for result in results:
        hyp = result[0]
        # lexicon-free: tokens contains token indices, words is empty
        if hyp.words:
            text = " ".join(hyp.words)
        else:
            # reconstruct from token indices
            token_ids = hyp.tokens.tolist()
            text = tokenizer.decode(token_ids)
        decoded.append(text)
    return decoded

# def beam_decode_batch(
#     log_probs:     torch.Tensor,   # (B, T, vocab)
#     frame_lengths: torch.Tensor,   # (B,)
#     decoder,
# ) -> List[str]:
    
#     results = decoder(log_probs.cpu(), frame_lengths.cpu())
#     decoded = []
#     for result in results:
#         # result[0] is best hypothesis
#         words = result[0].words
#         decoded.append(" ".join(words))
#     return decoded

# Error rates

def edit_distance(a: List, b: List) -> int:

    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return edit_distance(list(hypothesis), list(reference)) / len(reference)


def wer(hypothesis: str, reference: str) -> float:

    h = hypothesis.split()
    r = reference.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(h, r) / len(r)


def compute_cer_wer(
    hypotheses: List[str],
    references: List[str],
) -> Tuple[float, float]:
    
    total_char_errors = 0
    total_chars       = 0
    total_word_errors = 0
    total_words       = 0

    for hyp, ref in zip(hypotheses, references):
        # CER — character level
        total_char_errors += edit_distance(list(hyp), list(ref))
        total_chars       += len(ref)

        # WER — word level
        hyp_words = hyp.split()
        ref_words = ref.split()
        total_word_errors += edit_distance(hyp_words, ref_words)
        total_words       += len(ref_words)

    cer = total_char_errors / max(total_chars, 1)
    wer = total_word_errors / max(total_words, 1)

    return cer, wer

# Beam Decoding Evaluation
def evaluate_beam(
    model,
    val_loader,
    decoder,
    device: str = "cuda",
) -> Tuple[float, float, float]:
        
    model.eval()
    total_char_errors = 0
    total_chars       = 0
    total_word_errors = 0
    total_words       = 0
    total_loss = 0.0
    total_batches = 0

    def edit_distance(a, b):
        # Simple Wagner-Fischer
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[j] = prev[j-1]
                else:
                    dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
        return dp[n]

    with torch.no_grad():
        for batch in val_loader:
            waveform      = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            tokens        = batch["targets"].to(device)
            token_lengths = batch["target_lengths"].to(device)
            references    = batch["transcripts"]

            log_probs, frame_lengths = model(waveform, audio_lengths)
            
            loss = torch.nn.functional.ctc_loss(
                log_probs.transpose(0, 1),
                tokens,
                frame_lengths,
                token_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            total_loss += loss.item()
            total_batches += 1

            hypotheses = beam_decode_batch(log_probs, frame_lengths, decoder)

            for hyp, ref in zip(hypotheses, references):
                total_char_errors += edit_distance(list(hyp), list(ref))
                total_chars       += max(len(ref), 1)
                total_word_errors += edit_distance(hyp.split(), ref.split())
                total_words       += max(len(ref.split()), 1)

    mean_loss = total_loss / max(total_batches, 1)
    cer = total_char_errors / max(total_chars, 1)
    wer = total_word_errors / max(total_words, 1)
    return mean_loss, cer, wer

# Evaluation loop

@torch.no_grad()
def evaluate(
    model,
    val_dl,
    tokenizer: CharTokenizer,
    device: str = "cuda",
    max_batches: int = None,
) -> Tuple[float, float]:
    
    model.eval()

    total_loss = 0.0
    n_batches  = 0
    hypotheses = []
    references = []

    for i, batch in enumerate(val_dl):
        if max_batches and i >= max_batches:
            break

        waveform      = batch["waveforms"].to(device)
        lengths       = batch["audio_lengths"].to(device)
        tokens        = batch["targets"].to(device)
        token_lengths = batch["target_lengths"].to(device)
        texts         = batch["transcripts"]

        log_probs, frame_lengths = model(waveform, lengths)

        loss = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1),
            tokens,
            frame_lengths,
            token_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        total_loss += loss.item()
        n_batches  += 1

        hyps = decode_batch(log_probs, frame_lengths, tokenizer)
        hypotheses.extend(hyps)
        references.extend(texts)

    mean_loss = total_loss / max(n_batches, 1)
    mean_cer, mean_wer = compute_cer_wer(hypotheses, references)
    return mean_loss, mean_cer, mean_wer


# Geometry metrics

def compute_isotropy(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    if S[0] < 1e-9:
        return 0.0
    return float(S[-1] / S[0])


@torch.no_grad()
def compute_geometry(
    model,
    val_dl,
    device: str = "cuda",
    max_batches: int = 5,
) -> Dict[str, float]:
    model.eval()

    all_repr = []
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        waveform = batch["waveforms"].to(device)
        lengths  = batch["audio_lengths"].to(device)

        # Get raw context vectors
        reprs = model.get_representations(waveform, lengths)  # (B, T, 256)

        # Flatten batch and time dims, move to CPU
        B, T, H = reprs.shape
        all_repr.append(reprs.reshape(B * T, H).cpu().float().numpy())

    if not all_repr:
        return {}

    vectors = np.concatenate(all_repr, axis=0)  # (N, 256)

    norms       = np.linalg.norm(vectors, axis=-1)
    variances   = vectors.var(axis=0)
    active_dims = int((variances > 0.01).sum())
    isotropy    = compute_isotropy(vectors)

    return {
        "norm_mean":   float(norms.mean()),
        "norm_std":    float(norms.std()),
        "active_dims": active_dims,
        "isotropy":    isotropy,
    }