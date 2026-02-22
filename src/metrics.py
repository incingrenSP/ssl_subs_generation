import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, Any

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from src.ssl_model import compute_mask


# 1. Feature Health
def feature_health(features: np.ndarray, threshold: float = 0.01) -> Dict[str, Any]:
    D = features.shape[1]
    std_per_dim = features.std(axis=0)
    active_dims = int((std_per_dim > threshold).sum())

    # Collapse: if top singular value dominates -> features collapsed
    # Use small subsample for speed
    sub = features[:min(2000, len(features))]
    sub_centered = sub - sub.mean(axis=0)
    _, s, _ = np.linalg.svd(sub_centered, full_matrices=False)
    total_var = (s ** 2).sum()
    collapse_score = float(s[0] ** 2 / (total_var + 1e-9))  # closer to 1.0 = collapsed

    norms = np.linalg.norm(features, axis=1)
    return {
        "active_dims":    active_dims,
        "total_dims":     D,
        "active_ratio":   active_dims / D,
        "mean_var":       float(std_per_dim.mean()),
        "collapse_score": collapse_score,   # i want < 0.1
        "norm_mean":      float(norms.mean()),
        "norm_std":       float(norms.std()),
    }


# 2. Isotropy Score
def isotropy_score(features: np.ndarray, n_samples: int = 1000) -> float:
    idx = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
    sub = features[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    # Random pair cosine sims
    n = len(sub)
    i = np.random.randint(0, n, 500)
    j = np.random.randint(0, n, 500)
    sims = (sub[i] * sub[j]).sum(axis=1)
    # For isotropic = mean ~0, for collapsed = mean ~1
    return float(1.0 - abs(sims.mean()))  # want close to 1.0


# 3. Self-ABX proxy (frame-level, no phoneme labels)
def abx_proxy(features_A: np.ndarray, features_B: np.ndarray,
              features_X: np.ndarray, n_trials: int = 1000) -> float:
    # ABX accuracy (want > 0.5, random = 0.5, good > 0.65).
    correct = 0
    N = min(len(features_A), len(features_B), len(features_X))
    n_trials = min(n_trials, N)
    idx = np.random.choice(N, n_trials, replace=False)

    for i in idx:
        a = features_A[i]
        b = features_B[i]
        x = features_X[i]
        d_ax = 1 - np.dot(a, x) / (np.linalg.norm(a) * np.linalg.norm(x) + 1e-8)
        d_bx = 1 - np.dot(b, x) / (np.linalg.norm(b) * np.linalg.norm(x) + 1e-8)
        if d_ax < d_bx:
            correct += 1

    return correct / n_trials


# 4. Clustering quality (k-means proxy)
def clustering_quality(features: np.ndarray, n_clusters: int = 64,
                        max_iter: int = 100) -> Dict[str, float]:

    sub = features[:min(5000, len(features))]
    km = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter,
                          n_init=3, random_state=42)
    labels = km.fit_predict(sub)

    inertia_norm = float(km.inertia_ / len(sub))

    # Silhouette on small subsample (expensive)
    sil_sub = sub[:min(2000, len(sub))]
    sil_labels = labels[:len(sil_sub)]
    try:
        sil = float(silhouette_score(sil_sub, sil_labels, metric="cosine",
                                      sample_size=500))
    except Exception:
        sil = float("nan")

    return {
        "inertia_norm": inertia_norm,
        "silhouette":   sil,          # want > 0
        "n_clusters":   n_clusters,
    }


# 5. Alignment score (positive pair similarity)
def alignment_score(z1: np.ndarray, z2: np.ndarray) -> float:
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    return float((z1 * z2).sum(axis=1).mean())


# Master evaluation function
@torch.no_grad()
def compute_feature_metrics(model, dataloader, device, n_samples: int = 500) -> Dict[str, Any]:
    model.eval()
    all_features = []
    all_pos_pairs_ctx = []   # context (masked) embeddings at masked positions
    all_pos_pairs_tgt = []   # target (CNN) embeddings at masked positions
    
    collected = 0
    for batch in dataloader:
        if collected >= n_samples:
            break
        waveform = batch["waveform"].to(device)
        lengths  = batch["lengths"].to(device)

        # Extract context features
        feats = model.extract_features(waveform, lengths)   # (B, T, H)
        B, T, H = feats.shape

        # Pool per utterance (mean over non-padded frames)
        frame_lengths = (lengths.float() / waveform.size(1) * T).long().clamp(max=T)
        for i in range(B):
            fl = int(frame_lengths[i].item())
            utt_feat = feats[i, :fl].mean(dim=0).cpu().numpy()
            all_features.append(utt_feat)
            collected += 1

    all_features = np.array(all_features)

    results = {}

    # 1. Feature health
    fh = feature_health(all_features)
    results.update({f"fhealth_{k}": v for k, v in fh.items()})

    # 2. Isotropy
    results["isotropy"] = isotropy_score(all_features)

    # 3. Clustering
    cq = clustering_quality(all_features)
    results.update({f"cluster_{k}": v for k, v in cq.items()})

    # 4. Self-ABX proxy (temporal: consecutive pairs ~same, random ~different)
    if len(all_features) >= 6:
        n = len(all_features)
        step = max(1, n // 3)
        A = all_features[:step]
        X = all_features[1: step + 1]    # consecutive ≈ same speaker context
        B = all_features[step * 2: step * 2 + step]  # random
        min_len = min(len(A), len(X), len(B))
        A, X, B_ = A[:min_len], X[:min_len], B[:min_len]
        results["abx_proxy"] = abx_proxy(A, B_, X)

    model.train()
    return results