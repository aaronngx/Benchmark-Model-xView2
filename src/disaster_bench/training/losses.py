"""
Long-tail loss functions and logit adjustment utilities.

Losses:
  ce          — standard CrossEntropyLoss (baseline, no change)
  cb_ce       — Class-Balanced CrossEntropy (Cui et al. 2019)
  focal       — Focal Loss (Lin et al. 2017), optionally with CB weights
  ldam_drw    — LDAM Loss + Deferred Re-Weighting (Cao et al. 2019)

Utilities:
  compute_class_counts(records)   -> np.ndarray (num_classes,)
  compute_class_prior(records)    -> np.ndarray (num_classes,)  sums to 1
  build_cb_weights(counts, beta)  -> np.ndarray (num_classes,)
  build_ldam_margins(counts, max_m) -> np.ndarray (num_classes,)
  logit_adjust(logits, log_prior, tau) -> adjusted logits
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Class statistics
# ---------------------------------------------------------------------------

def compute_class_counts(records: list[dict], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Count raw training-label occurrences (before sampling).  Shape: (num_classes,)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for r in records:
        idx = r.get("label_idx", -1)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    return counts


def compute_class_prior(records: list[dict], num_classes: int = NUM_CLASSES,
                        eps: float = 1e-8) -> np.ndarray:
    """Empirical class prior from raw training labels.  Sums to 1."""
    counts = compute_class_counts(records, num_classes)
    total  = counts.sum()
    if total == 0:
        return np.ones(num_classes, dtype=np.float64) / num_classes
    return (counts + eps) / (total + num_classes * eps)


# ---------------------------------------------------------------------------
# Class-Balanced weights  (Cui et al. 2019)
# ---------------------------------------------------------------------------

def build_cb_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """
    Effective number of samples: E_n = (1 - beta^n) / (1 - beta).
    Weight = 1 / E_n,  normalized so weights sum to num_classes.
    """
    counts = np.maximum(counts, 1)
    effective = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / effective
    w = w / w.sum() * len(counts)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# LDAM margins
# ---------------------------------------------------------------------------

def build_ldam_margins(counts: np.ndarray, max_m: float = 0.5) -> np.ndarray:
    """
    Per-class margins: m_i = max_m / (n_i ^ (1/4)),  then normalized so
    max margin == max_m.
    """
    counts = np.maximum(counts, 1)
    m = max_m / (counts ** 0.25)
    # Normalize so the largest-count (smallest margin) class gets a floor
    m = m / m.max() * max_m
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Logit adjustment
# ---------------------------------------------------------------------------

def logit_adjust(
    logits: torch.Tensor,
    log_prior: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Add tau * log(p_class) to logits at inference time to correct for
    train-set label imbalance (Menon et al. 2021).

    logits:    (B, C) or (C,)
    log_prior: (C,) — log of empirical training prior, computed from raw counts
    """
    return logits - tau * log_prior.to(logits.device)


def make_log_prior(records: list[dict], num_classes: int = NUM_CLASSES,
                   device: str = "cpu") -> torch.Tensor:
    """Build a (num_classes,) log-prior tensor from training records."""
    prior = compute_class_prior(records, num_classes)
    return torch.tensor(np.log(prior), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    (Lin et al. 2017)

    weight: optional class-weight tensor (e.g. CB weights). If provided,
            acts as alpha_t.
    gamma:  focusing parameter (default 2.0)
    reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # (B, C) log-softmax
        log_p  = F.log_softmax(logits, dim=1)
        # Gather log-prob of true class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt     = log_pt.exp()                                         # (B,)

        focal_weight = (1.0 - pt) ** self.gamma                      # (B,)

        # Class weight (alpha)
        if self.weight is not None:
            w = self.weight.to(logits.device)
            alpha_t = w[targets]                                      # (B,)
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# LDAM Loss
# ---------------------------------------------------------------------------

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al. 2019).

    Penalizes misclassification more for rare classes by adding a per-class
    margin to the true-class logit before computing CE.

    margins: (C,) per-class margins (larger for rarer classes)
    s:       logit scale factor (default 30.0)
    weight:  optional class-weight tensor for DRW phase (None = uniform)
    """
    def __init__(
        self,
        margins: torch.Tensor,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("margins", margins)
        self.s = s
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(logits.device)                      # (C,)
        # Subtract margin from the true-class logit
        # Equivalent: for each sample, reduce logit[y] by m[y]
        batch_m  = margins[targets]                                   # (B,)
        # Build modified logits: all logits the same except true-class - margin
        index    = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        logits_m = logits.clone()
        logits_m[index] -= batch_m

        output = self.s * logits_m
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(output, targets, weight=weight)


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_criterion(
    loss_type: str,
    train_records: list[dict],
    num_classes: int = NUM_CLASSES,
    # CB / focal params
    beta: float = 0.9999,
    gamma: float = 2.0,
    # LDAM params
    max_m: float = 0.5,
    s: float = 30.0,
    drw_start_epoch: int = 10,
    # DRW weights built here, switched in training loop
) -> tuple[nn.Module, dict]:
    """
    Build criterion and return (criterion, info_dict).

    info_dict contains: counts, prior, weights (if any), margins (LDAM).
    DRW is handled externally: call `apply_drw_weights(criterion, counts)` at
    the right epoch.
    """
    counts  = compute_class_counts(train_records, num_classes)
    prior   = compute_class_prior(train_records, num_classes)
    info    = {"counts": counts.tolist(), "prior": prior.tolist(), "loss_type": loss_type}

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()

    elif loss_type == "cb_ce":
        cb_w = build_cb_weights(counts, beta)
        info["weights"] = cb_w.tolist()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(cb_w, dtype=torch.float32)
        )

    elif loss_type == "focal":
        cb_w = build_cb_weights(counts, beta)
        info["weights"] = cb_w.tolist()
        criterion = FocalLoss(
            weight=torch.tensor(cb_w, dtype=torch.float32),
            gamma=gamma,
        )

    elif loss_type == "ldam_drw":
        margins = build_ldam_margins(counts, max_m)
        info["margins"] = margins.tolist()
        info["drw_start_epoch"] = drw_start_epoch
        # Start with uniform weights (DRW phase 1)
        criterion = LDAMLoss(
            margins=torch.tensor(margins, dtype=torch.float32),
            s=s,
            weight=None,
        )

    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. "
                         f"Choose from: ce, cb_ce, focal, ldam_drw")

    return criterion, info


def apply_drw_weights(
    criterion: LDAMLoss,
    counts: np.ndarray,
    beta: float = 0.9999,
    device: str = "cpu",
) -> None:
    """Switch LDAM criterion to class-balanced weights (DRW phase 2)."""
    cb_w = build_cb_weights(counts, beta)
    criterion.weight = torch.tensor(cb_w, dtype=torch.float32, device=device)
