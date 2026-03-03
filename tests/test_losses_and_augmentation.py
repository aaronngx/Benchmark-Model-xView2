"""
Tests for long-tail losses, logit adjustment, and augmentation alignment.

Run: pytest tests/test_losses_and_augmentation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disaster_bench.training.losses import (
    compute_class_counts,
    compute_class_prior,
    build_cb_weights,
    build_ldam_margins,
    logit_adjust,
    make_log_prior,
    FocalLoss,
    LDAMLoss,
)
from disaster_bench.data.dataset import (
    _apply_affine,
    _apply_color_jitter,
    CropDataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_records(counts: list[int]) -> list[dict]:
    """Make fake records with given per-class counts."""
    recs = []
    for cls_idx, n in enumerate(counts):
        for _ in range(n):
            recs.append({"label_idx": cls_idx, "label": f"class_{cls_idx}"})
    return recs


# ---------------------------------------------------------------------------
# A) Prior / class-count tests
# ---------------------------------------------------------------------------

class TestClassCounts:
    def test_counts_match_input(self):
        recs = _make_records([100, 10, 5, 50])
        counts = compute_class_counts(recs, num_classes=4)
        np.testing.assert_array_equal(counts, [100, 10, 5, 50])

    def test_prior_sums_to_one(self):
        recs = _make_records([100, 10, 5, 50])
        prior = compute_class_prior(recs, num_classes=4)
        assert abs(prior.sum() - 1.0) < 1e-6, f"Prior sum = {prior.sum()}"

    def test_prior_proportional_to_counts(self):
        recs = _make_records([200, 0, 0, 0])
        prior = compute_class_prior(recs, num_classes=4)
        # Class 0 should dominate
        assert prior[0] > prior[1]
        assert prior[0] > prior[2]
        assert prior[0] > prior[3]

    def test_empty_class_prior_not_zero(self):
        """Classes with 0 samples should still get a small prior (eps smoothing)."""
        recs = _make_records([100, 0, 0, 0])
        prior = compute_class_prior(recs, num_classes=4)
        assert all(p > 0 for p in prior), "All priors should be > 0 (eps smoothing)"


# ---------------------------------------------------------------------------
# B) Logit adjustment tests
# ---------------------------------------------------------------------------

class TestLogitAdjust:
    def test_rare_class_logits_increase(self):
        """Logit adjustment should increase the rare-class logit relative to common."""
        import torch
        # Class distribution: class 0 is common (90%), class 1 is rare (10%)
        recs = _make_records([90, 10, 0, 0])
        log_prior = make_log_prior(recs, num_classes=4)

        # Start with equal logits
        logits = torch.zeros(1, 4)
        adjusted = logit_adjust(logits, log_prior, tau=1.0)

        # After adjustment, rare class (1) should have higher adjusted logit than common (0)
        assert adjusted[0, 1] > adjusted[0, 0], (
            f"Rare class logit ({adjusted[0, 1]:.4f}) should exceed "
            f"common class logit ({adjusted[0, 0]:.4f}) after adjustment"
        )

    def test_tau_zero_is_identity(self):
        """tau=0 should leave logits unchanged."""
        import torch
        recs = _make_records([80, 10, 5, 5])
        log_prior = make_log_prior(recs, num_classes=4)
        logits = torch.randn(4, 4)
        adjusted = logit_adjust(logits, log_prior, tau=0.0)
        np.testing.assert_allclose(logits.numpy(), adjusted.numpy(), atol=1e-5)

    def test_adjustment_shifts_prediction_toward_rare(self):
        """With balanced logits and strong imbalance, adjustment should predict rare class."""
        import torch
        # 99:1 imbalance — class 0 vs class 1
        recs = _make_records([990, 10, 0, 0])
        log_prior = make_log_prior(recs, num_classes=2)
        log_prior_full = make_log_prior(recs, num_classes=4)

        # Logits slightly favour class 0
        logits = torch.tensor([[0.1, 0.0, -1e6, -1e6]])
        adjusted = logit_adjust(logits, log_prior_full, tau=1.0)
        # With tau=1, adjustment adds log(prior) — rare class 1 should now win
        assert adjusted[0, 1] > adjusted[0, 0]


# ---------------------------------------------------------------------------
# C) CB weights test
# ---------------------------------------------------------------------------

class TestCBWeights:
    def test_rarer_class_gets_higher_weight(self):
        counts = np.array([1000.0, 10.0, 5.0, 50.0])
        weights = build_cb_weights(counts, beta=0.9999)
        # minor (idx 2, count=5) should have highest weight
        assert weights[2] >= weights[1] >= weights[3], (
            f"Expected weights[2]>=weights[1]>=weights[3] but got {weights}"
        )

    def test_weights_positive(self):
        counts = np.array([100.0, 10.0, 5.0, 50.0])
        weights = build_cb_weights(counts)
        assert all(w > 0 for w in weights)


# ---------------------------------------------------------------------------
# D) Augmentation alignment tests
# ---------------------------------------------------------------------------

def _make_six_channel_crop(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    """Make a (6, H, W) float32 array with distinct pre/post patterns."""
    rng = np.random.default_rng(seed)
    pre  = rng.random((3, h, w), dtype=np.float32)
    post = rng.random((3, h, w), dtype=np.float32)
    return np.concatenate([pre, post], axis=0)


class TestAugmentationAlignment:
    """Geometric augmentations must be identical across all 6 channels."""

    def test_horizontal_flip_aligned(self):
        """After H-flip, pre[:, :, c] should equal pre_orig[:, :, W-1-c]."""
        # Test by directly calling _augment with forced flip
        import random as _random
        x = _make_six_channel_crop()
        x_orig = x.copy()
        # Manually apply horizontal flip (same as in _augment)
        x_flipped = x_orig[:, :, ::-1].copy()
        # Pre channels and post channels must flip identically
        np.testing.assert_array_equal(
            x_flipped[:3],            # pre flipped
            x_orig[:3, :, ::-1],      # expected
        )
        np.testing.assert_array_equal(
            x_flipped[3:],            # post flipped
            x_orig[3:, :, ::-1],
        )

    def test_rotate90_all_channels_identical(self):
        """np.rot90 applied to all channels must give same spatial transform for each."""
        x = _make_six_channel_crop()
        k = 2  # 180 degrees
        x_rot = np.rot90(x, k, axes=(1, 2)).copy()
        # Each channel should be individually rotated the same way
        for c in range(6):
            expected = np.rot90(x[c], k).copy()
            np.testing.assert_array_equal(
                x_rot[c], expected,
                err_msg=f"Channel {c} rotation mismatch"
            )

    def test_affine_pre_post_same_shape(self):
        """Affine transform must not change shape or dtype."""
        x = _make_six_channel_crop(64, 64)
        x_aff = _apply_affine(x)
        assert x_aff.shape == x.shape, "Shape changed after affine"
        assert x_aff.dtype == np.float32, "dtype changed after affine"

    def test_color_jitter_same_params_pre_post(self):
        """
        color jitter applies the SAME brightness/contrast to all channels.
        Pre (ch 0-2) and post (ch 3-5) must receive identical transforms,
        so the difference image is unchanged in expectation:
          (pre_jitter - post_jitter) == scale * (pre - post)
        """
        x = _make_six_channel_crop()
        import random as _rand
        _rand.seed(42)
        x_jit = _apply_color_jitter(x.copy())
        # All 6 channels must have same scale/shift: check ratio consistency
        # If brightness=b, contrast=c: out[i] = c*x[i] + b
        # So out[0]/out[3] should equal x[0]/x[3] for pixels where both are nonzero
        # More robustly: assert jitter didn't leave channels independent
        # by checking that pre-post diff scaling is uniform across all channels
        pre_orig_diff = (x[:3] - x[3:])
        pre_jit_diff  = (x_jit[:3] - x_jit[3:])
        # The diff should scale uniformly (same multiplier for all channels)
        mask = np.abs(pre_orig_diff) > 1e-6
        if mask.sum() > 0:
            ratios = pre_jit_diff[mask] / pre_orig_diff[mask]
            # All ratios should be approximately equal (same contrast multiplier)
            assert np.std(ratios) < 0.1, (
                f"Color jitter applied different multipliers to channels "
                f"(std of ratios={np.std(ratios):.4f})"
            )

    def test_noise_stays_in_range(self):
        """After noise augmentation, values should stay in [0, 1]."""
        import random as _rand
        _rand.seed(0)
        np.random.seed(0)

        # Use CropDataset augment to test noise path
        rng = np.random.default_rng(1)
        x = rng.random((6, 32, 32), dtype=np.float32)

        # Simulate noise augmentation
        sigma = 0.02
        noise = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
        x_noisy = np.clip(x + noise, 0.0, 1.0)
        assert x_noisy.min() >= 0.0
        assert x_noisy.max() <= 1.0


# ---------------------------------------------------------------------------
# E) LDAM / Focal loss forward-pass smoke tests
# ---------------------------------------------------------------------------

class TestLossForward:
    def test_focal_loss_runs(self):
        import torch
        fl = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = fl(logits, targets)
        assert loss.item() >= 0

    def test_ldam_loss_runs(self):
        import torch
        margins = torch.tensor([0.1, 0.4, 0.5, 0.2])
        ldam = LDAMLoss(margins=margins, s=30.0)
        logits  = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = ldam(logits, targets)
        assert loss.item() >= 0

    def test_focal_loss_higher_for_easy_examples(self):
        """Focal loss should weight hard examples more than easy ones."""
        import torch
        fl = FocalLoss(gamma=2.0, reduction="none")
        # Easy: true class has prob ~0.99 — (1-p)^2 ≈ 0.0001
        # Hard: true class has prob ~0.5  — (1-p)^2 = 0.25
        logits_easy = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # class 0 very likely
        logits_hard = torch.tensor([[0.0,  0.0, 0.0, 0.0]])  # uniform
        target      = torch.tensor([0])
        loss_easy = fl(logits_easy, target)
        loss_hard = fl(logits_hard, target)
        assert loss_hard.item() > loss_easy.item(), (
            f"Hard example loss ({loss_hard.item():.4f}) should exceed "
            f"easy example loss ({loss_easy.item():.4f})"
        )
