"""
Tests for nested CV calibration split, deterministic threshold fitting,
and predict_with_policy behaviour.

Run: pytest tests/test_nested_cv_and_threshold.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_damage import (
    _tile_grouped_calib_split,
    _fit_thresholds,
    _fit_thresholds_ordinal,
    predict_with_policy,
    _predict_cascade,
    _fit_thresholds_cascade,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_records(tile_counts: dict[str, list[int]]) -> list[dict]:
    """
    tile_counts: {tile_id: [n_no, n_minor, n_major, n_dest]}
    Returns a flat list of records with tile_id and label_idx.
    """
    records = []
    for tile_id, counts in tile_counts.items():
        for label_idx, n in enumerate(counts):
            for _ in range(n):
                records.append({"tile_id": tile_id, "label_idx": label_idx,
                                 "label": ["no-damage", "minor-damage",
                                           "major-damage", "destroyed"][label_idx]})
    return records


# ---------------------------------------------------------------------------
# A) Tile-grouped calib split — no tile overlap
# ---------------------------------------------------------------------------

class TestTileGroupedCalibSplit:
    def test_no_tile_overlap(self):
        """No tile_id should appear in both train_fit and calib."""
        records = _make_records({
            f"tile_{i:03d}": [10, 0, 0, 0] for i in range(50)
        })
        train_fit, calib = _tile_grouped_calib_split(records, calib_fraction=0.2, seed=42)
        fit_tiles   = {r["tile_id"] for r in train_fit}
        calib_tiles = {r["tile_id"] for r in calib}
        assert fit_tiles.isdisjoint(calib_tiles), \
            f"Overlap found: {fit_tiles & calib_tiles}"

    def test_all_records_accounted_for(self):
        """Every record appears in exactly one of train_fit / calib."""
        records = _make_records({f"t{i}": [5, 1, 1, 2] for i in range(20)})
        train_fit, calib = _tile_grouped_calib_split(records, calib_fraction=0.2, seed=0)
        assert len(train_fit) + len(calib) == len(records)

    def test_calib_fraction_respected(self):
        """Calib tile count is approximately calib_fraction of total tiles."""
        n_tiles = 100
        records = _make_records({f"t{i}": [8, 0, 0, 2] for i in range(n_tiles)})
        train_fit, calib = _tile_grouped_calib_split(records, calib_fraction=0.2, seed=7)
        calib_tiles = {r["tile_id"] for r in calib}
        assert 10 <= len(calib_tiles) <= 30, \
            f"Expected ~20 calib tiles, got {len(calib_tiles)}"

    def test_outer_test_tiles_not_in_calib(self):
        """
        Simulate the nested CV flow: outer_test tiles must be disjoint from
        both train_fit and calib.
        """
        all_records = _make_records({f"tile_{i:03d}": [10, 1, 1, 3] for i in range(40)})
        # Outer fold: first 8 tiles
        outer_test_tiles = {f"tile_{i:03d}" for i in range(8)}
        train_all = [r for r in all_records if r["tile_id"] not in outer_test_tiles]

        train_fit, calib = _tile_grouped_calib_split(train_all, calib_fraction=0.2, seed=99)

        fit_tiles   = {r["tile_id"] for r in train_fit}
        calib_tiles = {r["tile_id"] for r in calib}

        assert outer_test_tiles.isdisjoint(fit_tiles),   "outer test leaked into train_fit"
        assert outer_test_tiles.isdisjoint(calib_tiles), "outer test leaked into calib"
        assert fit_tiles.isdisjoint(calib_tiles),        "train_fit / calib overlap"


# ---------------------------------------------------------------------------
# B) Threshold fitting — recall = 1.0 guaranteed
# ---------------------------------------------------------------------------

class TestFitThresholds:
    def _make_probs_and_labels(self, pos_probs_minor, pos_probs_major,
                                n_neg=100, seed=0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_pos_minor = len(pos_probs_minor)
        n_pos_major = len(pos_probs_major)

        # Negative samples: p(minor) and p(major) from uniform [0, 0.3]
        neg_minor = rng.uniform(0, 0.3, n_neg)
        neg_major = rng.uniform(0, 0.3, n_neg)

        probs = np.zeros((n_neg + n_pos_minor + n_pos_major, 4), dtype=np.float32)
        labels = np.zeros(n_neg + n_pos_minor + n_pos_major, dtype=np.int64)

        # Negatives: class 0 (no-damage)
        probs[:n_neg, 0] = 1.0 - neg_minor - neg_major
        probs[:n_neg, 1] = neg_minor
        probs[:n_neg, 2] = neg_major
        probs[:n_neg, 3] = 0.0
        labels[:n_neg] = 0

        # Minor positives
        for j, p in enumerate(pos_probs_minor):
            idx = n_neg + j
            probs[idx, 1] = p
            probs[idx, 0] = 1.0 - p
            labels[idx] = 1

        # Major positives
        for j, p in enumerate(pos_probs_major):
            idx = n_neg + n_pos_minor + j
            probs[idx, 2] = p
            probs[idx, 0] = 1.0 - p
            labels[idx] = 2

        return probs, labels

    def test_recall_1_achieved_on_calib(self):
        """When target_recall=1.0, every calib positive must be above threshold."""
        pos_minor = [0.31, 0.45, 0.72, 0.55]
        pos_major = [0.40, 0.60, 0.25]
        probs, labels = self._make_probs_and_labels(pos_minor, pos_major)

        thresholds, summary = _fit_thresholds(
            probs, labels,
            target_recall_minor=1.0, target_recall_major=1.0,
            never_miss=True,
        )

        tau_minor = thresholds["minor"]
        tau_major  = thresholds["major"]

        # Every minor positive must have prob >= tau_minor
        minor_mask = (labels == 1)
        assert all(probs[minor_mask, 1] >= tau_minor), \
            "Not all minor positives are above fitted threshold"

        major_mask = (labels == 2)
        assert all(probs[major_mask, 2] >= tau_major), \
            "Not all major positives are above fitted threshold"

    def test_recall_1_achieves_target(self):
        """Summary should report achieved_recall == 1.0 for recall=1.0 target."""
        pos_minor = [0.50, 0.65, 0.80]
        pos_major = [0.45, 0.70]
        probs, labels = self._make_probs_and_labels(pos_minor, pos_major)

        _, summary = _fit_thresholds(
            probs, labels,
            target_recall_minor=1.0, target_recall_major=1.0,
            never_miss=True,
        )
        assert summary["minor"]["achieved_recall"] == 1.0, \
            f"Minor achieved_recall={summary['minor']['achieved_recall']}"
        assert summary["major"]["achieved_recall"] == 1.0, \
            f"Major achieved_recall={summary['major']['achieved_recall']}"

    def test_partial_recall_is_tighter(self):
        """
        Threshold for recall=0.8 should be >= threshold for recall=1.0
        (tighter threshold = fewer positives flagged).
        """
        pos_minor = [0.2, 0.35, 0.5, 0.65, 0.8]
        pos_major = [0.3, 0.5, 0.7]
        probs, labels = self._make_probs_and_labels(pos_minor, pos_major)

        thr_100, _ = _fit_thresholds(probs, labels, 1.0, 1.0, never_miss=True)
        thr_080, _ = _fit_thresholds(probs, labels, 0.8, 0.8, never_miss=False)

        assert thr_080["minor"] >= thr_100["minor"], \
            "80% recall threshold should be >= 100% recall threshold"
        assert thr_080["major"] >= thr_100["major"], \
            "80% recall threshold should be >= 100% recall threshold"

    def test_zero_positives_fallback(self):
        """When n_pos == 0, tau = 0.0 (never_miss) or 0.5 (default)."""
        probs  = np.zeros((50, 4), dtype=np.float32)
        probs[:, 0] = 1.0
        labels = np.zeros(50, dtype=np.int64)   # all no-damage

        thr_nm, summ_nm = _fit_thresholds(probs, labels, 1.0, 1.0, never_miss=True)
        thr_def, summ_def = _fit_thresholds(probs, labels, 0.8, 0.8, never_miss=False)

        assert thr_nm["minor"] == 0.0
        assert thr_nm["major"] == 0.0
        assert thr_def["minor"] == 0.5
        assert thr_def["major"] == 0.5
        assert summ_nm["minor"]["n_pos"] == 0
        assert summ_nm["major"]["n_pos"] == 0


# ---------------------------------------------------------------------------
# C) predict_with_policy — edge cases
# ---------------------------------------------------------------------------

class TestPredictWithPolicy:
    def _probs(self, rows: list[list[float]]) -> np.ndarray:
        return np.array(rows, dtype=np.float32)

    def test_argmax_policy(self):
        """policy='argmax' must always return argmax regardless of thresholds."""
        probs = self._probs([
            [0.1, 0.6, 0.2, 0.1],   # argmax=1
            [0.8, 0.05, 0.1, 0.05], # argmax=0
        ])
        thresholds = {"minor": 0.05, "major": 0.05}   # very low — would catch both
        preds = predict_with_policy(probs, "argmax", thresholds)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_no_thresholds_returns_argmax(self):
        """policy='per_class_threshold' with thresholds=None falls back to argmax."""
        probs = self._probs([[0.7, 0.1, 0.1, 0.1]])
        preds = predict_with_policy(probs, "per_class_threshold", None)
        assert preds[0] == 0

    def test_single_minor_candidate(self):
        """Only minor exceeds threshold -> predict minor."""
        probs = self._probs([[0.5, 0.4, 0.05, 0.05]])
        thresholds = {"minor": 0.30, "major": 0.20}
        # p(minor)=0.4 >= 0.30 -> minor_cand
        # p(major)=0.05 < 0.20 -> not major_cand
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert preds[0] == 1

    def test_single_major_candidate(self):
        """Only major exceeds threshold -> predict major."""
        probs = self._probs([[0.6, 0.05, 0.3, 0.05]])
        thresholds = {"minor": 0.20, "major": 0.25}
        # p(minor)=0.05 < 0.20 -> not minor_cand
        # p(major)=0.30 >= 0.25 -> major_cand
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert preds[0] == 2

    def test_both_candidates_minor_wins_by_margin(self):
        """Both exceed threshold; minor has larger (prob - tau) margin -> predict minor."""
        # minor_margin = 0.40 - 0.30 = 0.10
        # major_margin = 0.30 - 0.25 = 0.05
        probs = self._probs([[0.25, 0.40, 0.30, 0.05]])
        thresholds = {"minor": 0.30, "major": 0.25}
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert preds[0] == 1, f"Expected minor(1), got {preds[0]}"

    def test_both_candidates_major_wins_by_margin(self):
        """Both exceed threshold; major has larger margin -> predict major."""
        # minor_margin = 0.35 - 0.30 = 0.05
        # major_margin = 0.40 - 0.25 = 0.15
        probs = self._probs([[0.20, 0.35, 0.40, 0.05]])
        thresholds = {"minor": 0.30, "major": 0.25}
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert preds[0] == 2, f"Expected major(2), got {preds[0]}"

    def test_no_candidate_falls_back_to_argmax(self):
        """Neither class exceeds its threshold -> use argmax (no-damage wins here)."""
        probs = self._probs([[0.70, 0.10, 0.10, 0.10]])
        thresholds = {"minor": 0.50, "major": 0.50}
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert preds[0] == 0   # argmax = no-damage

    def test_batch_processing(self):
        """Verify batch of N samples is processed correctly."""
        probs = self._probs([
            [0.8, 0.05, 0.10, 0.05],   # no minor/major cand -> argmax=0
            [0.3, 0.60, 0.05, 0.05],   # minor cand only     -> 1
            [0.3, 0.05, 0.60, 0.05],   # major cand only     -> 2
            [0.2, 0.40, 0.35, 0.05],   # both; minor margin=0.10 major=0.10 -> minor(tie->minor)
        ])
        thresholds = {"minor": 0.30, "major": 0.25}
        preds = predict_with_policy(probs, "per_class_threshold", thresholds)
        assert list(preds) == [0, 1, 2, 1]


# ---------------------------------------------------------------------------
# D) ordinal_threshold decision rule correctness
# ---------------------------------------------------------------------------

class TestOrdinalThresholdPolicy:
    def _probs(self, rows: list[list[float]]) -> np.ndarray:
        return np.array(rows, dtype=np.float32)

    def test_below_both_thresholds_predicts_no_damage(self):
        """p_damage < tau_damage -> predict no-damage."""
        probs = self._probs([[0.95, 0.02, 0.02, 0.01]])
        # p_damage = 0.05, p_severe = 0.03
        thresholds = {"tau_damage": 0.10, "tau_severe": 0.05}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert preds[0] == 0

    def test_above_damage_only_predicts_minor(self):
        """p_damage >= tau_damage but p_severe < tau_severe -> argmax over {1,2,3}."""
        # p_damage = 0.60, p_severe = 0.05 (major+dest)
        probs = self._probs([[0.40, 0.55, 0.03, 0.02]])
        thresholds = {"tau_damage": 0.50, "tau_severe": 0.20}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert preds[0] == 1  # minor has highest prob among {1,2,3}

    def test_above_damage_only_predicts_best_of_damaged(self):
        """p_damage >= tau_damage, p_severe < tau_severe -> argmax({minor,major,dest})."""
        # p_minor=0.2, p_major=0.5, p_dest=0.1 => argmax=major(2)
        probs = self._probs([[0.20, 0.20, 0.50, 0.10]])
        thresholds = {"tau_damage": 0.50, "tau_severe": 0.70}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert preds[0] == 2

    def test_above_severe_threshold_predicts_major(self):
        """p_severe >= tau_severe -> argmax over {major(2), destroyed(3)}."""
        # p_severe = 0.60+0.05 = 0.65; major > destroyed -> predict major
        probs = self._probs([[0.25, 0.10, 0.60, 0.05]])
        thresholds = {"tau_damage": 0.30, "tau_severe": 0.50}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert preds[0] == 2

    def test_above_severe_threshold_predicts_destroyed(self):
        """p_severe >= tau_severe, destroyed > major -> predict destroyed(3)."""
        probs = self._probs([[0.10, 0.05, 0.15, 0.70]])
        thresholds = {"tau_damage": 0.30, "tau_severe": 0.50}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert preds[0] == 3

    def test_argmax_policy_unchanged(self):
        """Ordinal thresholds must not affect argmax policy."""
        probs = self._probs([[0.1, 0.6, 0.2, 0.1]])
        thresholds = {"tau_damage": 0.01, "tau_severe": 0.01}
        preds = predict_with_policy(probs, "argmax", thresholds)
        assert preds[0] == 1   # argmax is minor

    def test_batch_ordinal(self):
        """Batch processing with ordinal_threshold is correct."""
        probs = self._probs([
            [0.95, 0.02, 0.02, 0.01],   # p_damage=0.05 < 0.10 -> 0
            [0.40, 0.55, 0.03, 0.02],   # p_damage=0.60>=0.10, p_severe=0.05<0.20 -> argmax{1,2,3}=1
            [0.10, 0.05, 0.60, 0.25],   # p_severe=0.85>=0.20 -> argmax{2,3}=2
            [0.05, 0.05, 0.20, 0.70],   # p_severe=0.90>=0.20 -> argmax{2,3}=3
        ])
        thresholds = {"tau_damage": 0.10, "tau_severe": 0.20}
        preds = predict_with_policy(probs, "ordinal_threshold", thresholds)
        assert list(preds) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# E) Ordinal threshold fitting — recall = 1.0 guaranteed
# ---------------------------------------------------------------------------

class TestFitThresholdsOrdinal:
    def _make_probs_labels(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Synthetic 4-class probs:
          label 0 (no-damage):  p_no ~ high
          label 1 (minor):      p_minor ~ 0.5, p_no ~ 0.4
          label 2 (major):      p_major ~ 0.6, p_no ~ 0.3
          label 3 (destroyed):  p_dest  ~ 0.7, p_no ~ 0.2
        """
        rng = np.random.default_rng(seed)
        rows, labs = [], []
        for _ in range(80):
            p = rng.dirichlet([8, 0.5, 0.3, 0.2])
            rows.append(p); labs.append(0)
        for _ in range(15):
            p = rng.dirichlet([1.5, 4, 1, 0.5])
            rows.append(p); labs.append(1)
        for _ in range(12):
            p = rng.dirichlet([1, 1, 5, 1])
            rows.append(p); labs.append(2)
        for _ in range(8):
            p = rng.dirichlet([0.5, 0.5, 1, 6])
            rows.append(p); labs.append(3)
        return np.array(rows, dtype=np.float32), np.array(labs, dtype=np.int64)

    def test_recall_1_damage(self):
        """With target_recall=1.0, all damage positives must be >= tau_damage."""
        probs, labels = self._make_probs_labels()
        thresholds, summary = _fit_thresholds_ordinal(
            probs, labels,
            target_recall_damage=1.0, target_recall_severe=1.0, never_miss=True,
        )
        tau_d = thresholds["tau_damage"]
        p_damage = probs[:, 1] + probs[:, 2] + probs[:, 3]
        pos_mask = np.isin(labels, [1, 2, 3])
        assert all(p_damage[pos_mask] >= tau_d), \
            "Some damage positives are below tau_damage"

    def test_recall_1_severe(self):
        """With target_recall=1.0, all severe positives must be >= tau_severe."""
        probs, labels = self._make_probs_labels()
        thresholds, _ = _fit_thresholds_ordinal(
            probs, labels,
            target_recall_damage=1.0, target_recall_severe=1.0, never_miss=True,
        )
        tau_s = thresholds["tau_severe"]
        p_severe = probs[:, 2] + probs[:, 3]
        pos_mask = np.isin(labels, [2, 3])
        assert all(p_severe[pos_mask] >= tau_s), \
            "Some severe positives are below tau_severe"

    def test_achieved_recall_in_summary(self):
        """Summary achieved_recall == 1.0 for recall=1.0 target."""
        probs, labels = self._make_probs_labels()
        _, summary = _fit_thresholds_ordinal(
            probs, labels,
            target_recall_damage=1.0, target_recall_severe=1.0, never_miss=True,
        )
        assert summary["tau_damage"]["achieved_recall"] == 1.0, \
            f"damage achieved_recall={summary['tau_damage']['achieved_recall']}"
        assert summary["tau_severe"]["achieved_recall"] == 1.0, \
            f"severe achieved_recall={summary['tau_severe']['achieved_recall']}"

    def test_tighter_threshold_for_lower_recall(self):
        """Threshold for recall=0.8 >= threshold for recall=1.0."""
        probs, labels = self._make_probs_labels()
        thr_100, _ = _fit_thresholds_ordinal(probs, labels, 1.0, 1.0, never_miss=True)
        thr_080, _ = _fit_thresholds_ordinal(probs, labels, 0.8, 0.8, never_miss=False)
        assert thr_080["tau_damage"] >= thr_100["tau_damage"], \
            "80% tau_damage should be >= 100% tau_damage"
        assert thr_080["tau_severe"] >= thr_100["tau_severe"], \
            "80% tau_severe should be >= 100% tau_severe"

    def test_zero_positives_fallback(self):
        """All no-damage: tau = 0.0 (never_miss) or 0.5 (default)."""
        probs  = np.zeros((50, 4), dtype=np.float32)
        probs[:, 0] = 1.0
        labels = np.zeros(50, dtype=np.int64)

        thr_nm, _ = _fit_thresholds_ordinal(probs, labels, 1.0, 1.0, never_miss=True)
        thr_df, _ = _fit_thresholds_ordinal(probs, labels, 0.8, 0.8, never_miss=False)

        assert thr_nm["tau_damage"] == 0.0
        assert thr_nm["tau_severe"] == 0.0
        assert thr_df["tau_damage"] == 0.5
        assert thr_df["tau_severe"] == 0.5


# ---------------------------------------------------------------------------
# F) crossfit_pool — pooling excludes the evaluated fold
# ---------------------------------------------------------------------------

class TestCrossfitPool:
    """
    Verify the leakage-safety guarantee: calib predictions saved for fold k
    must NOT be used when evaluating fold k outer test.

    We simulate the pool assembly that happens inside run():
    - calib_preds for each fold are represented by unique sentinel labels.
    - When pooling for fold k, only sibling fold data should appear.
    """

    def _make_fold_preds(self, n_folds: int = 5, n_per_fold: int = 20) -> list[np.ndarray]:
        """Return list of y_true arrays, one per fold, with unique sentinel values."""
        rng = np.random.default_rng(99)
        fold_labels = []
        for f in range(n_folds):
            # Unique sentinel: all labels are 'f' (won't happen in real data but easy to check)
            labs = np.full(n_per_fold, f, dtype=np.int64)
            fold_labels.append(labs)
        return fold_labels

    def test_excluded_fold_not_in_pool(self):
        """Pool assembled for fold k must contain no samples from fold k."""
        fold_labels = self._make_fold_preds(n_folds=5)
        for eval_fold in range(5):
            pool = np.concatenate([
                fold_labels[s] for s in range(5) if s != eval_fold
            ])
            # The sentinel value 'eval_fold' must not appear in the pool
            assert eval_fold not in pool, \
                f"Fold {eval_fold} labels leaked into its own pool!"

    def test_pool_contains_all_sibling_folds(self):
        """Pool for fold k must contain samples from all other 4 folds."""
        fold_labels = self._make_fold_preds(n_folds=5, n_per_fold=20)
        for eval_fold in range(5):
            pool = np.concatenate([
                fold_labels[s] for s in range(5) if s != eval_fold
            ])
            for sib in range(5):
                if sib == eval_fold:
                    continue
                assert sib in pool, \
                    f"Sibling fold {sib} missing from pool for eval_fold {eval_fold}"

    def test_pool_size_correct(self):
        """Pool should contain exactly (n_folds - 1) * n_per_fold samples."""
        n_folds, n_per_fold = 5, 30
        fold_labels = self._make_fold_preds(n_folds, n_per_fold)
        for eval_fold in range(n_folds):
            pool = np.concatenate([
                fold_labels[s] for s in range(n_folds) if s != eval_fold
            ])
            assert len(pool) == (n_folds - 1) * n_per_fold, \
                f"Pool size {len(pool)} != {(n_folds - 1) * n_per_fold}"


# ---------------------------------------------------------------------------
# F) Cascade threshold policy — predict_cascade decision rule
# ---------------------------------------------------------------------------

class TestCascadeThresholdPolicy:
    """Tests for _predict_cascade: correct decision rules for cascade branches."""

    def _make_inputs(self, n=10, seed=0):
        rng = np.random.default_rng(seed)
        p_damage       = rng.uniform(0, 1, n)
        p_severe       = rng.uniform(0, 1, n)
        severity_logits = rng.standard_normal((n, 3))
        return p_damage, p_severe, severity_logits

    def test_no_damage_when_both_below_tau(self):
        """All samples with p_damage < tau_damage and p_severe < tau_severe → no-damage (0)."""
        p_damage = np.array([0.1, 0.2, 0.05])
        p_severe = np.array([0.1, 0.15, 0.05])
        severity_logits = np.ones((3, 3))
        preds = _predict_cascade(p_damage, p_severe, severity_logits, tau_damage=0.5, tau_severe=0.5)
        assert (preds == 0).all(), f"Expected all no-damage, got {preds}"

    def test_severe_branch_overrides_damage(self):
        """p_severe >= tau_severe → pred must be in {2, 3} (major or destroyed)."""
        p_damage = np.array([0.9])
        p_severe = np.array([0.9])
        # severity_logits: major(1) >> destroyed(2) >> minor(0)
        severity_logits = np.array([[0.0, 10.0, 1.0]])
        preds = _predict_cascade(p_damage, p_severe, severity_logits, tau_damage=0.5, tau_severe=0.5)
        assert preds[0] == 2, f"Expected major(2), got {preds[0]}"  # severity idx 1 → 4-class 2

    def test_severe_branch_destroyed(self):
        """Severe branch, destroyed wins (severity_logits[2] >> logits[1])."""
        p_damage = np.array([0.9])
        p_severe = np.array([0.9])
        severity_logits = np.array([[0.0, 1.0, 10.0]])   # destroyed(idx=2) wins
        preds = _predict_cascade(p_damage, p_severe, severity_logits, tau_damage=0.5, tau_severe=0.5)
        assert preds[0] == 3, f"Expected destroyed(3), got {preds[0]}"

    def test_damage_branch_minor(self):
        """p_damage >= tau, p_severe < tau → damage branch; minor(0) wins → 4-class minor(1)."""
        p_damage = np.array([0.9])
        p_severe = np.array([0.1])
        severity_logits = np.array([[10.0, 0.0, 0.0]])   # minor(idx=0) wins
        preds = _predict_cascade(p_damage, p_severe, severity_logits, tau_damage=0.5, tau_severe=0.5)
        assert preds[0] == 1, f"Expected minor(1), got {preds[0]}"

    def test_damage_branch_major(self):
        """Damage branch; major(idx=1) wins → 4-class major(2)."""
        p_damage = np.array([0.9])
        p_severe = np.array([0.1])
        severity_logits = np.array([[0.0, 10.0, 0.0]])   # major(idx=1) wins
        preds = _predict_cascade(p_damage, p_severe, severity_logits, tau_damage=0.5, tau_severe=0.5)
        assert preds[0] == 2, f"Expected major(2), got {preds[0]}"

    def test_output_range_is_0_to_3(self):
        """Output labels must always be in {0, 1, 2, 3}."""
        p_damage, p_severe, severity_logits = self._make_inputs(n=200)
        preds = _predict_cascade(p_damage, p_severe, severity_logits, 0.5, 0.5)
        assert set(preds.tolist()).issubset({0, 1, 2, 3}), \
            f"Invalid labels found: {set(preds.tolist())}"

    def test_dtype_int64(self):
        """Output dtype must be int64."""
        p_damage, p_severe, severity_logits = self._make_inputs()
        preds = _predict_cascade(p_damage, p_severe, severity_logits)
        assert preds.dtype == np.int64, f"dtype={preds.dtype}"


# ---------------------------------------------------------------------------
# G) Cascade threshold fitting — _fit_thresholds_cascade
# ---------------------------------------------------------------------------

class TestFitThresholdsCascade:
    """Tests for _fit_thresholds_cascade."""

    def _make_calib(self, n_no=100, n_minor=30, n_major=20, n_dest=10, seed=42):
        """Synthetic calibration data: high p_damage for damaged, high p_severe for severe."""
        rng = np.random.default_rng(seed)
        labels, p_dmg, p_sev = [], [], []

        # no-damage: low p_damage, low p_severe
        labels.extend([0] * n_no)
        p_dmg.extend(rng.uniform(0.0, 0.3, n_no).tolist())
        p_sev.extend(rng.uniform(0.0, 0.2, n_no).tolist())

        # minor: high p_damage, low p_severe
        labels.extend([1] * n_minor)
        p_dmg.extend(rng.uniform(0.6, 1.0, n_minor).tolist())
        p_sev.extend(rng.uniform(0.0, 0.3, n_minor).tolist())

        # major: high p_damage, high p_severe
        labels.extend([2] * n_major)
        p_dmg.extend(rng.uniform(0.7, 1.0, n_major).tolist())
        p_sev.extend(rng.uniform(0.6, 1.0, n_major).tolist())

        # destroyed: high p_damage, high p_severe
        labels.extend([3] * n_dest)
        p_dmg.extend(rng.uniform(0.8, 1.0, n_dest).tolist())
        p_sev.extend(rng.uniform(0.7, 1.0, n_dest).tolist())

        return (np.array(p_dmg), np.array(p_sev), np.array(labels))

    def test_never_miss_recall_one(self):
        """never_miss=True must achieve recall=1.0 on calibration positives."""
        p_dmg, p_sev, labels = self._make_calib()
        thresholds, summary = _fit_thresholds_cascade(
            p_dmg, p_sev, labels,
            target_recall_damage=1.0, target_recall_severe=1.0, never_miss=True,
        )
        assert summary["tau_damage"]["achieved_recall"] == 1.0, \
            f"damage recall={summary['tau_damage']['achieved_recall']}"
        assert summary["tau_severe"]["achieved_recall"] == 1.0, \
            f"severe recall={summary['tau_severe']['achieved_recall']}"

    def test_thresholds_in_range(self):
        """Fitted thresholds must be in [0, 1]."""
        p_dmg, p_sev, labels = self._make_calib()
        thresholds, _ = _fit_thresholds_cascade(p_dmg, p_sev, labels)
        assert 0.0 <= thresholds["tau_damage"] <= 1.0
        assert 0.0 <= thresholds["tau_severe"] <= 1.0

    def test_returns_correct_keys(self):
        """Return dict must have tau_damage and tau_severe."""
        p_dmg, p_sev, labels = self._make_calib()
        thresholds, summary = _fit_thresholds_cascade(p_dmg, p_sev, labels)
        assert "tau_damage" in thresholds
        assert "tau_severe" in thresholds
        assert "tau_damage" in summary
        assert "tau_severe" in summary

    def test_no_positives_fallback_never_miss(self):
        """When no positives in calib, never_miss=True → tau=0.0."""
        labels = np.zeros(50, dtype=int)  # all no-damage
        p_dmg = np.random.uniform(0, 1, 50)
        p_sev = np.random.uniform(0, 1, 50)
        thresholds, _ = _fit_thresholds_cascade(p_dmg, p_sev, labels, never_miss=True)
        assert thresholds["tau_damage"] == 0.0
        assert thresholds["tau_severe"] == 0.0

    def test_quantile_mode_lower_recall(self):
        """target_recall=0.5 should produce higher tau than target_recall=1.0."""
        p_dmg, p_sev, labels = self._make_calib()
        thr_nm, _ = _fit_thresholds_cascade(
            p_dmg, p_sev, labels, target_recall_damage=1.0, target_recall_severe=1.0,
            never_miss=True)
        thr_q,  _ = _fit_thresholds_cascade(
            p_dmg, p_sev, labels, target_recall_damage=0.5, target_recall_severe=0.5,
            never_miss=False)
        assert thr_q["tau_damage"] >= thr_nm["tau_damage"], \
            "Quantile tau_damage should be >= never-miss tau_damage"
        assert thr_q["tau_severe"] >= thr_nm["tau_severe"], \
            "Quantile tau_severe should be >= never-miss tau_severe"


# ---------------------------------------------------------------------------
# H) Cascade crossfit pooling excludes evaluated fold
# ---------------------------------------------------------------------------

class TestCrossfitPoolCascade:
    """Verify that crossfit pooling logic correctly excludes the evaluated fold."""

    def _make_fold_cascade_data(self, n_folds=5, n_per_fold=40, seed=0):
        """Simulate calib_preds_cascade.npz contents for n_folds folds."""
        rng = np.random.default_rng(seed)
        data = {}
        for f in range(n_folds):
            data[f] = {
                "y_true":          rng.integers(0, 4, n_per_fold),
                "p_damage":        rng.uniform(0, 1, n_per_fold),
                "p_severe":        rng.uniform(0, 1, n_per_fold),
                "severity_logits": rng.standard_normal((n_per_fold, 3)),
            }
        return data

    def test_excludes_eval_fold(self):
        """Pooled data must not contain any samples from the evaluated fold."""
        n_folds, n_per_fold = 5, 40
        data = self._make_fold_cascade_data(n_folds, n_per_fold)
        for eval_fold in range(n_folds):
            pool_labels = np.concatenate([
                data[s]["y_true"] for s in range(n_folds) if s != eval_fold
            ])
            eval_labels = data[eval_fold]["y_true"]
            # Verify pool size = (n_folds - 1) * n_per_fold
            assert len(pool_labels) == (n_folds - 1) * n_per_fold

    def test_threshold_fit_from_pooled_cascade(self):
        """_fit_thresholds_cascade on pooled (non-eval) data must succeed."""
        n_folds, n_per_fold = 5, 40
        data = self._make_fold_cascade_data(n_folds, n_per_fold)
        eval_fold = 2
        pp_dmg = np.concatenate([data[s]["p_damage"] for s in range(n_folds) if s != eval_fold])
        pp_sev = np.concatenate([data[s]["p_severe"] for s in range(n_folds) if s != eval_fold])
        pp_lbl = np.concatenate([data[s]["y_true"]   for s in range(n_folds) if s != eval_fold])
        thresholds, summary = _fit_thresholds_cascade(pp_dmg, pp_sev, pp_lbl, never_miss=True)
        assert "tau_damage" in thresholds
        assert "tau_severe" in thresholds

    def test_pool_excludes_exactly_one_fold(self):
        """Pool contains exactly n_folds-1 sibling folds."""
        n_folds = 5
        data = self._make_fold_cascade_data(n_folds, 20)
        for eval_fold in range(n_folds):
            sibling_folds = [s for s in range(n_folds) if s != eval_fold]
            assert len(sibling_folds) == n_folds - 1
            assert eval_fold not in sibling_folds

