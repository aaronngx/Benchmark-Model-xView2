"""
macro-F1, per-class F1, coverage.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def f1_per_class(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> dict[str, float]:
    """Per-class F1. Classes not in list are ignored for macro."""
    if classes is None:
        classes = DAMAGE_CLASSES
    scores: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        pred_pos = sum(1 for p in y_pred if p == c)
        true_pos = sum(1 for t in y_true if t == c)
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / true_pos if true_pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        scores[c] = {"precision": prec, "recall": rec, "f1": f1}
    return {c: scores[c]["f1"] for c in classes}


def macro_f1(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] | None = None,
) -> float:
    """Macro-averaged F1 over classes."""
    per_class = f1_per_class(y_true, y_pred, classes=classes)
    if not per_class:
        return 0.0
    return sum(per_class.values()) / len(per_class)


def coverage(
    num_matched_gt: int,
    num_total_gt: int,
) -> float:
    """Fraction of GT buildings matched by at least one prediction (0..1)."""
    if num_total_gt == 0:
        return 1.0
    return num_matched_gt / num_total_gt


def compute_metrics(
    rows: list[dict[str, Any]],
    *,
    gt_uid_col: str = "matched_gt_uid",
    gt_damage_col: str = "gt_damage",
    pred_damage_col: str = "pred_damage",
) -> dict[str, Any]:
    """
    From predictions CSV rows (with matched_gt_uid and gt_damage filled for evaluation):
    compute macro_f1, per_class_f1, coverage.
    Only rows with non-null matched_gt_uid and gt_damage are used for F1.
    Coverage = distinct matched GT UIDs / total GT (total GT must be provided or inferred).
    """
    # Rows that have a match and GT damage
    eval_rows = [r for r in rows if r.get(gt_uid_col) and r.get(gt_damage_col)]
    y_true = [r[gt_damage_col] for r in eval_rows]
    y_pred = [r[pred_damage_col] for r in eval_rows]
    per_class = f1_per_class(y_true, y_pred)
    macro = macro_f1(y_true, y_pred)
    # Coverage: need total GT count per tile from elsewhere; here we use num matched / num rows with match as proxy if total not given
    num_matched = len(set(r[gt_uid_col] for r in eval_rows))
    # If we don't have total_gt, we can't compute true coverage; caller can pass it
    return {
        "macro_f1": round(macro, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class.items()},
        "coverage": None,  # set by caller if total_gt known
        "num_matched_gt": num_matched,
        "num_eval_rows": len(eval_rows),
    }
