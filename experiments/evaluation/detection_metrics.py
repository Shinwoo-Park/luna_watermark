from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


def _safe_unique(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) >= 2


def _interpolate_tpr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, target_fpr: float,
) -> float:

    if len(fpr) == 0:
        return 0.0

    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    if idx < 0:
        return 0.0
    if idx >= len(fpr) - 1 or fpr[idx] == target_fpr:
        return float(tpr[idx])

    if fpr[idx] < target_fpr < fpr[idx + 1]:
        slope = (tpr[idx + 1] - tpr[idx]) / max(fpr[idx + 1] - fpr[idx], 1e-12)
        return float(tpr[idx] + slope * (target_fpr - fpr[idx]))
    return float(tpr[idx])


def _compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> Dict:

    fnr = 1.0 - tpr
    diff = fpr - fnr

    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:

        idx = int(np.argmin(np.abs(diff)))
        return {"eer": float((fpr[idx] + fnr[idx]) / 2), "eer_idx": idx}
    idx = int(sign_changes[0])

    if idx + 1 < len(diff) and diff[idx + 1] != diff[idx]:
        t = -diff[idx] / (diff[idx + 1] - diff[idx])
        eer = float(fpr[idx] + t * (fpr[idx + 1] - fpr[idx]))
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2)
    return {"eer": eer, "eer_idx": idx}


def compute_detection_metrics(
    positive_scores: List[float],
    negative_scores: List[float],
) -> Dict:

    from sklearn.metrics import roc_curve, roc_auc_score

    n_pos = len(positive_scores)
    n_neg = len(negative_scores)

    empty_metrics = {
        "tpr_at_1_fpr": 0.0,
        "tpr_at_5_fpr": 0.0,
        "tpr_at_10_fpr": 0.0,
        "auroc": float('nan'),
        "eer": float('nan'),
        "best_f1": 0.0,
        "best_f1_threshold": None,
        "best_f1_tpr": 0.0,
        "best_f1_fpr": 0.0,
        "best_f1_precision": 0.0,
        "best_f1_recall": 0.0,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    if n_pos == 0 or n_neg == 0:
        return _augment_with_score_stats(empty_metrics, positive_scores, negative_scores)

    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_score = np.array(list(positive_scores) + list(negative_scores), dtype=np.float64)

    # Filter NaN / inf
    mask = np.isfinite(y_score)
    if not mask.all():
        y_true = y_true[mask]
        y_score = y_score[mask]
    if not _safe_unique(y_true):
        return _augment_with_score_stats(empty_metrics, positive_scores, negative_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float('nan')

    tpr_at_1 = _interpolate_tpr_at_fpr(fpr, tpr, 0.01)
    tpr_at_5 = _interpolate_tpr_at_fpr(fpr, tpr, 0.05)
    tpr_at_10 = _interpolate_tpr_at_fpr(fpr, tpr, 0.10)

    eer_info = _compute_eer(fpr, tpr)

    best = {"f1": -1.0, "thr": None, "tpr": 0.0, "fpr": 0.0, "prec": 0.0, "rec": 0.0}
    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())

    for thr in thresholds[1:]:
        pred = (y_score >= thr)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / P if P > 0 else 0.0
        f1   = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best["f1"]:
            best.update({
                "f1": f1, "thr": float(thr),
                "tpr": rec, "fpr": (fp / N if N > 0 else 0.0),
                "prec": prec, "rec": rec,
            })

    metrics = {
        "tpr_at_1_fpr": tpr_at_1,
        "tpr_at_5_fpr": tpr_at_5,
        "tpr_at_10_fpr": tpr_at_10,
        "auroc": auroc,
        "eer": eer_info["eer"],
        "best_f1": best["f1"] if best["f1"] >= 0 else 0.0,
        "best_f1_threshold": best["thr"],
        "best_f1_tpr": best["tpr"],
        "best_f1_fpr": best["fpr"],
        "best_f1_precision": best["prec"],
        "best_f1_recall": best["rec"],
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
    return _augment_with_score_stats(metrics, positive_scores, negative_scores)


def _augment_with_score_stats(
    metrics: Dict,
    positive_scores: List[float],
    negative_scores: List[float],
) -> Dict:

    def _stats(scores, prefix):
        finite = np.array([s for s in scores if np.isfinite(s)], dtype=np.float64)
        if len(finite) == 0:
            return {
                f"{prefix}_mean": float('nan'),
                f"{prefix}_std": float('nan'),
                f"{prefix}_median": float('nan'),
                f"{prefix}_p10": float('nan'),
                f"{prefix}_p25": float('nan'),
                f"{prefix}_p75": float('nan'),
                f"{prefix}_p90": float('nan'),
            }
        return {
            f"{prefix}_mean": float(finite.mean()),
            f"{prefix}_std": float(finite.std()),
            f"{prefix}_median": float(np.median(finite)),
            f"{prefix}_p10": float(np.percentile(finite, 10)),
            f"{prefix}_p25": float(np.percentile(finite, 25)),
            f"{prefix}_p75": float(np.percentile(finite, 75)),
            f"{prefix}_p90": float(np.percentile(finite, 90)),
        }

    metrics.update(_stats(positive_scores, "pos"))
    metrics.update(_stats(negative_scores, "neg"))
    return metrics
