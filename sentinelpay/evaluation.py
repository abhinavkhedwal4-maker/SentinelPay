"""Metrics beyond accuracy: precision, recall, PR curves, contamination sweeps, business cost."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import CONTAMINATION_GRID
from .models import ModelResult, threshold_from_contamination


@dataclass
class PRAnalysis:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    average_precision: float
    roc_auc: float
    best: dict               # operating point with the maximum F1


def pr_analysis(y: np.ndarray, scores: np.ndarray) -> PRAnalysis:
    p, r, thr = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    best = {"threshold": float(thr[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
    return PRAnalysis(p, r, thr, float(average_precision_score(y, scores)), float(roc_auc_score(y, scores)), best)


def metrics_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float,
                         amounts: np.ndarray | None = None, review_cost: float = 2.0) -> dict:
    pred = scores >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "flagged_share": float(pred.mean()),
    }
    if amounts is not None:
        missed = float(amounts[(pred == 0) & (y == 1)].sum())
        caught = float(amounts[(pred == 1) & (y == 1)].sum())
        out.update(fraud_loss=missed, fraud_prevented=caught, review_cost=fp * review_cost,
                   total_cost=missed + (fp + tp) * review_cost)
    return out


def contamination_sweep(model: ModelResult, y: np.ndarray, amounts: np.ndarray | None = None,
                        grid: list[float] | None = None, review_cost: float = 2.0) -> pd.DataFrame:
    """Evaluate every candidate ``contamination`` value on the hold-out set."""
    rows = []
    for c in (grid or CONTAMINATION_GRID):
        thr = threshold_from_contamination(model.scores_calib, c)
        rows.append({"contamination": c, **metrics_at_threshold(y, model.scores_test, thr, amounts, review_cost)})
    return pd.DataFrame(rows)


def recommend(sweep: pd.DataFrame, target_recall: float = 0.80) -> dict:
    """Three defensible operating points: best F1, cheapest in dollars, smallest alert load at target recall."""
    out = {"max_f1": sweep.loc[sweep.f1.idxmax()].to_dict()}
    if "total_cost" in sweep:
        out["min_cost"] = sweep.loc[sweep.total_cost.idxmin()].to_dict()
    ok = sweep[sweep.recall >= target_recall]
    out["target_recall"] = ok.iloc[0].to_dict() if len(ok) else None   # grid is ascending -> lowest alert load
    return out
