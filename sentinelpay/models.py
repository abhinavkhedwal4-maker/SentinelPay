"""Unsupervised detectors: Isolation Forest, Local Outlier Factor and a rank-fusion ensemble.

Both detectors are trained on **normal transactions only**. Every model exposes the same
convention: a *higher score means more anomalous*.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .config import PipelineConfig
from .data import SplitData

warnings.filterwarnings("ignore", message="X does not have valid feature names")

IFOREST, LOF, FUSION = "Isolation Forest", "Local Outlier Factor", "Fusion (IF + LOF)"


@dataclass
class ModelResult:
    name: str
    scores_calib: np.ndarray      # scores of held-out *normal* rows -> defines "what normal looks like"
    scores_test: np.ndarray       # scores of the hold-out test set
    fit_time: float
    score_time: float
    params: dict = field(default_factory=dict)
    estimator: object | None = None

    @property
    def total_time(self) -> float:
        return self.fit_time + self.score_time

    @property
    def latency_ms_per_txn(self) -> float:
        return 1000 * self.score_time / max(len(self.scores_test), 1)


def run_isolation_forest(split: SplitData, cfg: PipelineConfig) -> ModelResult:
    """Anomalies are *few and different* -> they are isolated by fewer random splits (short paths)."""
    model = IsolationForest(n_estimators=cfg.if_n_estimators, max_samples=cfg.if_max_samples,
                            contamination="auto", n_jobs=-1, random_state=cfg.seed)
    t0 = time.perf_counter(); model.fit(split.X_fit); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_test = -model.score_samples(split.X_test)                 # sklearn: lower = more abnormal -> flip
    score_t = time.perf_counter() - t0
    s_cal = -model.score_samples(split.X_calib)
    return ModelResult(IFOREST, s_cal, s_test, fit_t, score_t,
                       {"n_estimators": cfg.if_n_estimators, "max_samples": cfg.if_max_samples}, model)


def run_lof(split: SplitData, cfg: PipelineConfig) -> ModelResult:
    """Points in regions that are much sparser than their neighbours' regions get a high LOF.

    ``novelty=True`` lets us fit on normal data and score unseen transactions. Training rows are
    capped because neighbour search grows quickly with data size.
    """
    X_fit = split.X_fit
    if len(X_fit) > cfg.lof_max_train:
        X_fit = X_fit.sample(cfg.lof_max_train, random_state=cfg.seed)
    model = LocalOutlierFactor(n_neighbors=cfg.lof_n_neighbors, novelty=True, n_jobs=-1)
    t0 = time.perf_counter(); model.fit(X_fit); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_test = -model.score_samples(split.X_test)
    score_t = time.perf_counter() - t0
    s_cal = -model.score_samples(split.X_calib)
    return ModelResult(LOF, s_cal, s_test, fit_t, score_t,
                       {"n_neighbors": cfg.lof_n_neighbors, "train_rows": len(X_fit)}, model)


def tail_score(calib: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map raw scores to ``-log10(P(normal row scores at least this high))`` using the calibration set.

    Puts models with different score scales on one comparable axis (needed for fusion).
    """
    ref = np.sort(calib)
    ecdf = np.searchsorted(ref, scores, side="right") / (len(ref) + 1)
    return -np.log10(1 - ecdf + 1 / (len(ref) + 1))


def run_fusion(a: ModelResult, b: ModelResult) -> ModelResult:
    """Average of the two models' calibrated tail scores – tree-based + density-based views."""
    s_test = (tail_score(a.scores_calib, a.scores_test) + tail_score(b.scores_calib, b.scores_test)) / 2
    s_cal = (tail_score(a.scores_calib, a.scores_calib) + tail_score(b.scores_calib, b.scores_calib)) / 2
    return ModelResult(FUSION, s_cal, s_test, a.fit_time + b.fit_time, a.score_time + b.score_time,
                       {"members": [a.name, b.name]})


def threshold_from_contamination(scores_calib: np.ndarray, contamination: float) -> float:
    """The deployable meaning of ``contamination``: flag the top ``c`` share of *normal-looking* traffic.

    Because the threshold comes from held-out normal rows it needs **no fraud labels** and the
    false-positive rate on legitimate traffic lands close to ``c``.
    """
    return float(np.quantile(scores_calib, 1 - contamination))
