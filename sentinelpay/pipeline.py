"""End-to-end orchestration: load -> preprocess -> fit detectors -> evaluate -> recommend."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIG_DIR, PipelineConfig
from .data import SplitData, eda_summary, load_data, preprocess
from .evaluation import PRAnalysis, contamination_sweep, metrics_at_threshold, pr_analysis, recommend
from .models import (FUSION, IFOREST, LOF, ModelResult, run_fusion, run_isolation_forest, run_lof,
                     threshold_from_contamination)


AP_TOLERANCE = 0.03   # PR-AUC points within which latency breaks the tie


@dataclass
class PipelineResult:
    cfg: PipelineConfig
    df: pd.DataFrame
    source: str
    eda: dict
    split: SplitData
    models: dict[str, ModelResult]
    pr: dict[str, PRAnalysis]
    sweeps: dict[str, pd.DataFrame]
    recs: dict[str, dict]
    headline: dict = field(default_factory=dict)

    def evaluate(self, model: str, contamination: float) -> dict:
        """Metrics for any model at any contamination – no refit needed (used by the dashboard slider)."""
        m = self.models[model]
        thr = threshold_from_contamination(m.scores_calib, contamination)
        return metrics_at_threshold(self.split.y_test, m.scores_test, thr,
                                    self.split.raw_test.Amount.to_numpy(), self.cfg.review_cost)

    def predictions(self, model: str, contamination: float) -> np.ndarray:
        thr = threshold_from_contamination(self.models[model].scores_calib, contamination)
        return (self.models[model].scores_test >= thr).astype(int)


def run_pipeline(cfg: PipelineConfig | None = None, verbose: bool = False) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    log = print if verbose else (lambda *a, **k: None)

    df, source = load_data(cfg.data_path, cfg.n_samples, cfg.fraud_rate, cfg.seed)
    log(f"[1/4] loaded {len(df):,} rows  ({source})")
    split = preprocess(df, cfg)
    log(f"[2/4] split: fit={len(split.X_fit):,} calib={len(split.X_calib):,} test={len(split.X_test):,} "
        f"(fraud in test: {int(split.y_test.sum())})")

    iforest = run_isolation_forest(split, cfg); log(f"[3/4] {IFOREST}: fit {iforest.fit_time:.1f}s")
    lof = run_lof(split, cfg); log(f"      {LOF}: fit {lof.fit_time:.1f}s, score {lof.score_time:.1f}s")
    models = {IFOREST: iforest, LOF: lof, FUSION: run_fusion(iforest, lof)}

    y, amounts = split.y_test, split.raw_test.Amount.to_numpy()
    pr = {n: pr_analysis(y, m.scores_test) for n, m in models.items()}
    sweeps = {n: contamination_sweep(m, y, amounts, review_cost=cfg.review_cost) for n, m in models.items()}
    recs = {n: recommend(sw, cfg.target_recall) for n, sw in sweeps.items()}

    # Real-time constraint: among models within AP_TOLERANCE of the best PR-AUC, pick the lowest-latency one.
    top_ap = max(p.average_precision for p in pr.values())
    contenders = [n for n in models if pr[n].average_precision >= top_ap - AP_TOLERANCE]
    best_model = min(contenders, key=lambda n: models[n].latency_ms_per_txn)
    op = recs[best_model]["max_f1"]
    headline = {"model": best_model, "contamination": float(op["contamination"]), "recall": float(op["recall"]),
                "precision": float(op["precision"]), "f1": float(op["f1"]), "fpr": float(op["fpr"]),
                "false_alarms": int(op["fp"]), "n_test": int(len(y)), "n_fraud_test": int(y.sum())}
    log(f"[4/4] selected model: {best_model}  | recommended contamination = {headline['contamination']}")
    return PipelineResult(cfg, df, source, eda_summary(df), split, models, pr, sweeps, recs, headline)


def export_metrics(res: PipelineResult, path: Path) -> None:
    """Write a compact JSON of everything the report and README quote."""
    out = {"source": res.source, "eda": res.eda, "headline": res.headline,
           "models": {n: {"average_precision": res.pr[n].average_precision, "roc_auc": res.pr[n].roc_auc,
                          "best_f1_point": res.pr[n].best, "fit_time_s": m.fit_time, "score_time_s": m.score_time,
                          "latency_ms_per_txn": m.latency_ms_per_txn, "params": {k: v for k, v in m.params.items()},
                          "recommendations": res.recs[n]} for n, m in res.models.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
