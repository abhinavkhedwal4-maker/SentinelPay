#!/usr/bin/env python
"""One command to reproduce everything: figures, metrics.json and the Performance Analysis Report.

    python run_pipeline.py                       # synthetic demo data
    python run_pipeline.py --data data/creditcard.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from sentinelpay import viz  # noqa: E402
from sentinelpay.config import FIG_DIR, REPORT_DIR, PipelineConfig  # noqa: E402
from sentinelpay.data import feature_separation  # noqa: E402
from sentinelpay.genai import NvidiaAnalyst  # noqa: E402
from sentinelpay.models import FUSION, IFOREST, LOF  # noqa: E402
from sentinelpay.pipeline import export_metrics, run_pipeline  # noqa: E402
from sentinelpay.report import write_report  # noqa: E402


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to creditcard.csv (default: synthetic demo data)")
    ap.add_argument("--samples", type=int, default=PipelineConfig.n_samples)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ai", action="store_true", help="skip NVIDIA API calls")
    a = ap.parse_args()

    cfg = PipelineConfig(data_path=a.data, n_samples=a.samples, seed=a.seed)
    res = run_pipeline(cfg, verbose=True)
    s, h = res.split, res.headline

    figs = {
        "01_class_imbalance": viz.plot_class_imbalance(res.df),
        "02_amount_time": viz.plot_amount_time(res.df),
        "03_scaling": viz.plot_scaling_comparison(res.df),
        "05_feature_separation": viz.plot_feature_separation(feature_separation(res.df)),
        "06_pr_curves": viz.plot_pr_curves(res.pr, s.y_test),
        "07_contamination_sweep": viz.plot_contamination_sweep(
            res.sweeps, {n: res.recs[n]["max_f1"]["contamination"] for n in res.models}),
        "09_runtime": viz.plot_runtime(res.models),
    }
    best = h["model"]
    pred = res.predictions(best, h["contamination"])
    figs["04_pca_2d"] = viz.plot_pca_scatter(s.X_test, s.y_test, pred, f"{best} · contamination {h['contamination']:.4f}")
    figs["04_pca_3d"] = viz.plot_pca_3d(s.X_test, s.y_test, pred, f"{best} in 3-D PCA space")
    cms = {n: res.evaluate(n, res.recs[n]["max_f1"]["contamination"]) for n in (IFOREST, LOF)}
    figs["08_confusion_matrices"] = viz.plot_confusion_pair(
        {f"{n}\n(contamination {res.recs[n]['max_f1']['contamination']:.4f})": cm for n, cm in cms.items()})
    figs["10_score_distribution"] = viz.plot_score_distribution(
        res.models[best].scores_test, s.y_test,
        res.evaluate(best, h["contamination"])["threshold"], f"{best} – anomaly scores")

    for name, fig in figs.items():
        viz.save(fig, name, FIG_DIR); plt.close(fig)
    viz.plot_banner(Path("docs/assets/banner.png"))

    analyst = NvidiaAnalyst() if not a.no_ai else NvidiaAnalyst(api_key="")
    print("NVIDIA analyst:", "online" if analyst.available else "offline (template text)")
    write_report(res, REPORT_DIR / "Performance_Analysis_Report.md", analyst)
    export_metrics(res, REPORT_DIR / "metrics.json")
    print(f"\nDone. Report -> reports/Performance_Analysis_Report.md   Figures -> {FIG_DIR}")
    print(f"Recommendation: {best}, contamination={h['contamination']:.4f}, recall={h['recall']:.0%}, "
          f"precision={h['precision']:.0%}, FPR={h['fpr']:.2%}")


if __name__ == "__main__":
    main()
