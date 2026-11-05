"""Generate the Performance Analysis Report (Markdown) from a PipelineResult."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .genai import NvidiaAnalyst
from .models import FUSION, IFOREST, LOF
from .pipeline import PipelineResult


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_report(res: PipelineResult, path: Path, analyst: NvidiaAnalyst | None = None,
                 fig_rel: str = "figures") -> str:
    e, h, cfg = res.eda, res.headline, res.cfg
    analyst = analyst or NvidiaAnalyst()
    summary, src = analyst.executive_summary({k: h[k] for k in
                                              ("model", "contamination", "recall", "precision", "fpr", "false_alarms", "n_test")})
    ai_note = "_Written by the NVIDIA-hosted LLM from the metrics below._" if src == "nvidia" else \
        "_Template summary (set `NVIDIA_API_KEY` to have the NVIDIA LLM write this section)._"

    perf = pd.DataFrame([{
        "Model": n, "PR-AUC (AP)": f"{res.pr[n].average_precision:.3f}", "ROC-AUC": f"{res.pr[n].roc_auc:.3f}",
        "Best F1": f"{res.pr[n].best['f1']:.3f}", "Fit (s)": f"{m.fit_time:.1f}", "Score test (s)": f"{m.score_time:.1f}",
        "Latency (ms/txn)": f"{m.latency_ms_per_txn:.3f}"} for n, m in res.models.items()])

    def op_row(name, label, r):
        if r is None:
            return {"Model": name, "Strategy": label, "contamination": "–", "Recall": "–", "Precision": "–",
                    "False alarms": "–", "FPR": "–", "Fraud $ missed": "–"}
        return {"Model": name, "Strategy": label, "contamination": f"{r['contamination']:.4f}",
                "Recall": f"{r['recall']:.1%}", "Precision": f"{r['precision']:.1%}", "False alarms": int(r["fp"]),
                "FPR": f"{r['fpr']:.2%}", "Fraud $ missed": f"${r['fraud_loss']:,.0f}"}

    ops = []
    for n in res.models:
        rc = res.recs[n]
        ops += [op_row(n, "Max F1", rc["max_f1"]), op_row(n, "Min total cost", rc.get("min_cost")),
                op_row(n, f"Recall ≥ {cfg.target_recall:.0%}", rc["target_recall"])]

    best = h["model"]
    sw = res.sweeps[best][["contamination", "precision", "recall", "f1", "fp", "fn", "fpr"]].copy()
    sw.columns = ["contamination", "precision", "recall", "F1", "false alarms", "missed fraud", "FPR"]
    for c in ("precision", "recall", "F1"):
        sw[c] = sw[c].map("{:.3f}".format)
    sw["FPR"] = sw["FPR"].map("{:.2%}".format)
    sw["contamination"] = sw["contamination"].map("{:.4f}".format)

    cost_op = res.recs[best].get("min_cost") or res.recs[best]["max_f1"]
    tgt = res.recs[best]["target_recall"]
    tgt_line = (f"* **Recall-first (compliance) option:** contamination **{tgt['contamination']:.3f}** reaches "
                f"**{tgt['recall']:.0%} recall** at the price of {int(tgt['fp'])} false alarms "
                f"({tgt['precision']:.0%} precision).") if tgt else "* No tested contamination reaches the target recall."

    md = f"""# Performance Analysis Report — SentinelPay

> **Data source:** {res.source}. {"These numbers come from the synthetic ULB-style demo set; re-run on the real Kaggle `creditcard.csv` to obtain the final figures." if "Synthetic" in res.source else ""}
> **Hold-out set:** {h['n_test']:,} transactions, {h['n_fraud_test']} of them fraud.

## 1. Executive summary

{summary}

{ai_note}

**Recommendation.** Deploy **{best}** with a **contamination threshold of {h['contamination']:.4f}**, which yields a
**recall of {h['recall']:.0%}**, **precision of {h['precision']:.0%}** (F1 {h['f1']:.2f}) and a false-positive rate of
**{h['fpr']:.2%}** ({h['false_alarms']} false alarms on {h['n_test']:,} transactions).

## 2. Data & exploratory findings

| Metric | Value |
|---|---|
| Transactions | {e['n_transactions']:,} |
| Fraud cases | {e['n_fraud']:,} ({e['fraud_rate']:.3%}) |
| Imbalance ratio | 1 : {e['imbalance_ratio']:,.0f} |
| Accuracy of a model that always says "legitimate" | **{e['naive_accuracy']:.2%}** (the metric trap) |
| Median / mean Amount | ${e['amount_median']:,.2f} / ${e['amount_mean']:,.2f} |
| Amount skewness | {e['amount_skew']:.1f} |

![Class imbalance]({fig_rel}/01_class_imbalance.png)
![Amount and time]({fig_rel}/02_amount_time.png)

**Preprocessing.** `Amount` (and `Time`) are scaled with `RobustScaler` (median / IQR). The scaler is fitted on normal
training rows only, so no information leaks from the test set or from fraud. 

![Scaler comparison]({fig_rel}/03_scaling.png)

## 3. Methodology

* **Training strategy:** detectors are trained on *normal* transactions only. A further 20 % of normal training rows are
  held out as a *calibration set*; the alert threshold is the `(1 − contamination)` quantile of calibration scores, so it
  needs **no fraud labels** and the false-positive rate lands close to the contamination value.
* **Isolation Forest:** {cfg.if_n_estimators} trees, `max_samples={cfg.if_max_samples}` – anomalies need fewer random splits to isolate.
* **Local Outlier Factor:** `n_neighbors={cfg.lof_n_neighbors}`, `novelty=True`, trained on ≤ {cfg.lof_max_train:,} rows.
* **Fusion (extra):** both scores are mapped to calibrated tail probabilities (`-log10 P(normal ≥ score)`) and averaged.
* **Model selection:** highest PR-AUC; if models are within 0.03 PR-AUC the lower-latency one wins (real-time constraint).
* **Metrics:** Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrices. Accuracy is deliberately not used.

## 4. Model comparison

{_md_table(perf)}

![PR curves]({fig_rel}/06_pr_curves.png)
![Runtime]({fig_rel}/09_runtime.png)

**Reading the results**

* **Isolation Forest vs LOF:** {"Isolation Forest" if res.pr[IFOREST].average_precision >= res.pr[LOF].average_precision else "LOF"} has the higher
  PR-AUC ({max(res.pr[IFOREST].average_precision, res.pr[LOF].average_precision):.3f} vs {min(res.pr[IFOREST].average_precision, res.pr[LOF].average_precision):.3f}).
  Isolation Forest also scores the test set {res.models[LOF].score_time / max(res.models[IFOREST].score_time, 1e-9):.0f}× faster than LOF here, and LOF's
  neighbour search scales poorly with data size, which is why it is trained on a capped sample.
* **Fusion** reaches PR-AUC {res.pr[FUSION].average_precision:.3f}: combining a tree-based and a density-based view helps
  when each model misses different fraud, at the cost of running both.
* ROC-AUC is high for every model (> {min(p.roc_auc for p in res.pr.values()):.2f}) *because of* the imbalance – PR-AUC is the honest metric.

## 5. Threshold tuning

**Scenario A – higher threshold (small contamination):** few false alarms, but fraud slips through.
**Scenario B – lower threshold (large contamination):** most fraud is caught, but many legitimate customers are challenged.

![Contamination sweep]({fig_rel}/07_contamination_sweep.png)

### Three defensible operating points

{_md_table(pd.DataFrame(ops))}

### Sweep for {best}

{_md_table(sw)}

## 6. Final recommendation

* **Model:** {best}.
* **Balanced option (max F1):** contamination **{h['contamination']:.4f}** → recall **{h['recall']:.0%}**, precision **{h['precision']:.0%}**, FPR **{h['fpr']:.2%}**.
* **Cost-optimal option ($):** contamination **{cost_op['contamination']:.4f}** → total cost ${cost_op['total_cost']:,.0f} (missed fraud + ${cfg.review_cost:.0f} per reviewed alert).
{tgt_line}

> *Example sentence for stakeholders:* "We recommend a contamination threshold of {h["contamination"]:g} using {best}, which yields a
> recall of {h['recall']:.0%} with a false-positive rate of {h['fpr']:.2%}."

![Confusion matrices]({fig_rel}/08_confusion_matrices.png)
![PCA scatter]({fig_rel}/04_pca_2d.png)

## 7. Limitations & next steps

1. The threshold is tuned on the same hold-out set that is reported; in production use a separate validation window.
2. Fraud evolves – re-calibrate on fresh normal traffic on a schedule and monitor score drift.
3. Features are PCA-anonymised, so explanations are statistical (which components deviate), not business-level.
4. Next: add a supervised second stage for analyst-confirmed cases, streaming scoring (Kafka / Flink) and model-drift alerts.

---
*Generated by `python run_pipeline.py` · SentinelPay v1.0*
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return md
