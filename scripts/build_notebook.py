"""Builds notebooks/SentinelPay_Analysis.ipynb *with executed outputs embedded*.

nbformat/jupyter are not required: cells are run with exec(), figures are captured as PNG and
written into standard nbformat-4 JSON. Open the result in Jupyter/VS Code/GitHub as usual.
    python scripts/build_notebook.py
"""
import ast, base64, contextlib, io, json, sys, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

MD, CODE = "markdown", "code"
cells = []
def md(t): cells.append((MD, t.strip("\n")))
def code(t): cells.append((CODE, t.strip("\n")))

# =====================================================================================
md("""
# 🛡️ SentinelPay — Real-Time Anomaly Detection in Financial Transactions
**Case study:** SecurePay FinTech · **Approach:** unsupervised (Isolation Forest + Local Outlier Factor) · **Stack:** pandas, scikit-learn, matplotlib

> Fraud is < 0.2 % of traffic, so a classifier that always answers *"legitimate"* is 99.8 % accurate and completely useless.
> Instead of learning what fraud looks like (it keeps changing), we learn what **normal** looks like and flag deviations.

**Notebook map** — Phase 1 data & EDA → Phase 2 models → Phase 3 evaluation & threshold tuning → Recommendation → GenAI analyst.
""")
code("""
import sys, time, warnings
sys.path.insert(0, "..")                      # make the `sentinelpay` package importable
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import average_precision_score

from sentinelpay import viz
from sentinelpay.config import PipelineConfig
from sentinelpay.data import load_data, eda_summary, preprocess, feature_separation
from sentinelpay.pipeline import run_pipeline
pd.set_option("display.precision", 3)
cfg = PipelineConfig()
""")

md("""
## Phase 1 — Data analysis & preprocessing
### 1.1 Load
`load_data` reads `data/creditcard.csv` (the public ULB / Kaggle dataset) when present; otherwise it generates a
**synthetic ULB-style dataset** with the same schema (`Time`, `V1…V28`, `Amount`, `Class`) so the project runs anywhere.
""")
code("""
df, source = load_data(cfg.data_path, cfg.n_samples, cfg.fraud_rate, cfg.seed)
print("Source:", source, "| shape:", df.shape)
df.head()
""")
code("""
eda = eda_summary(df)
pd.Series({k: (f"{v:,.4f}" if isinstance(v, float) else f"{v:,}") for k, v in eda.items()}, name="value").to_frame()
""")
md("### 1.2 The class imbalance — and the accuracy trap")
code("""
viz.plot_class_imbalance(df); plt.show()
print(f"A model that predicts 'legitimate' for everything scores {eda['naive_accuracy']:.2%} accuracy and catches 0 fraud.")
""")
md("### 1.3 Distribution of `Amount` and `Time`")
code("""
viz.plot_amount_time(df); plt.show()
""")
md("""
**Observations.** Amounts are extremely right-skewed (a handful of very large purchases), time follows a day/night cycle,
and fraud shows two behaviours: tiny "card-testing" amounts and larger purchases, with a skew towards night hours.
""")
md("""
### 1.4 Advanced scaling — `RobustScaler` instead of `StandardScaler`
Distance-based models (LOF) are wrecked by outliers. `StandardScaler` uses mean/σ, which the extreme purchases drag around;
`RobustScaler` uses **median and IQR**: `x' = (x − median) / IQR`.
""")
code("""
viz.plot_scaling_comparison(df); plt.show()
""")
code("""
# The mechanics, spelled out on Amount:
q1, med, q3 = df.Amount.quantile([.25, .5, .75])
manual = (df.Amount - med) / (q3 - q1)
sk = RobustScaler().fit_transform(df[["Amount"]]).ravel()
print(f"median={med:.2f}  IQR={q3 - q1:.2f}  | manual == sklearn: {np.allclose(manual, sk)}")
""")
md("""
### 1.5 Leakage-free split
* stratified train / test split → **only normal rows** are used for training (unsupervised strategy)
* normal training rows are split again into a **fit set** and a **calibration set** (used to set the alert threshold – no fraud labels needed)
* the `RobustScaler` is fitted on the fit set only
""")
code("""
split = preprocess(df, cfg)
print(f"fit (normal only): {len(split.X_fit):,} | calibration (normal only): {len(split.X_calib):,} | "
      f"test: {len(split.X_test):,} of which fraud = {int(split.y_test.sum())}")
split.X_fit[["Time", "Amount", "V1", "V2"]].describe().T
""")
md("### 1.6 Which PCA features carry the fraud signal?")
code("""
viz.plot_feature_separation(feature_separation(df, top=12)); plt.show()
""")

# =====================================================================================
md("""
## Phase 2 — Unsupervised model building
### 2.1 Isolation Forest — *"few and different"*
Random trees isolate a point by repeatedly choosing a random feature and split value. Anomalies sit alone in sparse space, so they
need **fewer splits** (short path length) → higher anomaly score. We fit on **normal data only**.
""")
code("""
iforest = IsolationForest(n_estimators=200, max_samples=512, contamination="auto", n_jobs=-1, random_state=42)
t0 = time.perf_counter(); iforest.fit(split.X_fit); t_fit = time.perf_counter() - t0
if_scores = -iforest.score_samples(split.X_test)            # flip sign: higher = more anomalous
print(f"fit in {t_fit:.2f}s | PR-AUC = {average_precision_score(split.y_test, if_scores):.3f}")
""")
md("**Hyper-parameter experiment — `n_estimators`.** More trees stabilise the score; returns diminish quickly.")
code("""
rows = []
for n in [25, 50, 100, 200, 400]:
    m = IsolationForest(n_estimators=n, max_samples=512, n_jobs=-1, random_state=42)
    t0 = time.perf_counter(); m.fit(split.X_fit); dt = time.perf_counter() - t0
    rows.append({"n_estimators": n, "fit_s": dt, "PR-AUC": average_precision_score(split.y_test, -m.score_samples(split.X_test))})
pd.DataFrame(rows)
""")
md("""
**Hyper-parameter experiment — `contamination`.** Contamination is the expected share of flagged traffic. Because we train on
normal data, we turn it into a threshold by taking the `(1 − c)` quantile of **calibration scores**: the false-positive rate then lands close to `c`.
""")
code("""
from sentinelpay.models import threshold_from_contamination
from sentinelpay.evaluation import metrics_at_threshold
cal_scores = -iforest.score_samples(split.X_calib)
rows = []
for c in [0.001, 0.005, 0.01, 0.02, 0.05]:
    thr = threshold_from_contamination(cal_scores, c)
    m = metrics_at_threshold(split.y_test, if_scores, thr)
    rows.append({"contamination": c, "recall": m["recall"], "precision": m["precision"], "f1": m["f1"], "false alarms": m["fp"], "FPR": m["fpr"]})
pd.DataFrame(rows)
""")
md("""
### 2.2 Local Outlier Factor — *density vs. neighbours*
LOF compares the local density around a point to the density around its `k` nearest neighbours. A point in a much sparser region than
its neighbours gets a score ≫ 1. `novelty=True` lets us **fit on normal data and score new transactions**.
LOF is neighbour-search based, so training rows are capped (`lof_max_train`).
""")
code("""
fit_sample = split.X_fit.sample(cfg.lof_max_train, random_state=42)
lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
t0 = time.perf_counter(); lof.fit(fit_sample); l_fit = time.perf_counter() - t0
t0 = time.perf_counter(); lof_scores = -lof.score_samples(split.X_test); l_score = time.perf_counter() - t0
print(f"fit {l_fit:.2f}s | score {l_score:.2f}s | PR-AUC = {average_precision_score(split.y_test, lof_scores):.3f}")
""")
code("""
rows = []
for k in [10, 20, 35, 60, 100]:
    m = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(fit_sample)
    rows.append({"n_neighbors": k, "PR-AUC": average_precision_score(split.y_test, -m.score_samples(split.X_test))})
pd.DataFrame(rows)
""")
md("### 2.3 Side by side (plus a rank-fusion ensemble) via the reusable pipeline")
code("""
res = run_pipeline(cfg)                                    # same models as above + calibration + fusion
cmp = pd.DataFrame({n: {"PR-AUC": res.pr[n].average_precision, "ROC-AUC": res.pr[n].roc_auc, "Fit (s)": m.fit_time,
                        "Score test (s)": m.score_time, "Latency (ms/txn)": m.latency_ms_per_txn}
                    for n, m in res.models.items()}).T
cmp
""")
code("""
viz.plot_runtime(res.models); plt.show()
print("LOF scoring is much slower than Isolation Forest, and the gap widens as the data grows.")
""")

# =====================================================================================
md("""
## Phase 3 — Evaluation & threshold tuning
Accuracy is banned. We use **precision** (share of alerts that are fraud) and **recall** (share of fraud caught).
* **Scenario A – high threshold:** high precision, low recall → few false alarms, missed fraud.
* **Scenario B – low threshold:** low precision, high recall → catches fraud, challenges many legitimate customers.
""")
code("""
y = split.y_test
viz.plot_pr_curves(res.pr, y); plt.show()
pd.DataFrame({n: p.best for n, p in res.pr.items()}).T.rename(columns={"threshold": "score threshold"})
""")
md("### Contamination sweep — the business trade-off")
code("""
viz.plot_contamination_sweep(res.sweeps, {n: res.recs[n]["max_f1"]["contamination"] for n in res.models}); plt.show()
""")
code("""
best = res.headline["model"]
res.sweeps[best][["contamination", "precision", "recall", "f1", "fp", "fn", "fpr", "total_cost"]].round(4)
""")
md("### Confusion matrices — both models at their max-F1 operating point")
code("""
from sentinelpay.models import IFOREST, LOF
cms = {f"{n}\\n(contamination {res.recs[n]['max_f1']['contamination']:.4f})": res.evaluate(n, res.recs[n]['max_f1']['contamination'])
       for n in (IFOREST, LOF)}
viz.plot_confusion_pair(cms); plt.show()
""")
md("### Where do the detected anomalies sit? (PCA projection)")
code("""
h = res.headline
pred = res.predictions(best, h["contamination"])
viz.plot_pca_scatter(split.X_test, y, pred, f"{best} · contamination {h['contamination']:.4f}"); plt.show()
""")
code("""
viz.plot_pca_3d(split.X_test, y, pred, f"{best} in 3-D PCA space"); plt.show()
""")
code("""
viz.plot_score_distribution(res.models[best].scores_test, y, res.evaluate(best, h["contamination"])["threshold"], f"{best} – anomaly scores"); plt.show()
""")

md("## Recommendation")
code("""
rec = res.recs[best]
for label, key in [("Balanced (max F1)", "max_f1"), ("Cost-optimal", "min_cost"), (f"Recall ≥ {cfg.target_recall:.0%}", "target_recall")]:
    r = rec.get(key)
    if r: print(f"{label:<22} contamination={r['contamination']:.4f}  recall={r['recall']:.0%}  precision={r['precision']:.0%}  "
                f"false alarms={int(r['fp'])}  FPR={r['fpr']:.2%}")
print(f"\\nRecommendation: use {best} with contamination = {h['contamination']:.4f}.")
""")

md("""
## GenAI analyst (NVIDIA API)
`sentinelpay.genai.NvidiaAnalyst` calls NVIDIA's OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`) to turn the top
anomalous features of an alert into a plain-English explanation and recommended action. It only receives anonymised PCA features and
scores. **Without `NVIDIA_API_KEY` it falls back to a rule-based explanation**, so this cell always runs.
""")
code("""
from dotenv import load_dotenv; load_dotenv("../.env")
from sentinelpay.genai import NvidiaAnalyst, feature_deviations, risk_band
analyst = NvidiaAnalyst()
mm = res.models[best]; i = int(np.argmax(mm.scores_test))
pct = float((mm.scores_calib < mm.scores_test[i]).mean() * 100)
case = {"model": best, "anomaly_score": round(float(mm.scores_test[i]), 3), "score_percentile": round(pct, 3), "risk_band": risk_band(pct),
        "amount": float(split.raw_test.Amount[i]), "hour_of_day": round(float(split.raw_test.Time[i] % 86400 / 3600), 1),
        "deviations": feature_deviations(split.X_test.iloc[i], split.X_fit)}
text, src = analyst.explain_transaction(case)
print(f"[source: {src}]\\n"); print(text)
""")
md("""
---
**Next steps:** run on the real Kaggle dataset (`data/creditcard.csv`), validate thresholds on a time-based hold-out, add drift monitoring,
and serve scores via the Streamlit dashboard (`streamlit run app/dashboard.py`).
""")

# =====================================================================================
def run_all():
    ns = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, src in cells:
        if kind == MD:
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
            continue
        count += 1
        outs = []
        ns["display"] = lambda obj, _o=outs: _o.append({"output_type": "display_data", "metadata": {},
                                                          "data": {"text/plain": [repr(obj)]}})
        buf = io.StringIO()
        tree = ast.parse(src)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<cell>", "exec"), ns)
                val = eval(compile(last, "<cell>", "eval"), ns) if last else None
        except Exception:
            print(f"--- cell {count} failed ---\n{src}\n"); traceback.print_exc(); sys.exit(1)
        if buf.getvalue():
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
        for num in plt.get_fignums():
            f = plt.figure(num); b = io.BytesIO(); f.savefig(b, format="png", bbox_inches="tight", dpi=110)
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        if val is not None:
            data = {"text/plain": repr(val).splitlines(True)}
            if hasattr(val, "_repr_html_"):
                data["text/html"] = val._repr_html_().splitlines(True)
            outs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outs,
                         "source": src.splitlines(True)})
        print(f"cell {count} ok")
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = ROOT / "notebooks" / "SentinelPay_Analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    os.chdir(ROOT / "notebooks")        # cells use paths relative to the notebooks/ folder, like Jupyter does
    run_all()
