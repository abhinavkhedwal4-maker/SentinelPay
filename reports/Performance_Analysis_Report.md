# Performance Analysis Report — SentinelPay

> **Data source:** Synthetic demo data (ULB-style). These numbers come from the synthetic ULB-style demo set; re-run on the real Kaggle `creditcard.csv` to obtain the final figures.
> **Hold-out set:** 30,000 transactions, 90 of them fraud.

## 1. Executive summary

We recommend Isolation Forest with a contamination of 0.0005. On the hold-out set it catches 52% of fraud at 71% precision, producing 19 false alarms among 30,000 transactions (false-positive rate 0.06%). Main risk: fraud patterns drift, so thresholds should be re-calibrated on fresh normal traffic regularly.

_Template summary (set `NVIDIA_API_KEY` to have the NVIDIA LLM write this section)._

**Recommendation.** Deploy **Isolation Forest** with a **contamination threshold of 0.0005**, which yields a
**recall of 52%**, **precision of 71%** (F1 0.60) and a false-positive rate of
**0.06%** (19 false alarms on 30,000 transactions).

## 2. Data & exploratory findings

| Metric | Value |
|---|---|
| Transactions | 100,000 |
| Fraud cases | 300 (0.300%) |
| Imbalance ratio | 1 : 332 |
| Accuracy of a model that always says "legitimate" | **99.70%** (the metric trap) |
| Median / mean Amount | $22.25 / $55.64 |
| Amount skewness | 11.6 |

![Class imbalance](figures/01_class_imbalance.png)
![Amount and time](figures/02_amount_time.png)

**Preprocessing.** `Amount` (and `Time`) are scaled with `RobustScaler` (median / IQR). The scaler is fitted on normal
training rows only, so no information leaks from the test set or from fraud. 

![Scaler comparison](figures/03_scaling.png)

## 3. Methodology

* **Training strategy:** detectors are trained on *normal* transactions only. A further 20 % of normal training rows are
  held out as a *calibration set*; the alert threshold is the `(1 − contamination)` quantile of calibration scores, so it
  needs **no fraud labels** and the false-positive rate lands close to the contamination value.
* **Isolation Forest:** 200 trees, `max_samples=512` – anomalies need fewer random splits to isolate.
* **Local Outlier Factor:** `n_neighbors=35`, `novelty=True`, trained on ≤ 20,000 rows.
* **Fusion (extra):** both scores are mapped to calibrated tail probabilities (`-log10 P(normal ≥ score)`) and averaged.
* **Model selection:** highest PR-AUC; if models are within 0.03 PR-AUC the lower-latency one wins (real-time constraint).
* **Metrics:** Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrices. Accuracy is deliberately not used.

## 4. Model comparison

| Model | PR-AUC (AP) | ROC-AUC | Best F1 | Fit (s) | Score test (s) | Latency (ms/txn) |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.580 | 0.974 | 0.608 | 0.4 | 0.3 | 0.009 |
| Local Outlier Factor | 0.287 | 0.979 | 0.456 | 2.1 | 2.8 | 0.092 |
| Fusion (IF + LOF) | 0.590 | 0.982 | 0.630 | 2.6 | 3.0 | 0.101 |

![PR curves](figures/06_pr_curves.png)
![Runtime](figures/09_runtime.png)

**Reading the results**

* **Isolation Forest vs LOF:** Isolation Forest has the higher
  PR-AUC (0.580 vs 0.287).
  Isolation Forest also scores the test set 10× faster than LOF here, and LOF's
  neighbour search scales poorly with data size, which is why it is trained on a capped sample.
* **Fusion** reaches PR-AUC 0.590: combining a tree-based and a density-based view helps
  when each model misses different fraud, at the cost of running both.
* ROC-AUC is high for every model (> 0.97) *because of* the imbalance – PR-AUC is the honest metric.

## 5. Threshold tuning

**Scenario A – higher threshold (small contamination):** few false alarms, but fraud slips through.
**Scenario B – lower threshold (large contamination):** most fraud is caught, but many legitimate customers are challenged.

![Contamination sweep](figures/07_contamination_sweep.png)

### Three defensible operating points

| Model | Strategy | contamination | Recall | Precision | False alarms | FPR | Fraud $ missed |
|---|---|---|---|---|---|---|---|
| Isolation Forest | Max F1 | 0.0005 | 52.2% | 71.2% | 19 | 0.06% | $1,908 |
| Isolation Forest | Min total cost | 0.0075 | 67.8% | 22.3% | 212 | 0.71% | $1,098 |
| Isolation Forest | Recall ≥ 80% | 0.0300 | 82.2% | 7.5% | 913 | 3.05% | $649 |
| Local Outlier Factor | Max F1 | 0.0010 | 33.3% | 56.6% | 23 | 0.08% | $5,159 |
| Local Outlier Factor | Min total cost | 0.0075 | 64.4% | 19.9% | 234 | 0.78% | $2,293 |
| Local Outlier Factor | Recall ≥ 80% | 0.0200 | 80.0% | 9.0% | 726 | 2.43% | $2,017 |
| Fusion (IF + LOF) | Max F1 | 0.0010 | 56.7% | 68.9% | 23 | 0.08% | $2,617 |
| Fusion (IF + LOF) | Min total cost | 0.0020 | 63.3% | 51.4% | 54 | 0.18% | $1,377 |
| Fusion (IF + LOF) | Recall ≥ 80% | 0.0150 | 80.0% | 12.6% | 498 | 1.66% | $853 |

### Sweep for Isolation Forest

| contamination | precision | recall | F1 | false alarms | missed fraud | FPR |
|---|---|---|---|---|---|---|
| 0.0002 | 0.755 | 0.444 | 0.559 | 13 | 50 | 0.04% |
| 0.0005 | 0.712 | 0.522 | 0.603 | 19 | 43 | 0.06% |
| 0.0010 | 0.591 | 0.578 | 0.584 | 36 | 38 | 0.12% |
| 0.0020 | 0.455 | 0.611 | 0.521 | 66 | 35 | 0.22% |
| 0.0030 | 0.384 | 0.622 | 0.475 | 90 | 34 | 0.30% |
| 0.0050 | 0.296 | 0.656 | 0.408 | 140 | 31 | 0.47% |
| 0.0075 | 0.223 | 0.678 | 0.336 | 212 | 29 | 0.71% |
| 0.0100 | 0.183 | 0.711 | 0.292 | 285 | 26 | 0.95% |
| 0.0150 | 0.138 | 0.767 | 0.234 | 431 | 21 | 1.44% |
| 0.0200 | 0.115 | 0.789 | 0.201 | 546 | 19 | 1.83% |
| 0.0300 | 0.075 | 0.822 | 0.137 | 913 | 16 | 3.05% |
| 0.0400 | 0.059 | 0.833 | 0.110 | 1204 | 15 | 4.03% |
| 0.0500 | 0.049 | 0.856 | 0.093 | 1490 | 13 | 4.98% |
| 0.0750 | 0.036 | 0.922 | 0.069 | 2222 | 7 | 7.43% |
| 0.1000 | 0.027 | 0.933 | 0.052 | 3027 | 6 | 10.12% |

## 6. Final recommendation

* **Model:** Isolation Forest.
* **Balanced option (max F1):** contamination **0.0005** → recall **52%**, precision **71%**, FPR **0.06%**.
* **Cost-optimal option ($):** contamination **0.0075** → total cost $1,644 (missed fraud + $2 per reviewed alert).
* **Recall-first (compliance) option:** contamination **0.030** reaches **82% recall** at the price of 913 false alarms (7% precision).

> *Example sentence for stakeholders:* "We recommend a contamination threshold of 0.0005 using Isolation Forest, which yields a
> recall of 52% with a false-positive rate of 0.06%."

![Confusion matrices](figures/08_confusion_matrices.png)
![PCA scatter](figures/04_pca_2d.png)

## 7. Limitations & next steps

1. The threshold is tuned on the same hold-out set that is reported; in production use a separate validation window.
2. Fraud evolves – re-calibrate on fresh normal traffic on a schedule and monitor score drift.
3. Features are PCA-anonymised, so explanations are statistical (which components deviate), not business-level.
4. Next: add a supervised second stage for analyst-confirmed cases, streaming scoring (Kafka / Flink) and model-drift alerts.

---
*Generated by `python run_pipeline.py` · SentinelPay v1.0*
