# Data

SentinelPay works out of the box with a **synthetic ULB-style dataset** (same schema as the real one).

To use the real data:

1. Download **Credit Card Fraud Detection** (ULB / Worldline) from Kaggle:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Save it as `data/creditcard.csv` (284,807 rows · columns `Time, V1…V28, Amount, Class`).
3. Re-run `python run_pipeline.py` and `python scripts/build_notebook.py` – the loader picks the file up automatically,
   and the dashboard shows "Real data: creditcard.csv".

The CSV is git-ignored (150 MB). On the full dataset LOF is the slow part – lower `lof_max_train` in `sentinelpay/config.py` if needed.
