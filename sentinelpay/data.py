"""Data loading, synthetic demo data, EDA helpers and leakage-free preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import DATA_DIR, PCA_FEATURES, PipelineConfig

DAY = 86_400


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def generate_synthetic_transactions(n_samples: int = 100_000, fraud_rate: float = 0.003,
                                    seed: int = 42) -> pd.DataFrame:
    """Create a dataset that mimics the public *ULB credit-card fraud* data.

    Same schema (``Time``, ``V1..V28``, ``Amount``, ``Class``), PCA-like decaying
    variances, heavy-ish tails, a day/night cycle, skewed amounts, and an extreme
    class imbalance. Fraud is only *partially* separable (some cases are subtle) so
    that threshold tuning is a real decision rather than a toy.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_samples * fraud_rate)), 30)
    n_norm = n_samples - n_fraud

    sigmas = 1.96 * (0.33 / 1.96) ** (np.arange(28) / 27)          # 1.96 ... 0.33 like real PCA output

    def t_noise(n):                                                # unit-variance Student-t (df=6)
        return rng.standard_t(6, size=(n, 28)) / np.sqrt(6 / 4)

    normal = t_noise(n_norm) * sigmas
    fraud = t_noise(n_fraud) * sigmas * 1.35

    shifts = {"V1": -2.0, "V2": 1.3, "V3": -2.3, "V4": 1.9, "V5": -1.0, "V7": -1.6, "V9": -1.2,
              "V10": -2.1, "V11": 1.7, "V12": -2.5, "V14": -2.8, "V16": -1.7, "V17": -2.1, "V18": -1.0}
    intensity = rng.uniform(0.25, 1.15, size=(n_fraud, 1))         # some fraud is subtle
    for name, mult in shifts.items():
        j = int(name[1:]) - 1
        fraud[:, j] += mult * sigmas[j] * intensity[:, 0]

    def sample_times(n, night_bias):
        out = []
        while sum(map(len, out)) < n:
            t = rng.uniform(0, 2 * DAY, size=n * 3)
            hour = (t % DAY) / 3600
            day_curve = 0.25 + 0.75 * np.clip(np.sin((hour - 6) / 24 * 2 * np.pi * 0.98 + 0.3), 0, 1)
            night = ((hour < 5.5)).astype(float)
            p = (1 - night_bias) * day_curve + night_bias * (0.15 + 0.85 * night)
            out.append(t[rng.uniform(size=t.size) < p])
        return np.concatenate(out)[:n]

    amount_n = rng.lognormal(3.1, 1.35, n_norm)
    is_testing = rng.uniform(size=n_fraud) < 0.45                  # "card testing" = tiny amounts
    amount_f = np.where(is_testing, rng.lognormal(1.0, 0.8, n_fraud), rng.lognormal(4.7, 1.1, n_fraud))

    df = pd.concat([
        pd.DataFrame(normal, columns=PCA_FEATURES).assign(Time=sample_times(n_norm, 0.0),
                                                           Amount=amount_n, Class=0),
        pd.DataFrame(fraud, columns=PCA_FEATURES).assign(Time=sample_times(n_fraud, 0.55),
                                                          Amount=amount_f, Class=1),
    ], ignore_index=True)
    df["Amount"] = df["Amount"].round(2)
    df["Time"] = df["Time"].round(0)
    df = df.sample(frac=1.0, random_state=seed).sort_values("Time").reset_index(drop=True)
    return df[["Time", *PCA_FEATURES, "Amount", "Class"]]


def load_data(path: str | Path | None = None, n_samples: int = 100_000, fraud_rate: float = 0.003,
              seed: int = 42) -> tuple[pd.DataFrame, str]:
    """Load ``creditcard.csv`` if present, otherwise fall back to synthetic demo data.

    Returns ``(dataframe, source_label)``.
    """
    candidates = [Path(path)] if path else []
    candidates.append(DATA_DIR / "creditcard.csv")
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            missing = {"Time", "Amount", "Class", *PCA_FEATURES} - set(df.columns)
            if missing:
                raise ValueError(f"{p} is missing expected columns: {sorted(missing)}")
            return df, f"Real data: {p.name}"
    return generate_synthetic_transactions(n_samples, fraud_rate, seed), "Synthetic demo data (ULB-style)"


# ----------------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------------
def eda_summary(df: pd.DataFrame) -> dict:
    """Headline numbers used by the dashboard, notebook and report."""
    fraud = df["Class"].sum()
    return {
        "n_transactions": int(len(df)),
        "n_fraud": int(fraud),
        "n_normal": int(len(df) - fraud),
        "fraud_rate": float(fraud / len(df)),
        "imbalance_ratio": float((len(df) - fraud) / max(fraud, 1)),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
        "amount_skew": float(df["Amount"].skew()),
        "fraud_amount_median": float(df.loc[df.Class == 1, "Amount"].median()),
        "span_hours": float((df["Time"].max() - df["Time"].min()) / 3600),
        "naive_accuracy": float(1 - fraud / len(df)),
    }


def feature_separation(df: pd.DataFrame, top: int = 10) -> pd.Series:
    """|standardised mean difference| between fraud and normal – which V-features matter."""
    f, n = df[df.Class == 1][PCA_FEATURES], df[df.Class == 0][PCA_FEATURES]
    d = (f.mean() - n.mean()).abs() / n.std()
    return d.sort_values(ascending=False).head(top)


# ----------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------
@dataclass
class SplitData:
    """Everything the models need – scaled, split and leakage-free."""
    X_fit: pd.DataFrame        # normal-only rows used to *train* the detectors
    X_calib: pd.DataFrame      # normal-only rows used to *calibrate* the alert threshold
    X_test: pd.DataFrame       # untouched hold-out: normal + fraud
    y_test: np.ndarray
    raw_test: pd.DataFrame     # un-scaled Amount/Time of the test rows (for business cost + display)
    scaler: RobustScaler
    scaled_cols: list[str]
    feature_names: list[str]


def preprocess(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> SplitData:
    """Stratified split -> RobustScaler fitted on *normal training data only* -> normal-only fit set.

    Why RobustScaler?  ``Amount`` is extremely right-skewed (a few huge purchases). The mean and
    standard deviation used by ``StandardScaler`` get dragged by those outliers; ``RobustScaler``
    uses the median and IQR, so the bulk of normal behaviour keeps a sensible scale.
    """
    cfg = cfg or PipelineConfig()
    scaled_cols = ["Amount", "Time"] + (PCA_FEATURES if cfg.scale_all else [])
    feature_names = ["Time", *PCA_FEATURES, "Amount"]

    train, test = train_test_split(df, test_size=cfg.test_size, stratify=df["Class"],
                                   random_state=cfg.seed)
    normal_train = train[train.Class == 0]
    fit_rows, calib_rows = train_test_split(normal_train, test_size=cfg.calib_size,
                                            random_state=cfg.seed)

    scaler = RobustScaler().fit(fit_rows[scaled_cols])             # statistics come from normal train only

    def transform(part: pd.DataFrame) -> pd.DataFrame:
        X = part[feature_names].copy()
        X[scaled_cols] = scaler.transform(part[scaled_cols])
        return X.reset_index(drop=True)

    return SplitData(
        X_fit=transform(fit_rows), X_calib=transform(calib_rows), X_test=transform(test),
        y_test=test["Class"].to_numpy(), raw_test=test[["Time", "Amount", "Class"]].reset_index(drop=True),
        scaler=scaler, scaled_cols=scaled_cols, feature_names=feature_names,
    )
