"""Central configuration: paths, hyper-parameters and the visual identity."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# --- Visual identity (shared by matplotlib figures and the Streamlit dashboard) ---
PALETTE = {
    "bg": "#0B1020",        # ink
    "panel": "#121A33",     # raised surface
    "grid": "#243055",
    "text": "#E6EAF5",
    "muted": "#8A94B8",
    "normal": "#6C8CFF",    # periwinkle  -> legitimate traffic
    "anomaly": "#FFB547",   # amber       -> detected anomalies / true fraud
    "alert": "#FF5C7A",     # coral       -> false alarms / missed fraud
    "good": "#3DDBB0",      # mint        -> positive outcomes
}

# Contamination values (expected share of flagged traffic) explored in the sweep.
CONTAMINATION_GRID = [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01,
                      0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]


@dataclass
class PipelineConfig:
    # data
    data_path: str | None = None          # path to creditcard.csv (None -> synthetic demo data)
    n_samples: int = 100_000              # synthetic demo size
    fraud_rate: float = 0.003             # synthetic demo fraud share
    test_size: float = 0.30
    calib_size: float = 0.20              # share of *normal training* rows held out to calibrate thresholds
    scale_all: bool = False               # True -> RobustScale V1..V28 too
    seed: int = 42
    # Isolation Forest
    if_n_estimators: int = 200
    if_max_samples: int | str = 512
    # Local Outlier Factor
    lof_n_neighbors: int = 35
    lof_max_train: int = 20_000           # LOF is O(n^2)-ish: cap training rows
    # decision making
    target_recall: float = 0.80
    review_cost: float = 2.0              # $ cost of manually reviewing / challenging one flagged transaction
    extra: dict = field(default_factory=dict)
