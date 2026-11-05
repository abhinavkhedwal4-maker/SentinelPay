import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinelpay.config import PipelineConfig
from sentinelpay.data import generate_synthetic_transactions, preprocess
from sentinelpay.evaluation import metrics_at_threshold, pr_analysis
from sentinelpay.genai import NvidiaAnalyst, feature_deviations
from sentinelpay.models import (IFOREST, run_fusion, run_isolation_forest, run_lof,
                                tail_score, threshold_from_contamination)

CFG = PipelineConfig(n_samples=20_000, lof_max_train=4_000, if_n_estimators=60)


@pytest.fixture(scope="module")
def setup():
    df = generate_synthetic_transactions(20_000, 0.004, seed=1)
    split = preprocess(df, CFG)
    return df, split


def test_synthetic_schema_and_imbalance():
    df = generate_synthetic_transactions(10_000, 0.003, seed=3)
    assert {"Time", "Amount", "Class", "V1", "V28"} <= set(df.columns)
    assert df.Class.mean() == pytest.approx(0.003, abs=5e-4)
    assert df.Amount.min() > 0


def test_training_sets_contain_only_normal_and_no_leak(setup):
    df, s = setup
    train_fraud = int(df.Class.sum() - s.y_test.sum())      # fraud rows that were dropped from training
    assert len(s.X_fit) + len(s.X_calib) + len(s.X_test) == len(df) - train_fraud
    assert s.y_test.sum() > 0
    # scaler statistics come from the normal training rows only -> fit-set median of Amount ~ 0 after scaling
    assert abs(s.X_fit["Amount"].median()) < 1e-6


def test_models_return_one_score_per_row(setup):
    _, s = setup
    a, b = run_isolation_forest(s, CFG), run_lof(s, CFG)
    f = run_fusion(a, b)
    for m in (a, b, f):
        assert m.scores_test.shape == (len(s.X_test),)
        assert np.isfinite(m.scores_test).all()
    assert pr_analysis(s.y_test, a.scores_test).average_precision > 0.2


def test_contamination_threshold_controls_false_positive_rate(setup):
    _, s = setup
    m = run_isolation_forest(s, CFG)
    for c in (0.01, 0.05):
        thr = threshold_from_contamination(m.scores_calib, c)
        fpr = metrics_at_threshold(s.y_test, m.scores_test, thr)["fpr"]
        assert fpr == pytest.approx(c, abs=0.012)


def test_metrics_confusion_counts():
    y = np.array([0, 0, 0, 1, 1]); sc = np.array([.1, .2, .9, .8, .3])
    m = metrics_at_threshold(y, sc, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_tail_score_monotonic():
    calib = np.random.default_rng(0).normal(size=1000)
    t = tail_score(calib, np.array([-3, 0, 3]))
    assert t[0] < t[1] < t[2]


def test_analyst_falls_back_offline_without_key():
    a = NvidiaAnalyst(api_key="")
    text, src = a.explain_transaction({"score_percentile": 99.95, "amount": 12.5, "hour_of_day": 3,
                                       "deviations": [{"feature": "V14", "robust_z": -6.1}]})
    assert src == "offline" and "CRITICAL" in text


def test_analyst_uses_nvidia_endpoint_when_key_present(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            seen.update(base_url=base_url, api_key=api_key)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kw):
            seen["model"] = kw["model"]
            msg = types.SimpleNamespace(content=" **Verdict:** test ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    a = NvidiaAnalyst(api_key="nvapi-test")
    text, src = a.explain_transaction({"score_percentile": 99})
    assert src == "nvidia" and text.startswith("**Verdict")
    assert seen["base_url"] == "https://integrate.api.nvidia.com/v1"


def test_feature_deviations_rank_by_robust_z(setup):
    _, s = setup
    row = s.X_fit.iloc[0].copy(); row["V14"] = 50
    assert feature_deviations(row, s.X_fit, top=1)[0]["feature"] == "V14"
