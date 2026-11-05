"""NVIDIA-hosted LLM analyst (OpenAI-compatible endpoint at build.nvidia.com).

The LLM never sees raw card data – only anonymised PCA features, the anomaly score and
aggregate metrics. If no ``NVIDIA_API_KEY`` is set, or the call fails, every method falls back to a
deterministic rule-based explanation so the app keeps working offline.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

SYSTEM_PROMPT = (
    "You are SentinelPay's senior fraud analyst. You receive output from unsupervised anomaly detectors "
    "(Isolation Forest, Local Outlier Factor) run on PCA-anonymised card transactions. Features V1-V28 are "
    "anonymous components: never invent merchants, countries or card details. Be concise, specific and honest "
    "about uncertainty. Anomaly != fraud: say so when evidence is weak."
)


def feature_deviations(x: pd.Series, reference: pd.DataFrame, top: int = 5) -> list[dict]:
    """Rank features by robust z-score against *normal* reference rows (median / IQR-based sigma)."""
    med = reference.median()
    sigma = (reference.quantile(0.75) - reference.quantile(0.25)) / 1.349
    z = (x[reference.columns] - med) / sigma.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    order = z.abs().sort_values(ascending=False).head(top).index
    return [{"feature": f, "value": round(float(x[f]), 3), "robust_z": round(float(z[f]), 2)} for f in order]


def risk_band(percentile: float) -> str:
    return "CRITICAL" if percentile >= 99.9 else "HIGH" if percentile >= 99 else "ELEVATED" if percentile >= 95 else "LOW"


class NvidiaAnalyst:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.model = model or os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL)
        self.last_error: str | None = None
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.api_key.startswith("nvapi-")

    def _complete(self, user_prompt: str, max_tokens: int = 450, temperature: float = 0.3) -> str:
        if self._client is None:
            from openai import OpenAI  # lazy import: the rest of the project works without it
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=40)
        r = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
        return r.choices[0].message.content.strip()

    def _ask(self, prompt: str, fallback: str, **kw) -> tuple[str, str]:
        """Return ``(text, source)`` where source is 'nvidia' or 'offline'."""
        if not self.available:
            return fallback, "offline"
        try:
            self.last_error = None
            return self._complete(prompt, **kw), "nvidia"
        except Exception as exc:  # network / auth / quota – never crash the UI
            self.last_error = f"{type(exc).__name__}: {exc}"
            return fallback, "offline"

    # ---------------------------------------------------------------- public API
    def explain_transaction(self, case: dict) -> tuple[str, str]:
        """``case``: amount, hour, score, percentile, model, deviations (list from feature_deviations)."""
        prompt = (
            "Explain why this transaction was flagged and recommend an action.\n"
            f"```json\n{json.dumps(case, indent=2)}\n```\n"
            "Reply in this exact format:\n**Verdict:** <one line>\n**Why it stands out:** <2-3 sentences>\n"
            "**Recommended action:** <approve | step-up authentication | hold for manual review | block> + one reason\n"
            "**Confidence caveat:** <one sentence>"
        )
        return self._ask(prompt, self._offline_case(case))

    def executive_summary(self, facts: dict) -> tuple[str, str]:
        prompt = (
            "Write a 130-word executive summary for the SecurePay leadership team from these evaluation results. "
            "State the recommended model and contamination, what recall/precision it delivers, the false-alarm "
            "burden, and one risk. No headings, plain prose.\n"
            f"```json\n{json.dumps(facts, indent=2, default=float)}\n```"
        )
        return self._ask(prompt, self._offline_summary(facts), max_tokens=350)

    def chat(self, question: str, context: dict) -> tuple[str, str]:
        prompt = (f"Context (JSON):\n```json\n{json.dumps(context, indent=2, default=float)}\n```\n"
                  f"Analyst question: {question}\nAnswer in under 150 words using only the context and general "
                  "fraud-analytics knowledge.")
        return self._ask(prompt, "The NVIDIA analyst is offline (no valid `NVIDIA_API_KEY`). Add one to `.env` to "
                                 "ask free-form questions about these results.", max_tokens=400)

    # ---------------------------------------------------------------- offline fallbacks
    @staticmethod
    def _offline_case(c: dict) -> str:
        band = risk_band(c.get("score_percentile", 0))
        devs = ", ".join(f"{d['feature']} ({d['robust_z']:+.1f}σ)" for d in c.get("deviations", [])[:3]) or "n/a"
        action = {"CRITICAL": "block", "HIGH": "hold for manual review", "ELEVATED": "step-up authentication"}.get(band, "approve")
        return (f"**Verdict:** {band} risk – scores higher than {c.get('score_percentile', 0):.2f}% of normal traffic.\n\n"
                f"**Why it stands out:** Largest deviations from normal behaviour: {devs}. "
                f"Amount ${c.get('amount', 0):,.2f} at hour {c.get('hour_of_day', '?')}.\n\n"
                f"**Recommended action:** {action}.\n\n"
                "**Confidence caveat:** Rule-based summary (NVIDIA analyst offline); an anomaly is not proof of fraud.")

    @staticmethod
    def _offline_summary(f: dict) -> str:
        return (f"We recommend {f.get('model')} with a contamination of {f.get('contamination')}. On the hold-out set it "
                f"catches {f.get('recall', 0):.0%} of fraud at {f.get('precision', 0):.0%} precision, producing "
                f"{f.get('false_alarms')} false alarms among {f.get('n_test'):,} transactions "
                f"(false-positive rate {f.get('fpr', 0):.2%}). Main risk: fraud patterns drift, so thresholds should be "
                "re-calibrated on fresh normal traffic regularly.")
