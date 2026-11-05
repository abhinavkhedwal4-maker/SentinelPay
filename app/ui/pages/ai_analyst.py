from __future__ import annotations

import streamlit as st

from sentinelpay.genai import feature_deviations, risk_band


_BAND_STYLE = {
    "CRITICAL": ("var(--coral)",  "rgba(255,92,122,.12)"),
    "HIGH":     ("var(--amber)",  "rgba(255,181,71,.12)"),
    "ELEVATED": ("#B8A2FF",       "rgba(184,162,255,.10)"),
    "LOW":      ("var(--muted)",  "rgba(138,148,184,.08)"),
}


def _band_badge(band: str) -> str:
    color, bg = _BAND_STYLE.get(band, _BAND_STYLE["LOW"])
    return (
        f'<span style="padding:3px 9px;border-radius:999px;font-size:.72rem;'
        f'font-weight:700;color:{color};background:{bg};'
        f'border:1px solid {color}40;">{band}</span>'
    )


def _score_bar(pct: float) -> str:
    w = max(2, int(pct))
    return (
        f'<div style="margin-top:6px;height:4px;border-radius:2px;'
        f'background:var(--line);overflow:hidden;">'
        f'<div style="width:{w}%;height:100%;background:var(--amber);'
        f'border-radius:2px;"></div></div>'
    )


def render(res, analyst) -> None:
    h, s = res.headline, res.split
    mdl = h["model"]
    mm = res.models[mdl]

    st.markdown(f"#### Highest-scoring alerts &nbsp;&nbsp; <code>{mdl}</code>", unsafe_allow_html=True)

    order = mm.scores_test.argsort()[::-1][:8]

    for rank, idx in enumerate(order, 1):
        pct = float((mm.scores_calib < mm.scores_test[idx]).mean() * 100)
        band = risk_band(pct)
        amount = float(s.raw_test.Amount.iloc[idx])
        hour = round(float(s.raw_test.Time.iloc[idx] % 86400 / 3600), 1)
        score = float(mm.scores_test[idx])
        is_fraud = bool(s.y_test[idx])

        badge = _band_badge(band)
        bar = _score_bar(pct)
        fraud_label = (
            ' &nbsp;<span style="color:var(--amber);font-size:.72rem;">&#9679; confirmed fraud</span>'
            if is_fraud else ""
        )

        st.markdown(
            f'<div class="glass fade-up" style="padding:16px 20px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
            f'<span style="font-family:\'Sora\',sans-serif;font-size:1.1rem;font-weight:700;'
            f'color:var(--text);">#{rank}</span>'
            f'{badge}'
            f'<span style="color:var(--text);font-weight:600;">\\${amount:,.2f}</span>'
            f'<span style="color:var(--muted);font-size:.85rem;">hour {hour:.1f}</span>'
            f'<span style="color:var(--muted);font-size:.85rem;">'
            f'score {score:.3f} &middot; {pct:.1f}th pct</span>'
            f'{fraud_label}'
            f'</div>'
            f'{bar}'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_explain, col_spacer = st.columns([1, 5])
        with col_explain:
            if st.button(f"Explain #{rank}", key=f"explain_{rank}_{idx}"):
                row = s.X_test.iloc[idx]
                case = {
                    "model": mdl,
                    "anomaly_score": round(score, 4),
                    "score_percentile": round(pct, 3),
                    "risk_band": band,
                    "amount": amount,
                    "hour_of_day": hour,
                    "deviations": feature_deviations(row, s.X_fit),
                }
                with st.spinner("Consulting the analyst\u2026"):
                    text, src = analyst.explain_transaction(case)
                safe_text = text.replace("$", "\\$")
                st.chat_message("assistant").markdown(safe_text)
                st.caption(
                    f"Source: {'NVIDIA \xb7 ' + analyst.model if src == 'nvidia' else 'offline rule-based fallback'}"
                    + (f" \xb7 {analyst.last_error}" if analyst.last_error else "")
                )

    st.markdown("---")
    st.markdown("#### Ask about the results")
    q = st.text_input(
        "Free-text question",
        placeholder="\u201cShould we favour recall or precision for a card-not-present channel?\u201d",
        label_visibility="collapsed",
    )
    if q:
        ctx = {
            "headline": h,
            "eda": res.eda,
            "recommendations": {
                n: {k: (v or {}).get("contamination") for k, v in r_.items()}
                for n, r_ in res.recs.items()
            },
        }
        with st.spinner("Thinking\u2026"):
            text, src = analyst.chat(q, ctx)
        safe_text = text.replace("$", "\\$")
        st.chat_message("assistant").markdown(safe_text)
