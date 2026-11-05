from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sentinelpay import viz


def _show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(res, analyst) -> None:
    h = res.headline

    st.markdown(
        '<div class="glass fade-up d1" style="padding:16px 18px 8px;">'
        '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
        'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
        'Precision\u2013Recall curves</div>',
        unsafe_allow_html=True,
    )
    _show(viz.plot_pr_curves(res.pr, res.split.y_test))
    st.markdown(
        '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
        'Dots mark the max-F1 operating point; use these curves to pick a contamination level.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    perf = pd.DataFrame({
        n: {
            "PR-AUC":    res.pr[n].average_precision,
            "ROC-AUC":   res.pr[n].roc_auc,
            "Best F1":   res.pr[n].best["f1"],
            "Fit (s)":   mm.fit_time,
            "Score (s)": mm.score_time,
            "ms / txn":  mm.latency_ms_per_txn,
        }
        for n, mm in res.models.items()
    }).T

    st.markdown(
        '<div class="glass fade-up d2" style="padding:16px 18px 8px;">'
        '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
        'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
        'Model performance summary</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="glass fade-up d3" style="padding:16px 18px 8px;">'
        '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
        'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
        'Contamination sweep</div>',
        unsafe_allow_html=True,
    )
    _show(viz.plot_contamination_sweep(
        res.sweeps, {n: res.recs[n]["max_f1"]["contamination"] for n in res.models}
    ))
    st.markdown(
        '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
        'Open circles mark the recommended max-F1 operating point per model.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    sw_model = st.selectbox(
        "Sweep table for model",
        list(res.models),
        index=list(res.models).index(h["model"]),
    )
    st.dataframe(
        res.sweeps[sw_model].drop(columns=["tn", "tp"]).style.format({
            "contamination": "{:.4f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "fpr": "{:.3%}", "threshold": "{:.4f}", "flagged_share": "{:.3%}",
        }),
        use_container_width=True,
    )

    st.markdown(
        '<div class="glass fade-up d4" style="padding:16px 18px 8px;">'
        '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
        'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
        'Execution time</div>',
        unsafe_allow_html=True,
    )
    _show(viz.plot_runtime(res.models))
    st.markdown(
        '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
        'Isolation Forest scores in \u22480.009\u202fms/txn \u2014 real-time safe.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
