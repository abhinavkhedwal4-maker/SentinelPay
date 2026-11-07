from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from sentinelpay import viz
from sentinelpay.config import CONTAMINATION_GRID
from sentinelpay.models import FUSION, IFOREST, LOF
from app.ui.components import kpi


def _show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(res, analyst) -> None:
    h, s = res.headline, res.split

    model = st.radio(
        "Model",
        [IFOREST, LOF, FUSION],
        index=[IFOREST, LOF, FUSION].index(h["model"]),
        horizontal=True,
        label_visibility="collapsed",
    )

    grid = sorted(set(CONTAMINATION_GRID + [h["contamination"]]))
    cont = st.select_slider(
        "Contamination (share of traffic flagged as anomalous)",
        grid,
        value=res.recs[model]["max_f1"]["contamination"],
        format_func=lambda v: f"{v:.2%}",
    )

    n_test = res.eda.get("n_transactions", 100_000) * 0.3
    approx_alerts = int(cont * n_test)
    st.caption(
        f"Flagging **{cont:.2%}** of traffic \u2248 **{approx_alerts:,} alerts** "
        f"per {n_test:,.0f} transactions"
    )

    m = _evaluate(res, model, cont)

    ka, kb, kc, kd = st.columns(4)
    kpi(ka, f"{m['recall']:.0%}", "recall", "good")
    kpi(kb, f"{m['precision']:.0%}", "precision", "warn")
    kpi(kc, f"{m['fp']}", "false alarms", "bad")
    kpi(kd, f"{m['fn']}", "missed fraud", "bad")

    ma, mb = st.columns(2)
    kpi(ma, f"\\${m['fraud_prevented']:,.0f}", "fraud prevented", "good")
    kpi(mb, f"\\${m['fraud_loss']:,.0f}", "fraud missed", "bad")

    st.caption(f"F1 {m['f1']:.3f} \xb7 FPR {m['fpr']:.2%}")

    pred = _predictions(res, model, cont)
    st.markdown(
        '<div class="glass fade-up d3" style="padding:16px 18px 8px;">'
        '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
        'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
        'PCA detection map</div>',
        unsafe_allow_html=True,
    )
    _show(viz.plot_pca_scatter(s.X_test, s.y_test, pred, f"{model} \xb7 contamination {cont:.2%}"))
    st.markdown(
        '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
        'Amber dots = caught fraud, coral rings = missed fraud, coral dots = false alarms.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="glass fade-up d4" style="padding:16px 18px 8px;">'
            '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
            'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
            'Confusion matrix</div>',
            unsafe_allow_html=True,
        )
        _show(viz.plot_confusion(m, "Confusion matrix"))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="glass fade-up d5" style="padding:16px 18px 8px;">'
            '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
            'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
            'Score distribution</div>',
            unsafe_allow_html=True,
        )
        _show(viz.plot_score_distribution(
            res.models[model].scores_test, s.y_test, m["threshold"], "Score distribution"
        ))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("3-D PCA view"):
        _show(viz.plot_pca_3d(s.X_test, s.y_test, pred, f"{model} in 3-D"))
