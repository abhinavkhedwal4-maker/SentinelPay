from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from sentinelpay import viz
from sentinelpay.data import feature_separation
from app.ui.components import kpi


def _show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(res, analyst) -> None:
    e, h = res.eda, res.headline
    c = st.columns(4)
    kpi(c[0], f"{e['n_transactions']:,}", "transactions analysed")
    kpi(c[1], f"{e['n_fraud']:,}", f"fraud cases ({e['fraud_rate']:.2%})", "warn")
    kpi(
        c[2],
        f"{e['naive_accuracy']:.2%}",
        '\u201calways legitimate\u201d accuracy \u2013 the metric trap',
        "bad",
    )
    kpi(c[3], h["model"].split(" (")[0], f"recommended model \xb7 {res.source}", "good")

    st.write("")

    tile_l, tile_r = st.columns([3, 4])
    with tile_l:
        st.markdown(
            '<div class="glass fade-up d1" style="padding:16px 18px 8px;">'
            '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
            'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
            'Class distribution</div>',
            unsafe_allow_html=True,
        )
        _show(viz.plot_class_imbalance(res.df))
        st.markdown(
            '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
            'Severe imbalance means accuracy is meaningless \u2014 use precision &amp; recall.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with tile_r:
        st.markdown(
            '<div class="glass fade-up d2" style="padding:16px 18px 8px;">'
            '<div style="font-size:.75rem;font-weight:700;color:var(--muted);'
            'letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">'
            'Amount &amp; time patterns</div>',
            unsafe_allow_html=True,
        )
        _show(viz.plot_amount_time(res.df))
        st.markdown(
            '<div style="color:var(--muted);font-size:.82rem;padding:4px 0 8px;">'
            'Fraud clusters at smaller amounts and shows a day/night pattern.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    exp_l, exp_r = st.columns(2)
    with exp_l:
        with st.expander("Why RobustScaler?"):
            _show(viz.plot_scaling_comparison(res.df))
    with exp_r:
        with st.expander("Which PCA features separate fraud?"):
            _show(viz.plot_feature_separation(feature_separation(res.df)))
