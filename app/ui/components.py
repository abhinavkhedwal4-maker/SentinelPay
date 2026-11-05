from __future__ import annotations

import streamlit as st


def kpi(col, value: str, label: str, tone: str = "") -> None:
    col.markdown(
        f'<div class="kpi {tone} fade-up">'
        f'<div class="v">{value}</div>'
        f'<div class="l">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_DOCK_CSS = """
<style>
.top-dock {
  position: sticky;
  top: 0;
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  margin: 0 0 20px 0;
  background: rgba(11, 16, 32, 0.88);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border-bottom: 1px solid var(--line);
  border-radius: 0 0 16px 16px;
}
.dock-logo {
  font-family: 'Sora', sans-serif;
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--text);
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 8px;
}
.dock-logo span { color: var(--amber); }
@media (max-width: 900px) {
  .top-dock { overflow-x: auto; flex-wrap: nowrap; gap: 12px; }
}
</style>
"""


def render_top_dock(has_local: bool) -> tuple:
    st.markdown(_DOCK_CSS, unsafe_allow_html=True)

    logo_col, nav_col, right_col = st.columns([2, 5, 3])

    with logo_col:
        st.markdown(
            '<div class="dock-logo">\U0001f6e1\ufe0f Sentinel<span>Pay</span></div>',
            unsafe_allow_html=True,
        )

    with nav_col:
        nav_choice = st.segmented_control(
            "Section",
            options=["Overview", "Detection", "Model lab", "AI analyst", "Report"],
            default="Overview",
            key="nav_section",
            label_visibility="collapsed",
        )

    with right_col:
        btn_col, pop_col = st.columns([1, 1])
        with pop_col:
            with st.popover("\u2699 Engine settings", use_container_width=True):
                st.markdown("**Data**")
                uploaded = st.file_uploader(
                    "creditcard.csv (optional)",
                    type="csv",
                    key="file_upload",
                )
                st.caption(
                    "Using `data/creditcard.csv`" if has_local and not uploaded
                    else "No file \u2192 synthetic ULB-style demo data" if not uploaded
                    else "Using uploaded file"
                )
                n_samples = st.slider(
                    "Demo size",
                    20_000, 150_000, 50_000, 10_000,
                    disabled=bool(uploaded) or has_local,
                    key="n_samples",
                )
                st.markdown("**Isolation Forest**")
                n_est = st.slider("n_estimators", 50, 500, 100, 50, key="n_est")
                st.markdown("**Local Outlier Factor**")
                n_nb = st.slider("n_neighbors", 10, 80, 35, 5, key="n_nb")
                lof_cap = st.select_slider(
                    "LOF training rows",
                    [5_000, 10_000, 20_000, 40_000],
                    value=10_000,
                    key="lof_cap",
                )

        with btn_col:
            run_pressed = st.button(
                "\u25b6 Run detection",
                use_container_width=True,
                type="primary",
                key="run_btn",
            )

    return nav_choice or "Overview", run_pressed, uploaded, n_samples, n_est, n_nb, lof_cap
