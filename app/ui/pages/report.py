from __future__ import annotations

import re

import streamlit as st

from sentinelpay.config import REPORT_DIR
from sentinelpay.report import write_report


def render(res, analyst) -> None:
    left, right = st.columns([3, 1])

    with right:
        if st.button("Generate report", use_container_width=True, type="primary"):
            with st.spinner("Writing report\u2026"):
                md = write_report(res, REPORT_DIR / "Performance_Analysis_Report.md", analyst)
            st.session_state.report_md = md
        md = st.session_state.get("report_md")
        if md:
            st.download_button(
                "Download .md",
                md,
                "Performance_Analysis_Report.md",
                "text/markdown",
                use_container_width=True,
            )

    with left:
        md = st.session_state.get("report_md")
        if md:
            safe_md = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", md)
            safe_md = safe_md.replace("$", "\\$")
            st.markdown(safe_md)
        else:
            st.caption(
                "Press **Generate report** to create "
                "`reports/Performance_Analysis_Report.md` with this run\u2019s numbers."
            )

    if st.session_state.get("report_md"):
        fig_dir = REPORT_DIR / "figures"
        figs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
        if figs:
            st.markdown("---")
            st.markdown("#### Figures")
            cols = st.columns(2)
            for i, fp in enumerate(figs):
                cols[i % 2].image(str(fp), use_container_width=True, caption=fp.stem)
