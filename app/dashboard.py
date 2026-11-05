from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import streamlit as st
from dotenv import load_dotenv

from sentinelpay.config import DATA_DIR
from sentinelpay.genai import NvidiaAnalyst
from sentinelpay.models import FUSION, IFOREST, LOF  # noqa: F401
from sentinelpay.pipeline import run_pipeline
from sentinelpay.config import PipelineConfig

from app.ui.theme import inject_theme
from app.ui.components import render_top_dock
from app.ui.hero import render_hero, render_pre_run_cards
from app.ui.pages import overview, detection, model_lab, ai_analyst, report as report_page

load_dotenv()
st.set_page_config(
    page_title="SentinelPay",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()

analyst = NvidiaAnalyst()
has_local = (DATA_DIR / "creditcard.csv").exists()

nav, run, uploaded, n_samples, n_est, n_nb, lof_cap = render_top_dock(has_local)

render_hero(nvidia_online=analyst.available)

if "ready" not in st.session_state:
    st.session_state.ready = False
if run:
    st.session_state.ready = True
if not st.session_state.ready:
    render_pre_run_cards()
    st.info(
        "Open **\u2699 Engine settings** above to configure the detectors, "
        "then press **\u25b6 Run detection** to start. The first run takes ~10 seconds."
    )
    st.stop()


@st.cache_resource(show_spinner="Training detectors on normal traffic\u2026")
def get_result(data_bytes, n_samples, n_est, n_nb, lof_cap):
    path = None
    if data_bytes is not None:
        path = str(DATA_DIR / "_uploaded.csv")
        DATA_DIR.mkdir(exist_ok=True)
        Path(path).write_bytes(data_bytes)
    cfg = PipelineConfig(
        data_path=path,
        n_samples=n_samples,
        if_n_estimators=n_est,
        lof_n_neighbors=n_nb,
        lof_max_train=lof_cap,
    )
    return run_pipeline(cfg)


res = get_result(uploaded.getvalue() if uploaded else None, n_samples, n_est, n_nb, lof_cap)

SECTIONS = ["Overview", "Detection", "Model lab", "AI analyst", "Report"]
section = nav if nav in SECTIONS else "Overview"

if section == "Overview":
    overview.render(res, analyst)
elif section == "Detection":
    detection.render(res, analyst)
elif section == "Model lab":
    model_lab.render(res, analyst)
elif section == "AI analyst":
    ai_analyst.render(res, analyst)
elif section == "Report":
    report_page.render(res, analyst)

data_label = "real creditcard.csv" if has_local else "synthetic ULB-style"
st.markdown(
    f'<div style="margin-top:48px;padding-top:16px;border-top:1px solid #243055;'
    f'text-align:center;color:#8A94B8;font-size:.8rem;">'
    f'SentinelPay &middot; Isolation Forest + LOF &middot; NVIDIA AI analyst'
    f' &middot; Data: {data_label}'
    f'</div>',
    unsafe_allow_html=True,
)
