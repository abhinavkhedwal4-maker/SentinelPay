"""SentinelPay – interactive fraud-detection dashboard (Streamlit).

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sentinelpay import viz
from sentinelpay.config import CONTAMINATION_GRID, DATA_DIR, REPORT_DIR, PipelineConfig
from sentinelpay.data import feature_separation
from sentinelpay.genai import NvidiaAnalyst, feature_deviations, risk_band
from sentinelpay.models import FUSION, IFOREST, LOF, tail_score
from sentinelpay.pipeline import run_pipeline
from sentinelpay.report import write_report

load_dotenv()
st.set_page_config(page_title="SentinelPay", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Manrope:wght@400;500;600&display=swap');
:root{--bg:#0B1020;--panel:#121A33;--line:#243055;--text:#E6EAF5;--muted:#8A94B8;--amber:#FFB547;--blue:#6C8CFF;--coral:#FF5C7A;--mint:#3DDBB0;}
html,body,[class*="css"]{font-family:'Manrope',sans-serif;}
.stApp{background:radial-gradient(1100px 520px at 85% -8%,rgba(108,140,255,.16),transparent 60%),
        radial-gradient(800px 400px at -5% 105%,rgba(255,181,71,.07),transparent 60%),var(--bg);}
h1,h2,h3{font-family:'Sora',sans-serif!important;letter-spacing:-.01em;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1280px;}

/* hero: one orchestrated moment – a radar sweep crossing a field of traffic */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:22px;padding:34px 38px;margin-bottom:22px;
      background:linear-gradient(135deg,rgba(18,26,51,.92),rgba(11,16,32,.92));}
.hero:before{content:"";position:absolute;inset:0;pointer-events:none;
      background:radial-gradient(circle,rgba(108,140,255,.55) 1px,transparent 1.4px) 0 0/22px 22px;opacity:.22;
      mask-image:linear-gradient(90deg,transparent 25%,#000 85%);-webkit-mask-image:linear-gradient(90deg,transparent 25%,#000 85%);}
.hero:after{content:"";position:absolute;top:0;bottom:0;width:140px;left:-160px;pointer-events:none;
      background:linear-gradient(90deg,transparent,rgba(255,181,71,.20),transparent);animation:sweep 6.5s ease-in-out 1 .4s forwards;}
@keyframes sweep{to{left:105%}}
.hero h1{margin:0;font-size:2.7rem;font-weight:800;color:var(--text);}
.hero p{margin:.5rem 0 0;color:var(--muted);font-size:1.05rem;max-width:640px;}
.pill{display:inline-block;margin:14px 8px 0 0;padding:5px 13px;border-radius:999px;font-size:.78rem;font-weight:600;
      border:1px solid var(--line);color:var(--text);background:rgba(108,140,255,.10);}
.pill.live{border-color:rgba(61,219,176,.5);color:var(--mint);background:rgba(61,219,176,.08);}
.pill.off{border-color:rgba(255,92,122,.4);color:var(--coral);background:rgba(255,92,122,.07);}

/* KPI cards */
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;transition:border-color .2s,box-shadow .2s;}
.kpi:hover{border-color:var(--amber);box-shadow:0 0 0 1px rgba(255,181,71,.35),0 10px 34px -12px rgba(255,181,71,.35);}
.kpi .v{font-family:'Sora';font-size:1.85rem;font-weight:700;color:var(--text);line-height:1.1;}
.kpi .l{color:var(--muted);font-size:.85rem;margin-top:6px;}
.kpi.warn .v{color:var(--amber);} .kpi.bad .v{color:var(--coral);} .kpi.good .v{color:var(--mint);}

/* tabs & widgets */
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:10px 18px;color:var(--muted);font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--amber)!important;background:rgba(255,181,71,.07);}
section[data-testid="stSidebar"]{background:#0D142B;border-right:1px solid var(--line);}
.stButton>button{border-radius:12px;border:1px solid var(--amber);background:rgba(255,181,71,.10);color:var(--amber);font-weight:700;
      transition:all .18s;} .stButton>button:hover{background:var(--amber);color:#0B1020;box-shadow:0 8px 26px -8px rgba(255,181,71,.7);}
.ai-box{border:1px solid var(--line);border-left:3px solid var(--blue);border-radius:12px;background:var(--panel);padding:16px 20px;}
.ai-box.nv{border-left-color:var(--mint);}
.muted{color:var(--muted);font-size:.85rem;}
@media (prefers-reduced-motion:reduce){.hero:after{animation:none;display:none}.kpi,.stButton>button{transition:none}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, value: str, label: str, tone: str = "") -> None:
    col.markdown(f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------------------------------------------------ sidebar
analyst = NvidiaAnalyst()
with st.sidebar:
    st.markdown("### 🛡️ SentinelPay")
    st.caption("Unsupervised fraud detection")
    st.markdown("**Data**")
    uploaded = st.file_uploader("creditcard.csv (optional)", type="csv")
    has_local = (DATA_DIR / "creditcard.csv").exists()
    st.caption("Using `data/creditcard.csv`" if has_local and not uploaded else
               "No file → synthetic ULB-style demo data" if not uploaded else "Using uploaded file")
    n_samples = st.slider("Demo size", 20_000, 150_000, 100_000, 10_000, disabled=bool(uploaded) or has_local)
    st.markdown("**Isolation Forest**")
    n_est = st.slider("n_estimators", 50, 500, 200, 50)
    st.markdown("**Local Outlier Factor**")
    n_nb = st.slider("n_neighbors", 10, 80, 35, 5)
    lof_cap = st.select_slider("LOF training rows", [5_000, 10_000, 20_000, 40_000], value=20_000)
    st.divider()
    run = st.button("Run detection", use_container_width=True, type="primary")


@st.cache_resource(show_spinner="Training detectors on normal traffic…")
def get_result(data_bytes, n_samples, n_est, n_nb, lof_cap):
    path = None
    if data_bytes is not None:
        path = str(DATA_DIR / "_uploaded.csv"); DATA_DIR.mkdir(exist_ok=True)
        Path(path).write_bytes(data_bytes)
    cfg = PipelineConfig(data_path=path, n_samples=n_samples, if_n_estimators=n_est,
                         lof_n_neighbors=n_nb, lof_max_train=lof_cap)
    return run_pipeline(cfg)


# ------------------------------------------------------------------ hero
badge = ('<span class="pill live">● NVIDIA analyst online</span>' if analyst.available
         else '<span class="pill off">○ NVIDIA analyst offline – add NVIDIA_API_KEY</span>')
st.markdown(f"""
<div class="hero"><h1>SentinelPay</h1>
<p>Teach the model what <em>normal</em> looks like. Everything that deviates gets flagged – in milliseconds.</p>
<span class="pill">Isolation Forest</span><span class="pill">Local Outlier Factor</span>
<span class="pill">RobustScaler</span><span class="pill">PR-curve tuning</span>{badge}</div>
""", unsafe_allow_html=True)

if "ready" not in st.session_state:
    st.session_state.ready = False
if run:
    st.session_state.ready = True
if not st.session_state.ready:
    st.info("Pick your settings in the sidebar and press **Run detection**. The first run takes ~10 seconds.")
    st.stop()

res = get_result(uploaded.getvalue() if uploaded else None, n_samples, n_est, n_nb, lof_cap)
s, h, e = res.split, res.headline, res.eda

tab_over, tab_det, tab_lab, tab_ai, tab_rep = st.tabs(["Overview", "Detection", "Model lab", "AI analyst", "Report"])

# ------------------------------------------------------------------ overview
with tab_over:
    c = st.columns(4)
    kpi(c[0], f"{e['n_transactions']:,}", "transactions analysed")
    kpi(c[1], f"{e['n_fraud']:,}", f"fraud cases ({e['fraud_rate']:.2%})", "warn")
    kpi(c[2], f"{e['naive_accuracy']:.2%}", "accuracy of “always legitimate” – the metric trap", "bad")
    kpi(c[3], h["model"].split(" (")[0], f"recommended model · {res.source}", "good")
    st.write("")
    show(viz.plot_class_imbalance(res.df))
    show(viz.plot_amount_time(res.df))
    with st.expander("Why RobustScaler?"):
        show(viz.plot_scaling_comparison(res.df))
    with st.expander("Which PCA features separate fraud?"):
        show(viz.plot_feature_separation(feature_separation(res.df)))

# ------------------------------------------------------------------ detection
with tab_det:
    l, r = st.columns([1, 2])
    with l:
        model = st.radio("Model", [IFOREST, LOF, FUSION], index=[IFOREST, LOF, FUSION].index(h["model"]))
        grid = sorted(set(CONTAMINATION_GRID + [h["contamination"]]))
        cont = st.select_slider("contamination (share of traffic flagged)", grid,
                                value=res.recs[model]["max_f1"]["contamination"], format_func=lambda v: f"{v:.2%}")
        m = res.evaluate(model, cont)
        a, b = st.columns(2); kpi(a, f"{m['recall']:.0%}", "recall", "good"); kpi(b, f"{m['precision']:.0%}", "precision", "warn")
        a, b = st.columns(2); kpi(a, f"{m['fp']}", "false alarms", "bad"); kpi(b, f"{m['fn']}", "missed fraud", "bad")
        st.caption(f"F1 {m['f1']:.3f} · FPR {m['fpr']:.2%} · fraud $ missed ${m['fraud_loss']:,.0f} · "
                   f"fraud $ prevented ${m['fraud_prevented']:,.0f}")
    with r:
        pred = res.predictions(model, cont)
        show(viz.plot_pca_scatter(s.X_test, s.y_test, pred, f"{model} · contamination {cont:.2%}"))
    c1, c2 = st.columns(2)
    with c1:
        show(viz.plot_confusion(m, "Confusion matrix"))
    with c2:
        show(viz.plot_score_distribution(res.models[model].scores_test, s.y_test, m["threshold"], "Score distribution"))
    with st.expander("3-D PCA view"):
        show(viz.plot_pca_3d(s.X_test, s.y_test, pred, f"{model} in 3-D"))

# ------------------------------------------------------------------ model lab
with tab_lab:
    show(viz.plot_pr_curves(res.pr, s.y_test))
    perf = pd.DataFrame({n: {"PR-AUC": res.pr[n].average_precision, "ROC-AUC": res.pr[n].roc_auc,
                             "Best F1": res.pr[n].best["f1"], "Fit (s)": mm.fit_time, "Score (s)": mm.score_time,
                             "ms / txn": mm.latency_ms_per_txn} for n, mm in res.models.items()}).T
    st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
    show(viz.plot_contamination_sweep(res.sweeps, {n: res.recs[n]["max_f1"]["contamination"] for n in res.models}))
    sw_model = st.selectbox("Sweep table", list(res.models), index=list(res.models).index(h["model"]))
    st.dataframe(res.sweeps[sw_model].drop(columns=["tn", "tp"]).style.format(
        {"contamination": "{:.4f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "fpr": "{:.3%}",
         "threshold": "{:.4f}", "flagged_share": "{:.3%}"}), use_container_width=True)
    show(viz.plot_runtime(res.models))

# ------------------------------------------------------------------ AI analyst
with tab_ai:
    mdl = h["model"]
    mm = res.models[mdl]
    st.markdown(f"#### Review the highest-scoring alerts · `{mdl}`")
    order = mm.scores_test.argsort()[::-1][:25]
    alerts = pd.DataFrame({
        "row": order, "score": mm.scores_test[order],
        "percentile": [(mm.scores_calib < v).mean() * 100 for v in mm.scores_test[order]],
        "Amount $": s.raw_test.Amount.to_numpy()[order],
        "hour": ((s.raw_test.Time.to_numpy()[order] % 86400) / 3600).round(1),
        "is_fraud (label)": s.y_test[order]})
    st.dataframe(alerts.style.format({"score": "{:.3f}", "percentile": "{:.2f}", "Amount $": "{:,.2f}"}),
                 use_container_width=True, height=260)
    pick = st.selectbox("Pick an alert to explain", alerts["row"].tolist(),
                        format_func=lambda i: f"row {i} · score {mm.scores_test[i]:.3f} · ${s.raw_test.Amount[i]:,.2f}")
    if st.button("Explain with NVIDIA AI"):
        row = s.X_test.iloc[pick]
        pct = float((mm.scores_calib < mm.scores_test[pick]).mean() * 100)
        case = {"model": mdl, "anomaly_score": round(float(mm.scores_test[pick]), 4),
                "score_percentile": round(pct, 3), "risk_band": risk_band(pct),
                "amount": float(s.raw_test.Amount[pick]), "hour_of_day": round(float(s.raw_test.Time[pick] % 86400 / 3600), 1),
                "deviations": feature_deviations(row, s.X_fit)}
        with st.spinner("Consulting the analyst…"):
            text, src = analyst.explain_transaction(case)
        st.markdown(f'<div class="ai-box {"nv" if src == "nvidia" else ""}">{text}</div>', unsafe_allow_html=True)
        st.caption(f"Source: {'NVIDIA · ' + analyst.model if src == 'nvidia' else 'offline rule-based fallback'}"
                   + (f" · {analyst.last_error}" if analyst.last_error else ""))
    st.markdown("#### Ask about the results")
    q = st.text_input("e.g. “Should we favour recall or precision for a card-not-present channel?”")
    if q:
        ctx = {"headline": h, "eda": e, "recommendations": {n: {k: (v or {}).get("contamination") for k, v in r_.items()}
                                                            for n, r_ in res.recs.items()}}
        with st.spinner("Thinking…"):
            text, src = analyst.chat(q, ctx)
        st.markdown(f'<div class="ai-box {"nv" if src == "nvidia" else ""}">{text}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ report
with tab_rep:
    if st.button("Generate report"):
        with st.spinner("Writing report…"):
            md = write_report(res, REPORT_DIR / "Performance_Analysis_Report.md", analyst)
        st.session_state.report_md = md
    md = st.session_state.get("report_md")
    if md:
        st.download_button("Download .md", md, "Performance_Analysis_Report.md", "text/markdown")
        st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", md))   # figures live in the other tabs
        st.caption("Figures are embedded in the saved report file (`reports/figures/`).")
    else:
        st.caption("Generates `reports/Performance_Analysis_Report.md` with this run's numbers.")
