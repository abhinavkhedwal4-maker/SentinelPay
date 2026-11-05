from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Manrope:wght@400;500;600&display=swap');

:root {
  --bg:    #0B1020;
  --panel: #121A33;
  --line:  #243055;
  --text:  #E6EAF5;
  --muted: #8A94B8;
  --amber: #FFB547;
  --blue:  #6C8CFF;
  --coral: #FF5C7A;
  --mint:  #3DDBB0;
}

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif !important; letter-spacing: -.01em; }

.stApp {
  background:
    radial-gradient(1100px 520px at 85% -8%,  rgba(108,140,255,.16), transparent 60%),
    radial-gradient(800px  400px at -5% 105%,  rgba(255,181,71,.07),  transparent 60%),
    var(--bg);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; max-width: 1320px; }

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

.glass {
  background: rgba(18, 26, 51, .72);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 20px;
}

@property --ang { syntax: '<angle>'; inherits: false; initial-value: 0deg; }

.glow-border {
  position: relative;
  border-radius: 20px;
  background: rgba(18, 26, 51, .78);
}
.glow-border::before {
  content: "";
  position: absolute;
  inset: -1px;
  border-radius: 21px;
  padding: 1px;
  z-index: -1;
  background: conic-gradient(
    from var(--ang),
    transparent 0 60%,
    #FFB547 80%,
    #6C8CFF 92%,
    transparent 100%
  );
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
}

.kpi {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 20px;
}
.kpi .v { font-family: 'Sora'; font-size: 1.85rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.kpi .l { color: var(--muted); font-size: .85rem; margin-top: 6px; }
.kpi.warn .v { color: var(--amber); }
.kpi.bad  .v { color: var(--coral); }
.kpi.good .v { color: var(--mint);  }

.hero {
  position: relative; overflow: hidden;
  border: 1px solid var(--line); border-radius: 22px;
  padding: 34px 38px; margin-bottom: 22px;
  background: linear-gradient(135deg, rgba(18,26,51,.92), rgba(11,16,32,.92));
}
.hero:before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(circle, rgba(108,140,255,.55) 1px, transparent 1.4px) 0 0 / 22px 22px;
  opacity: .22;
  mask-image: linear-gradient(90deg, transparent 25%, #000 85%);
  -webkit-mask-image: linear-gradient(90deg, transparent 25%, #000 85%);
}
.hero h1 { margin: 0; font-size: 2.7rem; font-weight: 800; color: var(--text); }
.hero p  { margin: .5rem 0 0; color: var(--muted); font-size: 1.05rem; max-width: 640px; }
.pill {
  display: inline-block; margin: 14px 8px 0 0; padding: 5px 13px;
  border-radius: 999px; font-size: .78rem; font-weight: 600;
  border: 1px solid var(--line); color: var(--text); background: rgba(108,140,255,.10);
}
.pill.live { border-color: rgba(61,219,176,.5); color: var(--mint);  background: rgba(61,219,176,.08); }
.pill.off  { border-color: rgba(255,92,122,.4);  color: var(--coral); background: rgba(255,92,122,.07); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--amber) !important; background: rgba(255,181,71,.07); }
.stButton > button {
  border-radius: 12px; border: 1px solid var(--amber);
  background: rgba(255,181,71,.10); color: var(--amber); font-weight: 700;
}
.ai-box {
  border: 1px solid var(--line); border-left: 3px solid var(--blue);
  border-radius: 12px; background: var(--panel); padding: 16px 20px;
}
.ai-box.nv { border-left-color: var(--mint); }
.muted { color: var(--muted); font-size: .85rem; }

.shimmer {
  background: linear-gradient(100deg, #121A33 30%, #1B2750 50%, #121A33 70%);
  background-size: 200% 100%;
  border-radius: 12px;
  min-height: 60px;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: none; }
}
.fade-up { animation: fadeUp .7s cubic-bezier(.2,.7,.2,1) both; }
.d1 { animation-delay: .05s; }
.d2 { animation-delay: .12s; }
.d3 { animation-delay: .19s; }
.d4 { animation-delay: .26s; }
.d5 { animation-delay: .33s; }
.d6 { animation-delay: .40s; }

@media (prefers-reduced-motion: reduce) {
  .hero:after      { animation: none; display: none; }
  .glow-border::before { animation: none; }
  .shimmer         { animation: none; background: var(--panel); }
  .fade-up         { animation: none; opacity: 1; transform: none; }
  .kpi, .stButton > button { transition: none; }
}

@media (prefers-reduced-motion: no-preference) {
  @keyframes spin { to { --ang: 360deg; } }
  .glow-border::before { animation: spin 7s linear infinite; }

  @keyframes sh { to { background-position: -200% 0; } }
  .shimmer { animation: sh 1.4s infinite; }

  .kpi { transition: border-color .2s, box-shadow .2s, transform .2s; }
  .kpi:hover {
    border-color: var(--amber);
    box-shadow: 0 0 0 1px rgba(255,181,71,.35), 0 10px 34px -12px rgba(255,181,71,.35);
    transform: translateY(-3px);
  }
  .stButton > button { transition: all .18s; }
  .stButton > button:hover {
    background: var(--amber); color: #0B1020;
    box-shadow: 0 8px 26px -8px rgba(255,181,71,.7);
  }
}

:focus-visible {
  outline: 2px solid var(--amber) !important;
  outline-offset: 2px !important;
}
button:focus-visible, [role="button"]:focus-visible,
a:focus-visible, input:focus-visible, select:focus-visible {
  outline: 2px solid var(--amber) !important;
  outline-offset: 2px !important;
}

@media (max-width: 900px) {
  .block-container { padding-left: 12px !important; padding-right: 12px !important; }
  .kpi .v { font-size: 1.4rem; }
}

.block-container { max-width: 1320px !important; margin: 0 auto; }

.stDataFrame { border-radius: 12px; overflow: hidden; }

.stExpander { border: 1px solid var(--line) !important; border-radius: 12px !important; }
.stExpander summary { color: var(--muted) !important; font-weight: 600 !important; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
