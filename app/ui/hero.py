from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

_HERO_HEIGHT = 340


def render_hero(nvidia_online: bool) -> None:
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: transparent; font-family: 'Manrope', 'Segoe UI', sans-serif; overflow: hidden; }}
  #hero {{
    position: relative; width: 100%; height: {_HERO_HEIGHT}px;
    border: 1px solid #243055; border-radius: 22px; overflow: hidden;
    background: linear-gradient(135deg, rgba(18,26,51,.96), rgba(11,16,32,.96));
  }}
  canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  .content {{
    position: relative; z-index: 2; padding: 36px 40px; height: 100%;
    display: flex; flex-direction: column; justify-content: center;
  }}
  h1 {{
    font-family: 'Sora', 'Segoe UI', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: #E6EAF5; margin: 0 0 10px 0; letter-spacing: -0.02em;
  }}
  p {{ color: #8A94B8; font-size: 1.05rem; max-width: 600px; margin: 0 0 18px 0; line-height: 1.55; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{
    padding: 5px 13px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    border: 1px solid #243055; color: #E6EAF5; background: rgba(108,140,255,.10); white-space: nowrap;
  }}
  .chip.live {{ border-color: rgba(61,219,176,.5); color: #3DDBB0; background: rgba(61,219,176,.08); }}
  .chip.off  {{ border-color: rgba(255,92,122,.4);  color: #FF5C7A; background: rgba(255,92,122,.07); }}
  #hero.static::before {{
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle, rgba(108,140,255,.55) 1px, transparent 1.4px) 0 0 / 22px 22px;
    opacity: .18;
  }}
</style>
</head>
<body>
<div id="hero">
  <canvas id="c"></canvas>
  <div class="content">
    <h1>SentinelPay</h1>
    <p>Teach the model what <em>normal</em> looks like. Everything that deviates gets flagged \u2014 in milliseconds.</p>
    <div class="chips">
      <span class="chip">Isolation Forest</span>
      <span class="chip">Local Outlier Factor</span>
      <span class="chip">RobustScaler</span>
      <span class="chip">PR-curve tuning</span>
      {"<span class='chip live'>\u25cf NVIDIA analyst online</span>" if nvidia_online else "<span class='chip off'>\u25cb NVIDIA analyst offline</span>"}
    </div>
  </div>
</div>
<script>
(function() {{
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hero = document.getElementById('hero');
  if (reduced) {{ hero.classList.add('static'); return; }}

  const c = document.getElementById('c');
  const x = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  let W, H;

  function resize() {{
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * dpr; c.height = H * dpr;
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
  resize();
  window.addEventListener('resize', resize);

  const TOTAL = 380;
  const ANOMALY_RATE = 0.015;
  const particles = Array.from({{ length: TOTAL }}, () => ({{
    x: Math.random(), y: Math.random(),
    vx: (Math.random() - 0.5) * 3.5e-4,
    vy: (Math.random() - 0.5) * 3.5e-4,
    r: Math.random() * 1.5 + 0.4,
    anom: Math.random() < ANOMALY_RATE,
    phase: Math.random() * Math.PI * 2,
  }}));

  let t = 0;

  function draw() {{
    t++;
    x.clearRect(0, 0, W, H);
    const sw = ((t * 0.004) % 1.3) - 0.15;

    for (const p of particles) {{
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > 1) p.vx *= -1;
      if (p.y < 0 || p.y > 1) p.vy *= -1;
      const px = p.x * W, py = p.y * H;

      if (p.anom) {{
        const pulse = 0.5 + 0.5 * Math.sin(t * 0.05 + p.phase);
        x.shadowColor = '#FFB547'; x.shadowBlur = 12;
        x.fillStyle = `rgba(255,181,71,${{0.55 + 0.45 * pulse}})`;
        x.beginPath(); x.arc(px, py, 2.8 + pulse * 1.6, 0, Math.PI * 2); x.fill();
        x.shadowBlur = 0;
      }} else {{
        const nearSweep = Math.max(0, 1 - Math.abs(p.x - sw) * 7);
        x.fillStyle = `rgba(108,140,255,${{0.22 + 0.55 * nearSweep}})`;
        x.beginPath(); x.arc(px, py, p.r, 0, Math.PI * 2); x.fill();
      }}
    }}

    if (sw > -0.1 && sw < 1.1) {{
      const grd = x.createLinearGradient((sw - 0.06) * W, 0, (sw + 0.06) * W, 0);
      grd.addColorStop(0, 'rgba(255,181,71,0)');
      grd.addColorStop(0.5, 'rgba(255,181,71,0.09)');
      grd.addColorStop(1, 'rgba(255,181,71,0)');
      x.fillStyle = grd; x.fillRect(0, 0, W, H);
    }}
    requestAnimationFrame(draw);
  }}
  draw();
}})();
</script>
</body>
</html>
"""
    components.html(html, height=_HERO_HEIGHT + 8, scrolling=False)


def render_pre_run_cards() -> None:
    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px;">
  <div class="glass fade-up d1" style="padding:22px 24px;">
    <div style="font-size:1.4rem;margin-bottom:10px;">\U0001f332</div>
    <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#E6EAF5;margin-bottom:6px;">Isolation Forest</div>
    <div style="color:#8A94B8;font-size:.88rem;line-height:1.5;">
      Randomly partitions the feature space. Anomalies are isolated in fewer splits \u2014 giving them shorter average path lengths and higher scores.
    </div>
  </div>
  <div class="glass fade-up d2" style="padding:22px 24px;">
    <div style="font-size:1.4rem;margin-bottom:10px;">\U0001f52d</div>
    <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#E6EAF5;margin-bottom:6px;">Local Outlier Factor</div>
    <div style="color:#8A94B8;font-size:.88rem;line-height:1.5;">
      Compares each point\u2019s local density to its neighbours\u2019. Points in sparse neighbourhoods relative to their peers score as outliers.
    </div>
  </div>
  <div class="glass fade-up d3" style="padding:22px 24px;">
    <div style="font-size:1.4rem;margin-bottom:10px;">\U0001f916</div>
    <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#E6EAF5;margin-bottom:6px;">NVIDIA AI Analyst</div>
    <div style="color:#8A94B8;font-size:.88rem;line-height:1.5;">
      Explains each alert in plain English using a hosted LLM \u2014 with an offline fallback so nothing breaks without an API key.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
