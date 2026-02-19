"""
UII v14.1 — dashboard.py
Local phase space visualizer. Run alongside the Triad in a second terminal.

Usage:
    python dashboard.py                          # watches mentat_triad_v14_log.jsonl
    python dashboard.py path/to/session.jsonl    # custom log path

Then open: http://localhost:5050
Polls the log file every 2 seconds. Safe to start before the Triad — waits for the file.
"""

import http.server
import json
import os
import sys
import math
from pathlib import Path
from collections import deque

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "mentat_triad_v14_log.jsonl"
PORT = 5050

# ─── LOG READER ───────────────────────────────────────────────────────────────

def read_log(path: str) -> dict:
    """Parse JSONL log into dashboard-ready payload."""
    p = Path(path)
    if not p.exists():
        return {"status": "waiting", "path": path, "steps": [], "session": {}, "anomalies": []}

    steps = []
    session_start = {}
    session_end = {}

    try:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = obj.get("event", "")
                if event == "step_log":
                    steps.append(obj)
                elif event == "session_start":
                    session_start = obj
                elif event == "session_end":
                    session_end = obj
    except Exception as e:
        return {"status": "error", "error": str(e), "steps": [], "session": {}, "anomalies": []}

    processed = process_steps(steps)
    anomalies = detect_anomalies(processed)

    return {
        "status": "live" if steps and not session_end else ("done" if session_end else "waiting"),
        "path": path,
        "steps": processed,
        "session_start": session_start,
        "session_end": session_end,
        "anomalies": anomalies,
    }


def process_steps(raw: list) -> list:
    """Derive dashboard fields from raw StepLog entries."""
    out = []
    phi_window = deque(maxlen=7)

    for i, s in enumerate(raw):
        phi_after = s.get("phi_after", 0.0) or 0.0
        phi_before = s.get("phi_before", phi_after) or phi_after
        phi_window.append(phi_after)

        # dΦ/dt smoothed over window
        vals = list(phi_window)
        dphi_smoothed = (vals[-1] - vals[0]) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0

        # Prediction error proxy: coupling confidence inverse, gated by observations
        coupling_conf = s.get("coupling_confidence", 0.0) or 0.0
        coupling_obs  = s.get("coupling_observations", 0) or 0
        pred_error = max(0.0, 1.0 - coupling_conf) if coupling_obs >= 10 else 0.5

        # Mutation magnitude: boundary_pressure-weighted (how hard the system is pushing)
        boundary = s.get("boundary_pressure", 0.0) or 0.0

        # CRK violation count and severity
        violations = s.get("crk_violations", []) or []
        crk_count = len(violations)
        crk_severity = sum(v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else 0 for v in violations)

        # Axis pressure
        discovered = s.get("discovered_axes", 0) or 0

        # Virtual mode
        virtual_active = s.get("virtual_mode_active", False)
        virtual_phi    = s.get("virtual_phi_predicted")
        model_fidelity = s.get("model_fidelity", 0.5) or 0.5

        # Reality contact: step where impossibility was NOT detected and LLM was not invoked
        is_micro_only = not s.get("llm_invoked", False) and not s.get("impossibility_detected", False)

        out.append({
            "step":            s.get("step", i),
            "phi":             round(phi_after, 5),
            "phi_before":      round(phi_before, 5),
            "dphi":            round(phi_after - phi_before, 5),
            "dphi_smoothed":   round(dphi_smoothed, 6),
            "pred_error":      round(pred_error, 4),
            "boundary":        round(boundary, 4),
            "crk_count":       crk_count,
            "crk_severity":    round(crk_severity, 4),
            "coupling_conf":   round(coupling_conf, 4),
            "coupling_obs":    coupling_obs,
            "action_map_size": s.get("action_map_affordances", 0) or 0,
            "discovered_axes": discovered,
            "virtual_active":  1 if virtual_active else 0,
            "virtual_phi":     round(virtual_phi, 4) if virtual_phi is not None else None,
            "model_fidelity":  round(model_fidelity, 4),
            "impossibility":   1 if s.get("impossibility_detected") else 0,
            "llm_invoked":     1 if s.get("llm_invoked") else 0,
            "is_micro_only":   1 if is_micro_only else 0,
            "mitosis":         1 if s.get("mitosis_triggered") else 0,
            "freeze":          1 if s.get("freeze_verified") else 0,
            "attractor":       s.get("attractor_status", ""),
            "tokens_step":     s.get("tokens_used_this_step", 0) or 0,
            "eno_active":      1 if s.get("eno_active") else 0,
            "residual_expl":   s.get("residual_explanation", ""),
        })

    return out


def detect_anomalies(steps: list) -> list:
    """Check invariants against recent window. Returns list of active alerts."""
    if len(steps) < 10:
        return []

    recent = steps[-15:]
    last   = steps[-1]
    alerts = []

    # Runaway curvature: std(dΦ/dt) too high over recent window
    dphis = [s["dphi_smoothed"] for s in recent]
    mean_d = sum(dphis) / len(dphis)
    std_d  = math.sqrt(sum((x - mean_d) ** 2 for x in dphis) / len(dphis))
    if std_d > 0.015:
        alerts.append({"id": "curvature", "label": "RUNAWAY CURVATURE", "level": "warn",
                        "detail": f"dΦ/dt std={std_d:.4f}"})

    # Silent drift: pred_error rising while boundary pressure flat
    if len(steps) >= 20:
        err_slope = _slope([s["pred_error"] for s in steps[-20:]])
        bnd_slope = _slope([s["boundary"] for s in steps[-20:]])
        if err_slope > 0.004 and bnd_slope < 0.001:
            alerts.append({"id": "drift", "label": "SILENT DRIFT", "level": "crit",
                            "detail": f"err slope={err_slope:.4f}, boundary flat"})

    # Dead adaptation: |dΦ/dt| ≈ 0 for last 10 steps
    flat_window = steps[-10:]
    flatness = sum(abs(s["dphi_smoothed"]) for s in flat_window) / len(flat_window)
    if flatness < 0.0008 and len(steps) > 15:
        alerts.append({"id": "dead", "label": "DEAD ADAPTATION", "level": "warn",
                        "detail": f"mean |dΦ/dt|={flatness:.5f}"})

    # Axis explosion: discovered_axes growing fast with low coupling_conf
    if last["discovered_axes"] > 6 and last["coupling_conf"] < 0.3:
        alerts.append({"id": "axisexp", "label": "AXIS EXPLOSION", "level": "crit",
                        "detail": f"axes={last['discovered_axes']}, conf={last['coupling_conf']:.3f}"})

    # Virtual-mode self-delusion: high virtual ratio + low fidelity
    recent_virtual = sum(s["virtual_active"] for s in recent) / len(recent)
    if recent_virtual > 0.6 and last["model_fidelity"] < 0.45:
        alerts.append({"id": "virtual", "label": "VIRTUAL DELUSION", "level": "crit",
                        "detail": f"virtual={recent_virtual:.2f}, fidelity={last['model_fidelity']:.3f}"})

    # Gain miscalibration: high error, low boundary response
    if last["pred_error"] > 0.55 and last["boundary"] < 0.1 and len(steps) > 20:
        alerts.append({"id": "gain", "label": "GAIN MISCALIBRATION", "level": "warn",
                        "detail": f"err={last['pred_error']:.3f}, boundary={last['boundary']:.3f}"})

    return alerts


def _slope(vals: list) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(vals) / n
    num = sum((xs[i] - mx) * (vals[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


# ─── HTTP HANDLER ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>UII Triad — Phase Space</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #080808;
  --panel:    #0D0D0D;
  --border:   #1C1C1C;
  --grid:     #111;
  --text:     #C8C8C8;
  --dim:      #484848;
  --amber:    #F0A000;
  --amber-d:  #7A5200;
  --green:    #00C87A;
  --red:      #FF3C3C;
  --blue:     #3C8CFF;
  --purple:   #9060D0;
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  height: 100%;
  overflow-x: hidden;
}

#root { display: flex; flex-direction: column; min-height: 100vh; }

/* ─── HEADER ─── */
#header {
  border-bottom: 1px solid var(--border);
  padding: 8px 18px;
  display: flex;
  align-items: center;
  gap: 24px;
  background: var(--panel);
  flex-shrink: 0;
}
#header .title { font-size: 11px; color: var(--amber); letter-spacing: 0.18em; font-weight: 700; }
#header .sub   { font-size: 9px;  color: var(--dim);   letter-spacing: 0.12em; margin-top: 2px; }
#header .meta  { margin-left: auto; display: flex; gap: 20px; align-items: center; }
#header .kv    { font-size: 9px; color: var(--dim); }
#header .kv span { color: var(--text); }
#status-dot    { width: 7px; height: 7px; border-radius: 50%; background: var(--dim); display: inline-block; margin-right: 6px; }
#status-dot.live { background: var(--green); animation: pulse 2s ease-in-out infinite; }
#status-dot.done { background: var(--amber); }
#status-dot.waiting { background: var(--dim); animation: blink 1.5s step-end infinite; }
#status-dot.error   { background: var(--red); }

/* ─── ALERT BAR ─── */
#alert-bar {
  border-bottom: 1px solid var(--border);
  padding: 5px 18px;
  min-height: 30px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  background: var(--bg);
}
.alert-chip {
  font-size: 9px;
  letter-spacing: 0.1em;
  padding: 2px 8px;
  border: 1px solid;
}
.alert-chip.crit { border-color: var(--red); color: var(--red); animation: blink 1.2s step-end infinite; }
.alert-chip.warn { border-color: var(--amber); color: var(--amber); }
.nominal { font-size: 9px; color: var(--green); letter-spacing: 0.1em; }

/* ─── GRID ─── */
#panels {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 200px 200px 180px;
  gap: 1px;
  background: var(--border);
  overflow: hidden;
}

/* Panel A spans full width */
#panel-a { grid-column: 1 / 3; }

.panel {
  background: var(--panel);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-header {
  padding: 5px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 9px;
  letter-spacing: 0.1em;
  color: var(--dim);
  flex-shrink: 0;
}
.panel-header .bullet { color: var(--amber); }
.panel-header .tag { margin-left: auto; color: var(--amber-d); font-size: 8px; }
.panel-body { flex: 1; position: relative; min-height: 0; }

canvas { display: block; }

/* ─── SCALARS ROW ─── */
#scalars {
  border-top: 1px solid var(--border);
  padding: 8px 18px;
  display: flex;
  gap: 0;
  background: var(--panel);
  flex-shrink: 0;
}
.scalar {
  flex: 1;
  border-right: 1px solid var(--border);
  padding: 4px 12px;
  min-width: 0;
}
.scalar:last-child { border-right: none; }
.scalar .slabel { font-size: 8px; color: var(--dim); letter-spacing: 0.1em; margin-bottom: 3px; }
.scalar .svalue { font-size: 18px; font-weight: 700; line-height: 1; color: var(--amber); }
.scalar .svalue.ok   { color: var(--green); }
.scalar .svalue.bad  { color: var(--red); }
.scalar .svalue.dim  { color: var(--dim); }
.scalar .sunit { font-size: 9px; color: var(--dim); margin-left: 3px; }

/* ─── FOOTER ─── */
#footer {
  border-top: 1px solid var(--border);
  padding: 5px 18px;
  font-size: 8px;
  color: var(--dim);
  display: flex;
  gap: 20px;
  background: var(--bg);
  flex-shrink: 0;
}
#footer .event-log { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

@keyframes pulse { 0%,100%{opacity:.6} 50%{opacity:1} }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
</style>
</head>
<body>
<div id="root">

  <!-- HEADER -->
  <div id="header">
    <div>
      <div class="title">TRIAD TELEMETRY</div>
      <div class="sub">UII v14.1 — PHASE SPACE VISUALIZATION</div>
    </div>
    <div class="meta">
      <div class="kv"><span id="status-dot" class="waiting"></span><span id="status-text">WAITING</span></div>
      <div class="kv">STEP <span id="hdr-step">—</span></div>
      <div class="kv">Φ <span id="hdr-phi">—</span></div>
      <div class="kv">dΦ/dt <span id="hdr-dphi">—</span></div>
      <div class="kv">FIDELITY <span id="hdr-fidelity">—</span></div>
    </div>
  </div>

  <!-- ALERT BAR -->
  <div id="alert-bar"><span class="nominal">● WAITING FOR DATA</span></div>

  <!-- PANELS -->
  <div id="panels">
    <div class="panel" id="panel-a">
      <div class="panel-header">
        <span class="bullet">■</span>
        PROJECTION A — Φ GRADIENT
        <span class="tag">dΦ/dt smoothed · event-marked</span>
      </div>
      <div class="panel-body"><canvas id="canvas-a"></canvas></div>
    </div>

    <div class="panel" id="panel-b">
      <div class="panel-header">
        <span class="bullet">■</span>
        PROJECTION B — ERROR / MUTATION
        <span class="tag">control systems view</span>
      </div>
      <div class="panel-body"><canvas id="canvas-b"></canvas></div>
    </div>

    <div class="panel" id="panel-c">
      <div class="panel-header">
        <span class="bullet">■</span>
        PROJECTION C — AXIS PRESSURE
        <span class="tag">dimensional growth</span>
      </div>
      <div class="panel-body"><canvas id="canvas-c"></canvas></div>
    </div>

    <div class="panel" id="panel-d">
      <div class="panel-header">
        <span class="bullet">■</span>
        REALITY CONTACT
        <span class="tag">virtual ratio · model fidelity</span>
      </div>
      <div class="panel-body"><canvas id="canvas-d"></canvas></div>
    </div>
  </div>

  <!-- SCALARS -->
  <div id="scalars">
    <div class="scalar">
      <div class="slabel">Φ CURRENT</div>
      <div class="svalue" id="sc-phi">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">dΦ/dt</div>
      <div class="svalue" id="sc-dphi">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">PRED ERROR</div>
      <div class="svalue" id="sc-err">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">COUPLING CONF</div>
      <div class="svalue" id="sc-coup">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">MODEL FIDELITY</div>
      <div class="svalue" id="sc-fid">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">AXES (P / A)</div>
      <div class="svalue" id="sc-axes">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">VIRTUAL RATIO</div>
      <div class="svalue" id="sc-virt">—</div>
    </div>
    <div class="scalar">
      <div class="slabel">TOKENS THIS STEP</div>
      <div class="svalue" id="sc-tok">—</div>
    </div>
  </div>

  <!-- FOOTER -->
  <div id="footer">
    <div id="log-path">LOG: """ + LOG_PATH + r"""</div>
    <div class="event-log" id="event-log">—</div>
    <div id="poll-time">—</div>
  </div>

</div>

<script>
// ─── COLOURS ───────────────────────────────────────────────────────────────
const C = {
  bg:     '#080808', panel:  '#0D0D0D', border: '#1C1C1C',
  grid:   '#111111', text:   '#C8C8C8', dim:    '#484848',
  amber:  '#F0A000', amberD: '#7A5200', green:  '#00C87A',
  red:    '#FF3C3C', blue:   '#3C8CFF', purple: '#9060D0',
};

// ─── CANVAS CHART HELPERS ──────────────────────────────────────────────────
function resizeCanvas(canvas) {
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = parent.clientWidth, h = parent.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width  = w + 'px';
    canvas.style.height = h + 'px';
  }
  return { ctx: canvas.getContext('2d'), w, h, dpr };
}

function drawChart(canvas, series, opts = {}) {
  const { ctx, w, h, dpr } = resizeCanvas(canvas);
  ctx.resetTransform();
  ctx.scale(dpr, dpr);

  const PAD = { t: 10, r: 8, b: 22, l: 46 };
  const CW = w - PAD.l - PAD.r;
  const CH = h - PAD.t - PAD.b;

  ctx.fillStyle = C.panel;
  ctx.fillRect(0, 0, w, h);

  if (!series || series.length === 0 || series[0].data.length === 0) {
    ctx.fillStyle = C.dim;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText('WAITING FOR DATA', PAD.l + 8, h / 2);
    return;
  }

  // Compute y domain across all series
  let yMin = opts.yMin ?? Infinity, yMax = opts.yMax ?? -Infinity;
  for (const s of series) {
    for (const v of s.data) {
      if (v == null || isNaN(v)) continue;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (yMin === yMax) { yMin -= 0.01; yMax += 0.01; }
  if (opts.yMin !== undefined) yMin = opts.yMin;
  if (opts.yMax !== undefined) yMax = opts.yMax;
  const yRange = yMax - yMin;

  const xCount = series[0].data.length;

  function px(i) { return PAD.l + (i / Math.max(xCount - 1, 1)) * CW; }
  function py(v) { return PAD.t + CH - ((v - yMin) / yRange) * CH; }

  // Grid
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y = PAD.t + (i / gridLines) * CH;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + CW, y); ctx.stroke();
    const val = yMax - (i / gridLines) * yRange;
    ctx.fillStyle = C.dim;
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(3), PAD.l - 4, y + 3);
  }

  // Zero line if range crosses 0
  if (yMin < 0 && yMax > 0) {
    const y0 = py(0);
    ctx.strokeStyle = C.dim;
    ctx.setLineDash([2, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, y0); ctx.lineTo(PAD.l + CW, y0); ctx.stroke();
    ctx.setLineDash([]);
  }

  // X axis labels
  ctx.fillStyle = C.dim;
  ctx.font = '8px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const labelCount = Math.min(8, xCount);
  for (let i = 0; i < labelCount; i++) {
    const idx = Math.round((i / (labelCount - 1)) * (xCount - 1));
    ctx.fillText(idx, px(idx), h - 5);
  }

  // Event ref lines
  if (opts.events) {
    for (const ev of opts.events) {
      const xi = ev.step;
      if (xi < 0 || xi >= xCount) continue;
      const x = px(xi);
      ctx.strokeStyle = ev.color;
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + CH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      if (ev.label) {
        ctx.fillStyle = ev.color;
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(ev.label, x, PAD.t + 8);
      }
    }
  }

  // Series
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 1.5;
    if (s.dash) ctx.setLineDash(s.dash);
    ctx.globalAlpha = s.alpha ?? 1;

    if (s.fill) {
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < s.data.length; i++) {
        const v = s.data[i];
        if (v == null || isNaN(v)) { first = true; continue; }
        const x = px(i), y = py(v);
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.lineTo(px(s.data.length - 1), py(Math.max(yMin, 0)));
      ctx.lineTo(PAD.l, py(Math.max(yMin, 0)));
      ctx.closePath();
      ctx.fillStyle = s.fill;
      ctx.globalAlpha = 0.15;
      ctx.fill();
      ctx.globalAlpha = s.alpha ?? 1;
    }

    ctx.beginPath();
    let first = true;
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (v == null || isNaN(v)) { first = true; continue; }
      const x = px(i), y = py(v);
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // Clip box
  ctx.strokeStyle = C.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD.l, PAD.t, CW, CH);
}

// ─── RENDER ───────────────────────────────────────────────────────────────
function render(payload) {
  const steps = payload.steps || [];
  const n = steps.length;
  const last = n > 0 ? steps[n - 1] : null;
  const anomalies = payload.anomalies || [];

  // Status dot
  const dot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const status = payload.status || 'waiting';
  dot.className = status;
  statusText.textContent = status.toUpperCase();

  if (n === 0) return;

  // Header scalars
  const dphi = last.dphi_smoothed;
  document.getElementById('hdr-step').textContent = last.step;
  document.getElementById('hdr-phi').textContent = last.phi.toFixed(4);
  document.getElementById('hdr-dphi').textContent = dphi.toFixed(5);
  document.getElementById('hdr-fidelity').textContent = last.model_fidelity.toFixed(3);

  // Alert bar
  const alertBar = document.getElementById('alert-bar');
  alertBar.innerHTML = '';
  if (anomalies.length === 0) {
    alertBar.innerHTML = '<span class="nominal">● ALL INVARIANTS NOMINAL</span>';
  } else {
    for (const a of anomalies) {
      const chip = document.createElement('span');
      chip.className = `alert-chip ${a.level}`;
      chip.textContent = `${a.level === 'crit' ? '⚠' : '◈'} ${a.label}`;
      chip.title = a.detail || '';
      alertBar.appendChild(chip);
    }
  }

  // Extract series
  const phi     = steps.map(s => s.phi);
  const dphi_s  = steps.map(s => s.dphi_smoothed);
  const err     = steps.map(s => s.pred_error);
  const bnd     = steps.map(s => s.boundary);
  const coup    = steps.map(s => s.coupling_conf);
  const axes    = steps.map(s => s.discovered_axes);
  const virt    = steps.map(s => s.virtual_active);
  const fid     = steps.map(s => s.model_fidelity);

  // Events for ref lines
  const events = steps.reduce((acc, s, i) => {
    if (s.impossibility)  acc.push({ step: i, color: C.red,   label: '!' });
    if (s.mitosis)        acc.push({ step: i, color: C.purple, label: 'M' });
    if (s.freeze)         acc.push({ step: i, color: C.green,  label: '■' });
    return acc;
  }, []);

  // ── Canvas A: dΦ/dt ──
  drawChart(document.getElementById('canvas-a'), [
    { data: dphi_s, color: C.amber, width: 1.5, fill: C.amber },
    { data: phi,    color: C.dim,   width: 1,   dash: [2, 4] },
  ], { events });

  // ── Canvas B: pred_error + coupling_conf ──
  drawChart(document.getElementById('canvas-b'), [
    { data: err,  color: C.red,   width: 1.5 },
    { data: coup, color: C.green, width: 1.5, dash: [3, 2] },
    { data: bnd,  color: C.blue,  width: 1,   alpha: 0.5 },
  ], { yMin: 0, yMax: 1, events });

  // ── Canvas C: axes + admitted proxy ──
  drawChart(document.getElementById('canvas-c'), [
    { data: axes, color: C.amber, width: 2, fill: C.amber },
  ], { yMin: 0, events });

  // ── Canvas D: virtual + fidelity ──
  drawChart(document.getElementById('canvas-d'), [
    { data: fid,  color: C.green, width: 1.5 },
    { data: virt, color: C.amber, width: 1.5, dash: [2, 3], fill: C.amber },
  ], { yMin: 0, yMax: 1, events });

  // ── Bottom scalars ──
  function setScalar(id, val, good, bad) {
    const el = document.getElementById(id);
    if (val == null) { el.textContent = '—'; el.className = 'svalue dim'; return; }
    el.textContent = typeof val === 'number' ? val.toFixed(3) : val;
    el.className = 'svalue' + (bad ? ' bad' : good ? ' ok' : '');
  }

  setScalar('sc-phi',  last.phi,           last.phi > 0.5,  last.phi < 0.2);
  setScalar('sc-dphi', dphi,               dphi > 0.002,    Math.abs(dphi) > 0.03);
  setScalar('sc-err',  last.pred_error,    last.pred_error < 0.3, last.pred_error > 0.55);
  setScalar('sc-coup', last.coupling_conf, last.coupling_conf > 0.6, last.coupling_conf < 0.2);
  setScalar('sc-fid',  last.model_fidelity, last.model_fidelity > 0.7, last.model_fidelity < 0.4);

  const axesEl = document.getElementById('sc-axes');
  axesEl.textContent = `${last.discovered_axes}`;
  axesEl.className = 'svalue' + (last.discovered_axes > 6 ? ' bad' : last.discovered_axes > 0 ? ' ok' : '');

  const recentVirt = steps.slice(-15).reduce((a, s) => a + s.virtual_active, 0) / Math.min(15, steps.length);
  setScalar('sc-virt', recentVirt, recentVirt < 0.4, recentVirt > 0.6 && last.model_fidelity < 0.45);
  setScalar('sc-tok', last.tokens_step, false, last.tokens_step > 3000);

  // Footer event log
  const lastEvent = [...events].reverse()[0];
  if (lastEvent) {
    const s = steps[lastEvent.step];
    const label = s.impossibility ? `STEP ${s.step}: IMPOSSIBILITY — ${s.attractor}` :
                  s.mitosis ? `STEP ${s.step}: MITOSIS` :
                  s.freeze  ? `STEP ${s.step}: FREEZE VERIFIED` : '';
    document.getElementById('event-log').textContent = label;
  }

  document.getElementById('poll-time').textContent = `POLLED ${new Date().toLocaleTimeString()}`;
}

// ─── POLL ─────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/data');
    const payload = await r.json();
    render(payload);
    const interval = payload.status === 'done' ? 10000 : 2000;
    setTimeout(poll, interval);
  } catch (e) {
    console.error(e);
    setTimeout(poll, 3000);
  }
}

window.addEventListener('resize', () => {
  // Re-render on resize if we have data
  fetch('/data').then(r => r.json()).then(render).catch(() => {});
});

poll();
</script>
</body>
</html>"""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            payload = read_log(LOG_PATH)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/index.html":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress request logs


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"  UII Triad Dashboard")
    print(f"  Watching: {LOG_PATH}")
    print(f"  Open:     http://localhost:{PORT}")
    print(f"  Ctrl-C to stop")
    print()

    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")