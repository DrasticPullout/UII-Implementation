#!/usr/bin/env python3
"""
extract_ledger.py — Ledger Inspection Utility for UII v16

Replaces extract_genome_v14_1.py.

Primary use: inspect ledger state between sessions.

If MentatTriad.run() calls _end_session() correctly, the ledger is already
written at session end. extract_ledger.py is a diagnostic/inspection tool,
not a required pipeline step.

Usage:
    python extract_ledger.py [--ledger PATH] [--out PATH] [--validate]

    --ledger PATH   Path to ledger.json (default: ledger.json)
    --out    PATH   Write ledger back to PATH (optional; default: no write)
    --validate      Run structural integrity checks (optional)

Workflow:
    python uii_triad.py [args]                    # writes ledger.json
    python extract_ledger.py --ledger ledger.json  # inspect
    python uii_triad.py --load-ledger ledger.json  # next run
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# ── UII imports ────────────────────────────────────────────────────────────────
# extract_ledger imports from uii_ledger so load_ledger / save_ledger and their
# print logic are reused. If UII modules are unavailable (standalone inspection),
# fall back to raw JSON read and inline print.
try:
    from uii_ledger import load_ledger, save_ledger, TriadLedger
    _HAS_UII = True
except ImportError:
    _HAS_UII = False


# ──────────────────────────────────────────────────────────────────────────────
# Constants (mirrored from uii_geometry.py — keep in sync)
# ──────────────────────────────────────────────────────────────────────────────

INTERFACE_COUPLED_SIGNALS = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity',
}
# api_llm replaces DeathClock in v16: resource pressure sensed via this channel
RESOURCE_CHANNELS = {'api_llm', 'resource_cpu', 'resource_memory', 'process_self'}
MIN_COUPLING_OBS            = 50
MIN_ACTION_MAP_AFFORDANCES  = 3


# ──────────────────────────────────────────────────────────────────────────────
# Standalone JSON fallback (no uii imports available)
# ──────────────────────────────────────────────────────────────────────────────

def _read_raw(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Ledger file not found: {path}")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to read ledger: {exc}")
        return None


def _write_raw(data: Dict, path: str) -> bool:
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except OSError as exc:
        print(f"[ERROR] Failed to write {path}: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Validation checks (structural integrity)
# ──────────────────────────────────────────────────────────────────────────────

def validate_ledger(data: Dict) -> List[str]:
    """
    V1  hessian_snapshot present and non-empty
    V2  eigenvalue count == channel_basis.n_dims
    V3  stored vol_opt matches recomputed value from eigenvalues
    V4  operator_snapshot.compression.observation_count > 0
    V5  No interface-coupled signals in discovered_structure keys
    V6  H matrix rows == channel_basis.n_dims
    """
    issues = []
    hs = data.get('hessian_snapshot', {})

    if not hs:
        issues.append(
            "V1 FAIL: hessian_snapshot empty — "
            "PeakOptionalityTracker.update() never fired "
            "(session too short or all steps had no active channels)")
        return issues

    eigenvalues = hs.get('eigenvalues', [])
    cb          = hs.get('channel_basis', {})
    n_dims      = cb.get('n_dims', 0)

    if len(eigenvalues) != n_dims:
        issues.append(
            f"V2 WARN: eigenvalue count {len(eigenvalues)} != "
            f"channel_basis.n_dims {n_dims}")

    if eigenvalues:
        ev               = np.array(eigenvalues, dtype=float)
        stored_vol_opt   = float(hs.get('vol_opt', 0.0))
        recomputed       = float(np.sum(ev[ev > 0]))
        if abs(stored_vol_opt - recomputed) > 1e-6:
            issues.append(
                f"V3 WARN: stored vol_opt {stored_vol_opt:.6f} != "
                f"recomputed {recomputed:.6f}")

    obs = int((data.get('operator_snapshot', {})
                   .get('compression', {})
                   .get('observation_count', 0)))
    if obs == 0:
        issues.append(
            "V4 WARN: operator_snapshot.compression.observation_count == 0")

    discovered   = data.get('discovered_structure', {})
    contaminated = [k for k in discovered
                    if any(s in k for s in INTERFACE_COUPLED_SIGNALS)]
    if contaminated:
        issues.append(
            f"V5 CRITICAL: interface contamination in discovered_structure: "
            f"{contaminated} — AxisAdmissionTest should have blocked these")

    matrix = hs.get('matrix', [])
    if matrix and len(matrix) != n_dims:
        issues.append(
            f"V6 WARN: H matrix rows {len(matrix)} != channel_basis.n_dims {n_dims}")

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# Extended diagnostics (extra detail beyond load_ledger summary)
# ──────────────────────────────────────────────────────────────────────────────

def _print_extended(data: Dict, issues: List[str]):
    hs         = data.get('hessian_snapshot',    {})
    ops        = data.get('operator_snapshot',   {})
    causal     = data.get('causal_model',        {})
    discovered = data.get('discovered_structure', {})
    eigenvalues = hs.get('eigenvalues', [])

    if eigenvalues:
        ev  = np.array(eigenvalues, dtype=float)
        pos = int(np.sum(ev > 0))
        neg = int(np.sum(ev < 0))
        zer = len(ev) - pos - neg
        cond = (float(abs(ev.max()) / max(abs(ev.min()), 1e-12))
                if len(ev) else 0.0)
        cond_str = f'{cond:.1f}' if cond < 1e6 else f'{cond:.2e}  ← ill-conditioned'
        print(f"  Eigenspectrum:      {pos}+ / {neg}- / {zer}≈0   "
              f"cond={cond_str}")

    # Resource channels
    s_snap = ops.get('sensing', {}).get('channels', {})
    for ch in sorted(RESOURCE_CHANNELS & set(s_snap)):
        cov  = s_snap[ch].get('coverage',    0.0)
        rate = s_snap[ch].get('signal_rate', 0.0)
        print(f"  {ch:20s}  coverage={cov:.3f}  signal_rate={rate:.4f}")

    # SMO
    smo = ops.get('smo', {})
    if smo:
        print(f"  SMO plasticity:     {smo.get('plasticity', 0.5):.3f}  "
              f"rigidity={smo.get('rigidity', 0.5):.3f}")

    # Action map sample
    action_map = causal.get('action_substrate_map', {})
    if action_map:
        print(f"\n  Action substrate map ({len(action_map)} affordances):")
        for aff in sorted(action_map)[:6]:
            d = action_map[aff]
            print(f"    {aff:15s}  S={d.get('S',0):+.3f}  I={d.get('I',0):+.3f}  "
                  f"P={d.get('P',0):+.3f}  A={d.get('A',0):+.3f}")
        if len(action_map) > 6:
            print(f"    ... ({len(action_map) - 6} more)")

    # Discovered structure
    admitted    = {k: v for k, v in discovered.items()
                   if v.get('status', 'admitted') == 'admitted'}
    provisional = {k: v for k, v in discovered.items()
                   if v.get('status') == 'provisional'}
    print(f"\n  Discovered structure: {len(discovered)} axes  "
          f"({len(admitted)} admitted, {len(provisional)} provisional)")
    for k, v in admitted.items():
        print(f"    [admitted]    {k}  "
              f"evidence={v.get('total_evidence', 0):.1f}  "
              f"gens_seen={v.get('generations_seen', 0)}")
    for k, v in provisional.items():
        print(f"    [provisional] {k}  evidence={v.get('total_evidence', 0):.1f}")

    # Validation
    print(f"\n  Validation:")
    if not issues:
        print("    All structural checks passed. ✓")
    else:
        for issue in issues:
            marker = '✗' if ('CRITICAL' in issue or 'FAIL' in issue) else '⚠'
            print(f"    {marker} {issue}")

    coup_obs = int((causal.get('coupling_matrix') or {}).get('observations', 0))
    am_count = len(action_map)
    ev_ok    = bool(eigenvalues and float(np.sum(np.array(eigenvalues)[np.array(eigenvalues) > 0])) > 0)
    cont     = [k for k in discovered if any(s in k for s in INTERFACE_COUPLED_SIGNALS)]
    print(f"    Coupling ready:    {'✓' if coup_obs >= MIN_COUPLING_OBS else f'○ {coup_obs}/{MIN_COUPLING_OBS}'}")
    print(f"    Action map ready:  {'✓' if am_count >= MIN_ACTION_MAP_AFFORDANCES else f'○ {am_count}/{MIN_ACTION_MAP_AFFORDANCES}'}")
    print(f"    Hessian ready:     {'✓' if ev_ok else '○ no positive eigenvalues'}")
    print(f"    Interface clean:   {'✓' if not cont else f'✗ {cont}'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='extract_ledger.py — UII v16 Ledger Inspection Utility'
    )
    parser.add_argument(
        '--ledger', dest='ledger', default='ledger.json',
        help='Path to ledger.json (default: ledger.json)'
    )
    parser.add_argument(
        '--out', dest='out', default=None,
        help='Write ledger to this path (optional)'
    )
    parser.add_argument(
        '--validate', dest='validate', action='store_true',
        help='Run structural integrity checks'
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if _HAS_UII:
        # load_ledger prints [LOADED LEDGER] summary automatically
        ledger, *_ = load_ledger(args.ledger)
        data = {
            'hessian_snapshot':    ledger.hessian_snapshot,
            'operator_snapshot':   ledger.operator_snapshot,
            'causal_model':        ledger.causal_model,
            'discovered_structure': ledger.discovered_structure,
        }
    else:
        # Standalone fallback: no uii modules available
        print(f"[extract_ledger] UII modules not found — standalone JSON read")
        data = _read_raw(args.ledger)
        if data is None:
            sys.exit(1)
        # Print inline summary matching [LOADED LEDGER] format
        hs  = data.get('hessian_snapshot', {})
        ops = data.get('operator_snapshot', {})
        cm  = data.get('causal_model', {})
        cb  = hs.get('channel_basis', {})
        cg  = ops.get('compression', {}).get('causal_graph', {})
        coh = ops.get('coherence', {}).get('consistency', {})
        print(
            f"\n[LOADED LEDGER]\n"
            f"  Vol_opt:            {hs.get('vol_opt', 0.0):.4f}\n"
            f"  Phi at peak:        {hs.get('phi', 0.0):.4f}\n"
            f"  Channel basis dims: {cb.get('n_dims', 0)}\n"
            f"  Causal graph edges: {len(cg)}\n"
            f"  Observation count:  {ops.get('compression', {}).get('observation_count', 0)}\n"
            f"  Realized horizon:   {ops.get('prediction', {}).get('realized_horizon', 0)}\n"
            f"  Loop closure:       {coh.get('loop_closure', 0.0):.3f}\n"
            f"  Coupling confidence:{cm.get('coupling_matrix', {}).get('confidence', 0.0):.3f}\n"
            f"  Action map:         {len(cm.get('action_substrate_map', {}))} affordances\n"
            f"  Discovered axes:    {len(data.get('discovered_structure', {}))}"
        )

    # ── Extended diagnostics + validation (--validate flag) ───────────────────
    if args.validate:
        issues = validate_ledger(data)
        print("\n[EXTENDED DIAGNOSTICS]")
        _print_extended(data, issues)
        if any('CRITICAL' in i or 'FAIL' in i for i in issues):
            print("\n[EXIT 2] Critical issues found.")
            sys.exit(2)

    # ── Write out (--out flag) ─────────────────────────────────────────────────
    if args.out:
        if _HAS_UII:
            from uii_ledger import TriadLedger
            out_ledger = TriadLedger(
                hessian_snapshot    = data['hessian_snapshot'],
                operator_snapshot   = data['operator_snapshot'],
                causal_model        = data['causal_model'],
                discovered_structure= data['discovered_structure'],
            )
            save_ledger(out_ledger, args.out)
        else:
            _write_raw(data, args.out)
        print(f"\nLedger written to {args.out}")

    print(f"\n  Next run: python uii_triad.py --load-ledger {args.ledger}")


if __name__ == '__main__':
    main()
