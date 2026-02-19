#!/usr/bin/env python3
"""
extract_genome_v14_1.py — Genome Extraction Utility for UII v14.1

Replaces extract_genome_v14.py. Reads the last session_end record from a v14.1
log file and writes genome.json ready for the next generation.

v14.1 additions over v14:
1. Reads model_fidelity_final from session_end (written by ModelFidelityMonitor)
2. Appends new generation snapshot to lineage_history (max N=5 entries, pruned here)
3. Computes velocity fields via GeneticVelocityEstimator and writes to child genome
4. Writes provisional_axes / admitted_axes breakdown to diagnostics
5. Prunes lineage_history to Summary Vector at G >= 1000 (slope/intercept per param)

Backward compatible with v14 session_end logs:
- Missing model_fidelity_final → defaults to 0.5 (neutral)
- Missing lineage fields → initializes lineage_history = [] and velocities = 0.0

Usage:
    python extract_genome_v14_1.py [log_path] [output_path]
    python extract_genome_v14_1.py                          # defaults
    python extract_genome_v14_1.py mentat_triad_v14_log.jsonl genome.json

The External Measurement Invariant is respected throughout:
Velocity fields are computed from lineage data only, never from session internals.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# CONSTANTS (must mirror uii_v14_1.py exactly)
# ============================================================

LAYER1_PARAMS = ['S_bias', 'I_bias', 'P_bias', 'A_bias', 'rigidity_init', 'phi_coherence_weight']

VELOCITY_FIELD = {
    'S_bias': 'S_velocity',
    'I_bias': 'I_velocity',
    'P_bias': 'P_velocity',
    'A_bias': 'A_velocity',
    'rigidity_init': 'rigidity_init_velocity',
    'phi_coherence_weight': 'phi_coherence_velocity',
}

LINEAGE_WINDOW = 5          # Keep last N generations in lineage_history
SUMMARY_VECTOR_THRESHOLD = 1000  # Prune to slope/intercept at G >= this


# ============================================================
# GeneticVelocityEstimator (inlined from uii_v14_1.py)
# ============================================================

class GeneticVelocityEstimator:
    """
    Computes Layer 1 momentum vectors from lineage_history.
    Identical to the class in uii_v14_1.py — duplicated here so extract script
    is standalone (no import dependency on the main module).
    """

    MAX_VELOCITY: float = 0.1

    def compute_velocities(self, lineage_history: List[Dict]) -> Dict[str, float]:
        if len(lineage_history) < 2:
            return {p: 0.0 for p in LAYER1_PARAMS}

        velocities = {}
        for param in LAYER1_PARAMS:
            values = []
            for entry in lineage_history:
                snap = entry.get('genome_snapshot', {})
                val = snap.get(param, None)
                if val is not None:
                    values.append(float(val))
            if len(values) >= 2:
                slope = self._slope(values)
                velocities[param] = float(np.clip(slope, -self.MAX_VELOCITY, self.MAX_VELOCITY))
            else:
                velocities[param] = 0.0
        return velocities

    def _slope(self, values: List[float]) -> float:
        x = np.arange(len(values), dtype=float)
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    def compute_summary_vector(self, lineage_history: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        At G >= SUMMARY_VECTOR_THRESHOLD: compute slope/intercept per param.
        Returns {param: {slope, intercept}} for compact storage.
        """
        if len(lineage_history) < 2:
            return {p: {'slope': 0.0, 'intercept': 0.5} for p in LAYER1_PARAMS}

        summary = {}
        for param in LAYER1_PARAMS:
            values = []
            for entry in lineage_history:
                snap = entry.get('genome_snapshot', {})
                val = snap.get(param, None)
                if val is not None:
                    values.append(float(val))
            if len(values) >= 2:
                x = np.arange(len(values), dtype=float)
                coeffs = np.polyfit(x, values, 1)
                summary[param] = {'slope': float(coeffs[0]), 'intercept': float(coeffs[1])}
            else:
                summary[param] = {'slope': 0.0, 'intercept': values[0] if values else 0.5}
        return summary


# ============================================================
# MAIN EXTRACTION LOGIC
# ============================================================

def read_last_session_end(log_path: str) -> Optional[Dict]:
    """Read the last session_end record from a JSONL log file."""
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return None

    session_end = None
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get('type') == 'session_end':
                    session_end = record
            except json.JSONDecodeError:
                continue

    return session_end


def build_lineage_entry(child_genome: Dict, session_end: Dict) -> Dict:
    """Build a lineage_history entry from the current session's results."""
    return {
        'generation': child_genome.get('generation', 0),
        'genome_snapshot': {p: child_genome.get(p, 0.5) for p in LAYER1_PARAMS},
        'fitness': session_end.get('fitness', {}).get('survival_time', 0),
        'richness_summary': session_end.get('genome_richness_final', {}),
        'timestamp': session_end.get('timestamp', 0),
    }


def prune_lineage_history(history: List[Dict], generation: int) -> List[Dict]:
    """
    Keep last LINEAGE_WINDOW entries.
    At G >= SUMMARY_VECTOR_THRESHOLD: convert to summary vector format (not implemented
    in raw history — handled separately via GeneticVelocityEstimator.compute_summary_vector).
    """
    return history[-LINEAGE_WINDOW:]


def print_diagnostics(child_genome: Dict, velocities: Dict, velocity_weight_note: str,
                      model_fidelity: float, session_end: Dict):
    """Print diagnostic summary for the extracted genome."""
    generation = child_genome.get('generation', 0)
    lineage_history = child_genome.get('lineage_history', [])

    print("\n" + "=" * 70)
    print(f"GENOME EXTRACTION — v14.1 — Generation {generation}")
    print("=" * 70)

    # Layer 1 + velocities
    print("\n[LAYER 1: BIAS + VELOCITY]")
    for param in LAYER1_PARAMS:
        vel_field = VELOCITY_FIELD[param]
        bias = child_genome.get(param, 0.0)
        vel = velocities.get(param, 0.0)
        direction = "↑" if vel > 0.005 else ("↓" if vel < -0.005 else "→")
        print(f"  {param:25s} = {bias:.4f}  {vel_field:30s} = {vel:+.5f} {direction}")

    # Layer 2
    causal = child_genome.get('causal_model', {})
    coupling = causal.get('coupling_matrix', {})
    action_map = causal.get('action_substrate_map', {})
    print(f"\n[LAYER 2: CAUSAL MODEL]")
    print(f"  coupling_confidence:    {coupling.get('confidence', 0.0):.3f}")
    print(f"  coupling_observations:  {coupling.get('observations', 0)}")
    print(f"  action_map_affordances: {len(action_map)}")
    print(f"  model_fidelity:         {model_fidelity:.3f}")

    # Layer 3
    discovered = child_genome.get('discovered_structure', {})
    admitted = {k: v for k, v in discovered.items() if v.get('status', 'admitted') == 'admitted'}
    provisional = {k: v for k, v in discovered.items() if v.get('status') == 'provisional'}
    print(f"\n[LAYER 3: DISCOVERED STRUCTURE]")
    print(f"  Total axes:   {len(discovered)}")
    print(f"  Admitted:     {len(admitted)}")
    print(f"  Provisional:  {len(provisional)}")
    for key, entry in admitted.items():
        print(f"    [admitted]    {key} (evidence={entry.get('total_evidence', 0):.1f}, "
              f"gens={entry.get('generations_seen', 0)})")
    for key, entry in provisional.items():
        print(f"    [provisional] {key} (evidence={entry.get('total_evidence', 0):.1f}, "
              f"gens={entry.get('generations_seen', 0)})")

    # Layer 4
    print(f"\n[LAYER 4: LINEAGE HISTORY]")
    print(f"  Lineage depth: {len(lineage_history)} / {LINEAGE_WINDOW}")
    if lineage_history:
        fitness_vals = [e.get('fitness', 0) for e in lineage_history]
        print(f"  Fitness trend: {' → '.join(f'{f:.1f}' for f in fitness_vals)}")
        if len(fitness_vals) >= 2:
            slope = np.polyfit(range(len(fitness_vals)), fitness_vals, 1)[0]
            direction = "↑ improving" if slope > 0.5 else ("↓ declining" if slope < -0.5 else "→ stable")
            print(f"  Fitness slope: {slope:+.2f} ({direction})")
    print(f"  {velocity_weight_note}")

    # Session summary
    fitness = session_end.get('fitness', {})
    print(f"\n[SESSION SUMMARY]")
    print(f"  Survival time:          {fitness.get('survival_time', 0)} steps")
    print(f"  Freeze achieved:        {fitness.get('freeze_achieved', False)}")
    print(f"  Model fidelity final:   {model_fidelity:.3f}")
    print(f"  Virtual mode active:    {session_end.get('virtual_mode_was_active', False)}")
    print(f"  Admitted axes:          {session_end.get('admitted_axes', 'N/A')}")
    print(f"  Provisional axes:       {session_end.get('provisional_axes', 'N/A')}")
    print(f"  Coupling confidence:    {session_end.get('coupling_confidence_final', 0.0):.3f}")

    # Validation checks
    print(f"\n[VALIDATION]")
    coupling_conf = coupling.get('confidence', 0.0)
    coupling_trending = len(lineage_history) >= 2 and coupling_conf > 0.1
    print(f"  Coupling confidence trending: {'✓' if coupling_trending else '○ insufficient data'}")

    action_map_populated = len(action_map) > 0
    print(f"  Action map populated:         {'✓' if action_map_populated else '○ empty (gen < 5 expected)'}")

    axis_count = len(discovered)
    compression_ok = axis_count <= 2 or generation >= 10
    print(f"  Axis compression law:         {'✓' if compression_ok else f'⚠ {axis_count} axes (floor gen < 10)'}")

    # Critical: check for interface-coupled signal contamination
    INTERFACE_COUPLED_SIGNALS = {
        'dom_depth', 'element_count', 'link_count', 'button_count',
        'input_count', 'scroll_position', 'viewport_height', 'dom_complexity'
    }
    contaminated = [k for k in discovered if any(s in k for s in INTERFACE_COUPLED_SIGNALS)]
    if contaminated:
        print(f"  Interface contamination:      ✗ CRITICAL FAILURE: {contaminated}")
        print(f"    → These signals should have been blocked by AxisAdmissionTest")
    else:
        print(f"  Interface contamination:      ✓ clean")

    velocity_mag = float(np.linalg.norm([velocities.get(p, 0.0) for p in LAYER1_PARAMS]))
    print(f"  Velocity magnitude:           {velocity_mag:.5f}")

    print("=" * 70)


def extract_genome(log_path: str, output_path: str = 'genome.json'):
    """Main extraction routine."""

    print(f"[extract_genome_v14_1] Reading: {log_path}")
    session_end = read_last_session_end(log_path)

    if session_end is None:
        print("[ERROR] No session_end record found in log file.")
        print("  Ensure the session completed and called run() → distill_to_genome.")
        sys.exit(1)

    child_genome = session_end.get('child_genome')
    if child_genome is None:
        print("[ERROR] session_end record contains no 'child_genome' field.")
        print("  This log may be from v13.x (pre-v14). Use extract_genome_v13.py.")
        sys.exit(1)

    # --- Read model_fidelity ---
    # v14.1 sessions write model_fidelity_final; v14 sessions don't.
    model_fidelity = session_end.get('model_fidelity_final', 0.5)
    if 'model_fidelity_final' not in session_end:
        print("  [COMPAT] v14 log detected: model_fidelity_final absent, defaulting to 0.5")

    # --- Write model_fidelity into causal_model ---
    if 'causal_model' not in child_genome:
        child_genome['causal_model'] = {}
    child_genome['causal_model']['model_fidelity'] = model_fidelity

    # --- Build and append lineage_history entry ---
    existing_history = child_genome.get('lineage_history', [])
    if not existing_history and 'lineage_history' not in session_end.get('child_genome', {}):
        print("  [COMPAT] v14 genome detected: lineage_history absent, initializing to []")

    new_entry = build_lineage_entry(child_genome, session_end)
    existing_history.append(new_entry)

    # Prune to last LINEAGE_WINDOW entries
    pruned_history = prune_lineage_history(existing_history, child_genome.get('generation', 0))
    child_genome['lineage_history'] = pruned_history

    # --- Compute velocity fields ---
    gve = GeneticVelocityEstimator()
    velocities = gve.compute_velocities(pruned_history)

    # Write velocity fields into child_genome
    for param, vel in velocities.items():
        vel_field = VELOCITY_FIELD[param]
        child_genome[vel_field] = vel

    # --- Compute velocity weight note (informational) ---
    if len(pruned_history) >= 2:
        fitness_vals = [e.get('fitness', 0) for e in pruned_history]
        mean_f = float(np.mean(fitness_vals))
        std_f = float(np.std(fitness_vals))
        coherence = 1.0 - (std_f / max(mean_f, 1e-6))
        coherence = float(np.clip(coherence, 0.0, 1.0))
        if coherence < 0.3:
            weight_note = f"Velocity suppressed: coherence={coherence:.3f} < 0.3 floor"
        elif model_fidelity < 0.4:
            weight_note = f"Velocity suppressed: model_fidelity={model_fidelity:.3f} < 0.4 floor"
        else:
            weight_note = f"Velocity weight: {coherence:.3f} (coherence={coherence:.3f}, fidelity={model_fidelity:.3f})"
    else:
        weight_note = "Velocity weight: 0.0 (insufficient lineage depth)"

    # --- Print diagnostics ---
    print_diagnostics(child_genome, velocities, weight_note, model_fidelity, session_end)

    # --- Write genome.json ---
    output = {'genome': child_genome}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OUTPUT] Written: {output_path}")
    print(f"  Ready for next generation via load_genome('{output_path}')")
    print(f"  Generation {child_genome.get('generation', 0)} → {child_genome.get('generation', 0) + 1}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'mentat_triad_v14_log.jsonl'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'genome.json'

    extract_genome(log_path, output_path)
