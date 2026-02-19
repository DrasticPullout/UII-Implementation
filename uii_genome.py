"""
UII v14.1 — uii_genome.py
Memory & Continuity

Role: The genome is the fourth leg of the Triad — Memory.
It is not a config file. It carries heritable causal structure across generations.
Each session compresses what it learned into the child genome via FAO.distill_to_genome.
Velocity fields (v14.1) make the genome predictive, not just empirical.

Contents:
  - TriadGenome (four-layer heritable genome)
  - GeneticVelocityEstimator (least-squares slope per Layer 1 param)
  - LineageCoherenceCheck (velocity suppression for incoherent lineages)
  - ModelFidelityMonitor (virtual/real Φ delta + action map accuracy)
  - ProvisionalAxisManager (two-tier Layer 3 decay lifecycle)
  - load_genome() (momentum-weighted initialization + mutation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import copy
import json

from uii_types import LAYER1_PARAMS, VELOCITY_FIELD

@dataclass
class TriadGenome:
    """
    Four-layer heritable genome.

    Layer 1: Fixed bootstrap (6 floats + 6 velocity fields — v14.1 adds velocity)
    Layer 2: Learned causal model (grows each generation via FAO.distill_to_genome)
    Layer 3: Discovered structure (emergent axes, strict 4-condition admission)
    Layer 4: Lineage history (last 5 generations — enables velocity computation)

    Layers 2, 3, 4 start empty on generation 0.
    They are populated by FAO.distill_to_genome and extract_genome_v14_1.py at session end.
    Only predictively valid structure survives compression pressure.

    Backward compatible: v14 genome.json loads fine (velocity fields default 0.0,
    lineage_history defaults to []).
    """

    # Layer 1: Fixed bootstrap + velocity fields (v14.1 adds velocity)
    S_bias: float = 0.5
    S_velocity: float = 0.0
    I_bias: float = 0.5
    I_velocity: float = 0.0
    P_bias: float = 0.5
    P_velocity: float = 0.0
    A_bias: float = 0.7
    A_velocity: float = 0.0
    rigidity_init: float = 0.5
    rigidity_init_velocity: float = 0.0
    phi_coherence_weight: float = 0.7
    phi_coherence_velocity: float = 0.0

    # Evolution metadata
    generation: int = 0
    parent_fitness: float = 0.0

    # Layer 2: Learned causal model (dynamic — grows each generation)
    causal_model: Dict = field(default_factory=dict)

    # Layer 3: Discovered structure (emergent — strict admission required)
    discovered_structure: Dict = field(default_factory=dict)

    # Layer 4: Lineage history (last N=5 generations — enables velocity computation)
    lineage_history: List[Dict] = field(default_factory=list)

    def mutate(self, mutation_rate: float = 0.1) -> 'TriadGenome':
        """
        Gaussian mutation on Layer 1 bias fields only.
        Velocity fields are NOT mutated — they are computed by GeneticVelocityEstimator.
        Layers 2, 3, 4 pass through intact — FAO.distill_to_genome handles them.
        """
        mutated = copy.deepcopy(self)
        mutated.generation += 1
        for field_name in LAYER1_PARAMS:
            current = getattr(self, field_name)
            noise = np.random.normal(0, mutation_rate)
            setattr(mutated, field_name, np.clip(current + noise, 0, 1))
        # Velocity fields pass through unmodified
        return mutated

    def richness_summary(self) -> Dict:
        """Diagnostic summary of genome richness across all four layers."""
        causal = self.causal_model
        discovered = self.discovered_structure
        velocities = {p: getattr(self, VELOCITY_FIELD[p], 0.0) for p in LAYER1_PARAMS}
        velocity_magnitude = float(np.linalg.norm(list(velocities.values())))
        provisional_count = sum(1 for v in discovered.values() if v.get('status') == 'provisional')
        admitted_count = sum(1 for v in discovered.values() if v.get('status', 'admitted') == 'admitted')
        return {
            'generation': self.generation,
            'layer1': '6 floats + 6 velocity fields',
            'layer2_keys': list(causal.keys()),
            'coupling_confidence': causal.get('coupling_matrix', {}).get('confidence', 0.0),
            'coupling_observations': causal.get('coupling_matrix', {}).get('observations', 0),
            'action_map_affordances': len(causal.get('action_substrate_map', {})),
            'layer3_axes': len(discovered),
            'layer3_keys': list(discovered.keys()),
            'layer3_provisional': provisional_count,
            'layer3_admitted': admitted_count,
            'lineage_depth': len(self.lineage_history),
            'velocity_magnitude': velocity_magnitude,
            'coherence_score': causal.get('model_fidelity', None),
        }


# ============================================================
# ============================================================
# MODULE 0.1: v14.1 NEW CLASSES
# ============================================================

class GeneticVelocityEstimator:
    """
    Computes Layer 1 momentum vectors from lineage_history.

    Uses least-squares slope across last N generations per parameter.
    Called by extract_genome_v14_1.py at generation boundary — not at runtime.

    Generation 0 (empty history): all velocities = 0.0.
    Velocities clipped to ±0.1 to prevent runaway momentum.
    Window: uses all entries in lineage_history (max 5). Fewer if lineage is young.
    """

    MAX_VELOCITY: float = 0.1

    def compute_velocities(self, lineage_history: List[Dict]) -> Dict[str, float]:
        """
        Returns {param_name: velocity} for all 6 Layer 1 params.
        Requires len(lineage_history) >= 2 for meaningful slopes.
        """
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
        """Least-squares slope of values over their index."""
        x = np.arange(len(values), dtype=float)
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])


class LineageCoherenceCheck:
    """
    Scales velocity_weight [0.0, 1.0] based on lineage fitness variance
    and model fidelity.

    Suppresses momentum in incoherent lineages and low-fidelity sessions.
    Both conditions must pass for velocity to apply.

    COHERENCE_FLOOR = 0.3   (committee: middle of GPT 0.2 / Gemini 0.4)
    MODEL_FIDELITY_FLOOR = 0.4
    """

    COHERENCE_FLOOR: float = 0.3
    MODEL_FIDELITY_FLOOR: float = 0.4

    def get_velocity_weight(self,
                            lineage_history: List[Dict],
                            model_fidelity: float) -> float:
        """
        Returns velocity_weight in [0.0, 1.0].

        0.0 if:
          - lineage_history has fewer than 2 entries (no trend computable)
          - fitness variance too high (coherence < COHERENCE_FLOOR)
          - model_fidelity < MODEL_FIDELITY_FLOOR
        Otherwise: coherence score (proportional scaling).
        """
        if len(lineage_history) < 2:
            return 0.0

        fitness_values = [float(g.get('fitness', 0.0)) for g in lineage_history]
        mean_f = float(np.mean(fitness_values))
        std_f = float(np.std(fitness_values))
        coherence = 1.0 - (std_f / max(mean_f, 1e-6))
        coherence = float(np.clip(coherence, 0.0, 1.0))

        if coherence < self.COHERENCE_FLOOR:
            return 0.0
        if model_fidelity < self.MODEL_FIDELITY_FLOOR:
            return 0.0

        return coherence  # proportional: stable lineages get full momentum weight


class ModelFidelityMonitor:
    """
    Global causal model quality signal.

    Tracks two error sources:
      1. predict_delta error per affordance (from ResidualTracker summary)
      2. Virtual vs real Φ delta per committed trajectory (v14.1 new)

    model_fidelity = 0.6 * action_map_accuracy + 0.4 * coupling_accuracy

    Outputs float [0, 1]. Written to session_end and genome causal_model['model_fidelity'].
    All inputs externally derived — not self-referential.
    """

    def __init__(self):
        self.virtual_real_deltas: List[float] = []   # |predicted_phi - actual_phi|
        self.action_map_errors: List[float] = []     # per-step predict_delta MSE

    def record_trajectory_delta(self, predicted_phi: float, actual_phi: float):
        """Called after each committed trajectory: virtual prediction vs real execution."""
        self.virtual_real_deltas.append(abs(predicted_phi - actual_phi))

    def record_action_error(self, mse: float):
        """Called from step loop — per-step mean predict_delta MSE."""
        self.action_map_errors.append(mse)

    def get_fidelity(self) -> float:
        """
        Returns model_fidelity in [0, 1].
        Higher = model predictions more reliable.
        Returns 0.5 (neutral) if insufficient data — don't suppress on first generation.
        """
        if not self.virtual_real_deltas and not self.action_map_errors:
            return 0.5  # No data yet — neutral

        coupling_accuracy = 0.5
        if self.virtual_real_deltas:
            # Mean absolute error on Φ — normalize (error > 2.0 → fidelity = 0)
            mean_delta = float(np.mean(self.virtual_real_deltas))
            coupling_accuracy = float(np.clip(1.0 - mean_delta / 2.0, 0.0, 1.0))

        action_map_accuracy = 0.5
        if self.action_map_errors:
            mean_mse = float(np.mean(self.action_map_errors[-50:]))  # recent 50 only
            action_map_accuracy = float(np.clip(1.0 - mean_mse * 10, 0.0, 1.0))

        return float(0.6 * action_map_accuracy + 0.4 * coupling_accuracy)


class ProvisionalAxisManager:
    """
    Two-tier Layer 3 management: provisional and admitted.

    Provisional axes:
      - Evidence floor: 5 (vs 20 for admitted)
      - Decay: 0.5x per generation (vs 0.8x for admitted)
      - Short-session variant: 0.6x if session_length < 30 steps
      - Watched by ResidualTracker from session start (priority observation)

    Promotion: provisional → admitted when AxisAdmissionTest passes on real residuals.
    Removal: automatic via faster decay.

    Framing:
      Cheap to test (provisional costs almost nothing)
      Cheap to remove (0.5x decay clears in ~3 generations)
      Expensive to keep (full AxisAdmissionTest required for promotion)
      Continuously re-validated (decay never stops, even for admitted)
    """

    PROVISIONAL_EVIDENCE_FLOOR: float = 5.0
    ADMITTED_EVIDENCE_FLOOR: float = 20.0
    PROVISIONAL_DECAY: float = 0.5
    PROVISIONAL_DECAY_SHORT_SESSION: float = 0.6
    ADMITTED_DECAY: float = 0.8
    SHORT_SESSION_THRESHOLD: int = 30

    def decay_and_prune(self,
                        discovered_structure: Dict,
                        session_length: int) -> Dict:
        """
        Apply per-tier decay to all Layer 3 entries. Remove below floor.
        Called by FAO.distill_to_genome before returning child genome.
        """
        prov_decay = (self.PROVISIONAL_DECAY_SHORT_SESSION
                      if session_length < self.SHORT_SESSION_THRESHOLD
                      else self.PROVISIONAL_DECAY)

        result = {}
        for key, entry in discovered_structure.items():
            status = entry.get('status', 'admitted')
            if status == 'provisional':
                decay = prov_decay
                floor = self.PROVISIONAL_EVIDENCE_FLOOR
            else:
                decay = self.ADMITTED_DECAY
                floor = self.ADMITTED_EVIDENCE_FLOOR

            new_evidence = entry.get('total_evidence', 0.0) * decay
            if new_evidence >= floor:
                result[key] = {**entry, 'total_evidence': new_evidence}
            # else: pruned — did not survive compression pressure
        return result

    def try_promote(self,
                    key: str,
                    entry: Dict,
                    candidate: Dict,
                    residual_tracker,
                    phi_history: List[float],
                    axis_admission) -> Dict:
        """
        Attempt promotion from provisional → admitted.
        Runs full AxisAdmissionTest against real residuals.
        Returns updated entry dict (status may change to 'admitted').
        """
        if entry.get('status') != 'provisional':
            return entry

        admitted, results = axis_admission.evaluate(candidate, residual_tracker, phi_history)
        if admitted:
            return {**entry, 'status': 'admitted', 'admission_results': results}
        return entry

    def get_provisional_keys(self, discovered_structure: Dict) -> List[str]:
        """Keys of provisional axes — watched by ResidualTracker from session start."""
        return [k for k, v in discovered_structure.items()
                if v.get('status', 'admitted') == 'provisional']


# NEW: CouplingMatrixEstimator


# ============================================================
# GENOME UTILITIES
# ============================================================

def load_genome(path: str) -> TriadGenome:
    """
    Load genome from JSON, apply momentum-weighted Layer 1 initialization, then mutate.

    v14.1: reads velocity fields and lineage_history. Applies momentum before mutation.
    Backward compatible: v14 genome.json loads fine (velocity fields default 0.0,
    lineage_history defaults to []).
    """
    with open(path) as f:
        data = json.load(f)

    genome_dict = data['genome']

    # Layer 1 params + velocity fields (velocity defaults to 0.0 for v14 genomes)
    active_params = {k: v for k, v in genome_dict.items()
                    if k in ['S_bias', 'S_velocity',
                             'I_bias', 'I_velocity',
                             'P_bias', 'P_velocity',
                             'A_bias', 'A_velocity',
                             'rigidity_init', 'rigidity_init_velocity',
                             'phi_coherence_weight', 'phi_coherence_velocity',
                             'generation', 'parent_fitness']}

    # Layers 2 and 3 (unchanged)
    active_params['causal_model'] = genome_dict.get('causal_model', {})
    active_params['discovered_structure'] = genome_dict.get('discovered_structure', {})
    # Layer 4 (present in v14.1, empty list for v14 genomes)
    active_params['lineage_history'] = genome_dict.get('lineage_history', [])

    genome = TriadGenome(**active_params)

    # v14.1: Apply momentum-weighted Layer 1 initialization (before Gaussian mutation)
    model_fidelity = genome.causal_model.get('model_fidelity', 0.5)
    lcc = LineageCoherenceCheck()
    velocity_weight = lcc.get_velocity_weight(genome.lineage_history, model_fidelity)

    if velocity_weight > 0.0:
        for param in LAYER1_PARAMS:
            velocity_field = VELOCITY_FIELD[param]
            velocity = getattr(genome, velocity_field, 0.0)
            current = getattr(genome, param)
            new_val = float(np.clip(current + velocity_weight * velocity, 0.0, 1.0))
            setattr(genome, param, new_val)

    richness = genome.richness_summary()
    print(f"\n[LOADED GENOME]")
    print(f"  Generation: {genome.generation}")
    print(f"  Parent fitness: {genome.parent_fitness:.2f}")
    print(f"  Coupling confidence: {richness['coupling_confidence']:.2f}")
    print(f"  Action map: {richness['action_map_affordances']} affordances")
    print(f"  Discovered axes: {richness['layer3_axes']}")
    print(f"  Lineage depth: {richness['lineage_depth']}")
    print(f"  Velocity magnitude: {richness['velocity_magnitude']:.4f}")
    print(f"  Velocity weight applied: {velocity_weight:.2f}")

    # Layer 1 mutation only — Layers 2/3/4 pass through intact
    genome = genome.mutate(mutation_rate=0.1)
    print(f"  Mutated to generation: {genome.generation}")

    return genome


