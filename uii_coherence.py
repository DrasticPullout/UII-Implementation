from __future__ import annotations  # lazy annotations for cross-module type hints

"""
UII v14.1 -- uii_coherence.py
Coherence Management

Role: The CNS leg of the Mentat Triad. Maintains internal coherence through
continuous micro-perturbations and gradient-following (nablaΦ), not scalar thresholds.
Attempts geometric mitosis as a learning mechanism -- failure is the primary signal.
Detects structural impossibility geometrically, never motivationally.

Contents:
  - ExteriorNecessitationOperator (ENO -- gating under boundary pressure)
  - ControlAsymmetryMeasure (CAM -- empirical action->substrate map)
  - ExteriorGradientDescent (EGD -- pattern composition in ENO regime)
  - LatentDeathClock (step/token budget tracking)
  - TemporalPerturbationMemory (recent-action exclusion)
  - ContinuousRealityEngine (micro-action selection and prediction)
  - CNSMitosisOperator (geometric replication attempts)
  - ImpossibilityDetector (structural impossibility, Reality-derived triggers only)
  - AutonomousTrajectoryLab (trajectory testing in Reality, with virtual mode)
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import copy
import json
import hashlib
import time
from collections import deque
from pathlib import Path

from uii_types import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SMO, SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    RealityAdapter,
)
from uii_operators import SensingOperator, CompressionOperator, PredictionOperator, CoherenceOperator
from uii_genome import TriadGenome
from uii_fao import FailureAssimilationOperator

class ExteriorNecessitationOperator:
    """Detects externally gated BASE AFFORDANCES via empirical tracking."""

    def __init__(self, activation_window: int = 20, gating_threshold: float = 0.6):
        self.activation_window = activation_window
        self.gating_threshold = gating_threshold

        self.affordance_history: deque = deque(maxlen=100)

        self.gated_affordances: Set[str] = set()
        self.viable_affordances: Set[str] = BASE_AFFORDANCES.copy()

        self.active: bool = False

    def check_activation(self, smo: SMO, trace: StateTrace,
                         egd: Optional['ExteriorGradientDescent'] = None) -> bool:
        """
        ENO activation.
        v15.2 Step 5: trigger changed to C_local < floor via egd.gradient_lost().
        egd=None → legacy prediction-error trigger (bootstrap fallback).
        ENO deactivates when C_local recovers above C_LOCAL_FLOOR for 3+ steps.
        activation_window still gates escalation to Tier 3 (ENO must be long-running).
        """
        if egd is not None and hasattr(trace, 'c_local_history'):
            # v15.2: gradient-lost trigger — C_local below floor for 5 steps
            self.active = egd.gradient_lost(trace)
        else:
            # Legacy: prediction error low → external gate suspected
            if len(smo.prediction_error_history) < self.activation_window:
                self.active = False
                return False
            recent_errors = list(smo.prediction_error_history)[-self.activation_window:]
            all_low_error = all(e < 0.005 for e in recent_errors)
            not_locked = smo.rigidity < 0.85
            self.active = all_low_error and not_locked

        if self.active:
            self._update_gated_affordances()

        return self.active

    def record_affordance_outcome(self, affordance_type: str,
                                  success: bool, refusal: bool):
        """Record empirical outcome of affordance execution."""
        self.affordance_history.append({
            'affordance': affordance_type,
            'success': success,
            'refusal': refusal,
            'timestamp': len(self.affordance_history)
        })

    def _update_gated_affordances(self):
        """Detect which affordances are externally gated."""
        if len(self.affordance_history) < 10:
            return

        gated = set()

        for affordance in BASE_AFFORDANCES:
            recent = [
                r for r in list(self.affordance_history)[-30:]
                if r['affordance'] == affordance
            ]

            if len(recent) >= 3:
                refusal_rate = sum(1 for r in recent if r['refusal']) / len(recent)

                if refusal_rate > self.gating_threshold:
                    gated.add(affordance)

        self.gated_affordances = gated
        self.viable_affordances = BASE_AFFORDANCES - gated

    def get_gated_affordances(self) -> Set[str]:
        return self.gated_affordances.copy()

    def get_viable_affordances(self) -> Set[str]:
        return self.viable_affordances.copy()

    def is_active(self) -> bool:
        return self.active


class ControlAsymmetryMeasure:
    """Builds controllability covariance graph of affordances."""

    def __init__(self):
        self.action_sequences: deque = deque(maxlen=100)
        self.affordance_deltas: Dict[str, List[Dict]] = {
            aff: [] for aff in BASE_AFFORDANCES
        }
        # v15.2 Step 6: gradient alignment EMA per affordance
        self.alignment_ema: Dict[str, float] = {}

    def record_action_outcome(self, action: str,
                               delta: Dict[str, float],
                               gradient: Dict[str, float]):
        """
        v15.2 Step 6: EMA of gradient alignment per affordance.
        Called after each micro-perturbation with the current ∇Φ.
        """
        common    = set(delta.keys()) & set(gradient.keys())
        alignment = float(np.dot([delta[k] for k in common],
                                  [gradient[k] for k in common])) if common else 0.0
        alpha = 0.1
        self.alignment_ema[action] = (
            (1 - alpha) * self.alignment_ema.get(action, 0.0) + alpha * alignment
        )
        # Keep raw delta in affordance_deltas for SRE action_substrate_map
        self._record_delta(action, delta)

    def _record_delta(self, action: str, delta: Dict[str, float]):
        """Store raw delta for SRE action_substrate_map consumption."""
        if action in BASE_AFFORDANCES:
            self.affordance_deltas.setdefault(action, []).append({
                'delta': delta,
                'prev_affordance': None,
                'timestamp': len(self.action_sequences),
            })

    def get_gradient_aligned_action_map(self) -> Dict[str, float]:
        """v15.2: Returns EMA gradient alignment per affordance."""
        return dict(self.alignment_ema)

    def record_action_sequence(self, action: Dict, observed_delta: Dict,
                              prev_action: Optional[Dict] = None):
        """Record action → delta with context of previous action."""
        affordance = action.get('type', 'unknown')

        if affordance in BASE_AFFORDANCES:
            self.affordance_deltas[affordance].append({
                'delta': observed_delta,
                'prev_affordance': prev_action.get('type') if prev_action else None,
                'timestamp': len(self.action_sequences)
            })

        self.action_sequences.append({
            'affordance': affordance,
            'delta': observed_delta
        })

    def build_covariance_graph(self, viable_affordances: Set[str]) -> Dict[str, Dict[str, float]]:
        """Build affordance covariance graph."""
        graph = {aff: {} for aff in viable_affordances}

        if len(self.action_sequences) < 20:
            return graph

        for aff_a in viable_affordances:
            for aff_b in viable_affordances:
                if aff_a == aff_b:
                    continue

                ab_sequences = []
                for i in range(len(self.action_sequences) - 1):
                    if (self.action_sequences[i]['affordance'] == aff_a and
                        self.action_sequences[i+1]['affordance'] == aff_b):
                        combined_delta = self.action_sequences[i+1]['delta']
                        magnitude = sum(abs(combined_delta.get(dim, 0))
                                      for dim in ['S', 'I', 'P', 'A'])
                        ab_sequences.append(magnitude)

                a_alone = [
                    sum(abs(r['delta'].get(dim, 0)) for dim in ['S', 'I', 'P', 'A'])
                    for r in self.affordance_deltas.get(aff_a, [])[-10:]
                ]

                if len(ab_sequences) >= 2 and len(a_alone) >= 2:
                    ab_mean = np.mean(ab_sequences)
                    a_mean = np.mean(a_alone)

                    covariance = ab_mean - a_mean

                    if covariance > 0:
                        graph[aff_a][aff_b] = covariance

        return graph

    def extract_pattern_clusters(self, graph: Dict[str, Dict[str, float]],
                                 threshold: float = 0.01) -> List[Set[str]]:
        """Extract dense subgraphs (patterns) from covariance graph."""
        adjacency = {node: set() for node in graph.keys()}

        for node_a in graph:
            for node_b, weight in graph[node_a].items():
                if weight > threshold:
                    adjacency[node_a].add(node_b)
                    adjacency[node_b].add(node_a)

        visited = set()
        clusters = []

        def dfs(node, cluster):
            visited.add(node)
            cluster.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for node in adjacency:
            if node not in visited:
                cluster = set()
                dfs(node, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    def measure_cluster_control(self, cluster: Set[str]) -> float:
        """Measure total control of a pattern cluster."""
        if not cluster:
            return 0.0

        total_control = 0.0

        for affordance in cluster:
            recent = self.affordance_deltas.get(affordance, [])[-10:]
            for record in recent:
                magnitude = sum(abs(record['delta'].get(dim, 0))
                              for dim in ['S', 'I', 'P', 'A'])
                total_control += magnitude

        return total_control

    def get_empirical_action_map(self) -> Dict[str, Dict[str, float]]:
        """
        Export learned action->delta map for genome inheritance (Layer 2).
        Only exports affordances with >= 10 observations (sufficient evidence).
        """
        action_map = {}
        for affordance, deltas in self.affordance_deltas.items():
            if len(deltas) >= 10:
                action_map[affordance] = {
                    dim: float(np.mean([d['delta'].get(dim, 0.0) for d in deltas]))
                    for dim in SUBSTRATE_DIMS
                }
        return action_map


class ExteriorGradientDescent:
    """
    EGD — default CNS operating mode.
    v15.2: argmax_a ⟨Δ(a), ∇Φ(x)⟩. Not a special state — the protocol.
    gradient_lost() is the trigger for ENO activation.
    """

    C_LOCAL_FLOOR = 0.1

    def __init__(self):
        self.cluster_history: deque = deque(maxlen=20)
        self.zero_control_counter = 0
        self.zero_control_threshold = 3

    def gradient_align_score(self, action: str,
                              delta: Dict[str, float],
                              gradient: Dict[str, float]) -> float:
        """⟨Δ(action), ∇Φ(x)⟩ — gradient alignment for one action candidate."""
        common = set(delta.keys()) & set(gradient.keys())
        if not common:
            return 0.0
        return float(np.dot([delta[k] for k in common],
                             [gradient[k] for k in common]))

    def gradient_lost(self, trace: 'StateTrace') -> bool:
        """C_local below floor for 3+ consecutive steps — ENO should activate."""
        if len(trace.c_local_history) < 3:
            return False
        return float(np.mean(list(trace.c_local_history)[-5:])) < self.C_LOCAL_FLOOR

    def stability_check(self, action_delta: Dict[str, float],
                        coupling_matrix: np.ndarray,
                        dim_order: List[str]) -> float:
        """δ²Φ = ½ δxᵀ A(x) δx. Positive = stable. Negative = amplifying."""
        dv = np.array([action_delta.get(d, 0.0) for d in dim_order])
        return float(0.5 * dv @ coupling_matrix @ dv)

    def discover_and_select_pattern(self,
                                    eno: ExteriorNecessitationOperator,
                                    cam: 'ControlAsymmetryMeasure') -> Tuple[Set[str], List[Set[str]], Dict]:
        """Discover pattern structure and select best cluster (legacy path for Tier 2/3)."""
        viable = eno.get_viable_affordances()
        graph = cam.build_covariance_graph(viable)
        clusters = cam.extract_pattern_clusters(graph, threshold=0.01)
        if not clusters:
            clusters = [viable]

        cluster_controls = {}
        for i, cluster in enumerate(clusters):
            control = cam.measure_cluster_control(cluster)
            cluster_controls[i] = {'cluster': cluster, 'control': control}

        if cluster_controls:
            best_idx = max(cluster_controls.keys(),
                          key=lambda k: cluster_controls[k]['control'])
            selected_cluster = cluster_controls[best_idx]['cluster']
            max_control = cluster_controls[best_idx]['control']
        else:
            selected_cluster = viable
            max_control = 0.0

        if max_control < 0.01:
            self.zero_control_counter += 1
        else:
            self.zero_control_counter = 0

        self.cluster_history.append({
            'cluster': selected_cluster,
            'control': max_control,
            'num_clusters_found': len(clusters)
        })

        return selected_cluster, clusters, cluster_controls

    def all_patterns_collapsed(self) -> bool:
        return self.zero_control_counter >= self.zero_control_threshold

    def format_clusters_for_llm(self, clusters: List[Set[str]],
                                cluster_controls: Dict) -> str:
        """Format discovered pattern clusters for LLM prompt."""
        if not clusters:
            return "No pattern clusters discovered yet (insufficient data)"
        formatted = []
        for i, cluster_info in cluster_controls.items():
            cluster = cluster_info['cluster']
            control = cluster_info['control']
            formatted.append(
                f"Cluster {i+1} (control: {control:.3f}): " +
                "{" + ", ".join(sorted(cluster)) + "}"
            )
        return "\n".join(formatted)


# ============================================================
# MODULE 4.6: LATENT DEATH CLOCK
# ============================================================

class LatentDeathClock:
    """
    Dual-budget mortality with CONSTANT boundary pressure.

    v13.2: Actual token tracking from Groq API.
    """

    def __init__(self, step_budget: int, token_budget: int):
        self.step_budget = step_budget
        self.token_budget = token_budget
        self.current_steps = 0
        self.current_tokens = 0

    def tick_step(self):
        """Advance step counter."""
        self.current_steps += 1

    def tick_tokens(self, count: int):
        """Advance token counter by actual usage."""
        self.current_tokens += count

    def get_boundary_pressure(self) -> float:
        """
        v15: Dynamic boundary pressure — max(step_pressure, token_pressure).

        Token pressure spikes on every LLM call and never decays — tokens
        spent are gone permanently. The Triad feels token expenditure in its
        Φ field, naturally preferring geometric resolution over LLM calls
        when σ(x) outputs are sufficient.

        Step pressure rises monotonically — time passing.
        Token pressure is non-decreasing — resource permanently consumed.
        """
        step_pressure  = self.current_steps / self.step_budget if self.step_budget > 0 else 0.0
        token_pressure = self.current_tokens / self.token_budget if self.token_budget > 0 else 0.0
        return float(min(max(step_pressure, token_pressure), 1.0))

    def should_terminate(self) -> bool:
        """Hard termination when EITHER budget exhausted."""
        return (self.current_steps >= self.step_budget or
                self.current_tokens >= self.token_budget)

    def get_binding_constraint(self) -> str:
        """Identify binding constraint."""
        step_progress = self.current_steps / self.step_budget
        token_progress = self.current_tokens / self.token_budget
        return 'tokens' if token_progress > step_progress else 'steps'

    def get_remaining_budget(self) -> int:
        """Remaining units of binding constraint."""
        if self.get_binding_constraint() == 'tokens':
            return self.token_budget - self.current_tokens
        return self.step_budget - self.current_steps

    def get_degradation_progress(self) -> float:
        """Progress of binding constraint."""
        step_progress = self.current_steps / self.step_budget
        token_progress = self.current_tokens / self.token_budget
        return max(step_progress, token_progress)


# ============================================================
# MODULE 5: CONTINUOUS REALITY ENGINE
# ============================================================

class TemporalPerturbationMemory:
    """Bounded, short-term exclusion of recently perturbed loci."""

    def __init__(self, window_steps: int = 5, capacity: int = 20):
        self.memory: Dict[str, int] = {}
        self.window_steps = window_steps
        self.capacity = capacity

    def mark_perturbed(self, locus: str):
        self.memory[locus] = self.window_steps

        if len(self.memory) > self.capacity:
            oldest = min(self.memory.keys(), key=lambda k: self.memory[k])
            del self.memory[oldest]

    def is_recently_perturbed(self, locus: str) -> bool:
        return locus in self.memory and self.memory[locus] > 0

    def decay_all(self):
        expired = []
        for locus in self.memory:
            self.memory[locus] -= 1
            if self.memory[locus] <= 0:
                expired.append(locus)

        for locus in expired:
            del self.memory[locus]

    def get_exclusion_count(self) -> int:
        return len(self.memory)

    def clear(self):
        self.memory.clear()


import psutil

# Actions eligible for CRK pre-action scoring.
# Matches BASE_AFFORDANCES — all actions can be filtered.
SCOREABLE_AFFORDANCES = BASE_AFFORDANCES.copy()


class ContinuousRealityEngine:
    """CNS-driven micro-perturbation system."""

    def __init__(self, reality: RealityAdapter, inherited_action_map: Optional[Dict] = None,
                 crk: Optional[CRKMonitor] = None):
        self.reality = reality
        self.action_count = 0
        self.crk = crk   # v15.1: CRK for pre-action filtering; None → no filtering

        self.temporal_memory = TemporalPerturbationMemory(
            window_steps=5,
            capacity=20
        )
        # v14: Inherited action map from genome (primary predictor).
        self.learned_predictions: Dict = inherited_action_map or {}

        # v15 Tier 1: SRE action weights — set by apply_structural_weights()
        self._structural_weights: Dict[str, float] = {}

        # v15.2: cam and egd wired by MentatTriad after construction
        self.cam: Optional['ControlAsymmetryMeasure'] = None
        self.egd: Optional['ExteriorGradientDescent'] = None

    def apply_structural_weights(self, action_weights: Dict[str, float]):
        """
        v15 Tier 1: bias CNS micro-action selection toward SRE-recommended affordances.
        Weights persist until next SRE diagnosis (overwritten on next impossibility).
        """
        self._structural_weights = action_weights

    def choose_micro_action(self, state: SubstrateState, affordances: Dict,
                             trace: Optional['StateTrace'] = None,
                             sre_weights: Optional[Dict] = None,
                             phi_history: Optional[List[float]] = None) -> Dict:
        """
        v15.2 Step 7: gradient alignment as primary scoring mechanism.

        PRIMARY (when gradient and causal_graph available):
            score = 0.5×grad_align + 0.2×stability + 0.2×cam_align + 0.1×sre_weight

        FALLBACK (bootstrap — no graph yet):
            score = reflex heuristics (original path)

        CRK pre-action filter applied in both paths.

        v15.3: phi_history added — threads real Φ values into _build_field_state
        so C7 phi_trend is computed from actual Φ geometry, not trace S/I/P/A records
        which carry no phi key.
        """
        viable = self._build_viable_set(affordances)
        gradient = getattr(trace, '_last_gradient', {}) if trace else {}

        # ẋ = ∇Φ(x): score every action by how well its predicted channel-space
        # delta aligns with the current gradient field.
        #
        # PredictionOperator.test_virtual() is the authoritative forward model:
        # primary channel influence + causal graph propagation → channel-space delta.
        # Both delta and gradient are in channel space — no projection needed.
        #
        # Gate: gradient must be non-empty. No causal_graph requirement —
        # gradient is valid from coverage diagonal even before any edges form.
        if gradient and self.egd is not None and self.cam is not None:
            cam_scores = self.cam.get_gradient_aligned_action_map()
            try:
                coupling    = state.compression.to_coupling_matrix()
                have_coupling = True
            except Exception:
                coupling    = None
                have_coupling = False

            scores = {}
            for action in viable:
                channel_delta = state.prediction.test_virtual(
                    compression      = state.compression,
                    candidate_action = action,
                    phi_field        = None,   # not used in new implementation
                    sensing          = state.sensing,
                )
                grad_score = self.egd.gradient_align_score(action, channel_delta, gradient)

                # δ²Φ stability — project channel delta to SIPA for coupling matrix
                if have_coupling:
                    from uii_operators import CHANNEL_TO_DIM
                    sipa = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
                    for cid, v in channel_delta.items():
                        dim = CHANNEL_TO_DIM.get(cid)
                        if dim:
                            sipa[dim] += v
                    stability = self.egd.stability_check(sipa, coupling, ['S', 'I', 'P', 'A'])
                else:
                    stability = 0.0

                cam_score  = cam_scores.get(action, 0.0)
                sre_weight = sre_weights.get(action, 1.0) if sre_weights else 1.0
                scores[action] = (grad_score                           * 0.6 +
                                  float(np.clip(stability, 0.0, 1.0)) * 0.2 +
                                  cam_score                            * 0.1 +
                                  sre_weight                           * 0.1)
        else:
            # No gradient yet — reflex heuristics only
            scores = self._score_candidates_from_reflexes(viable, state, affordances, sre_weights)

        # CRK pre-action filter.
        # PHI_MOD_FLOOR: during bootstrap loop_closure=0 → phi_modifier=0 for every
        # action → all filtered scores zero → max() indeterminate. Floor preserves
        # gradient ranking through CRK without weakening the coherent=False block.
        PHI_MOD_FLOOR = 0.05
        if self.crk is not None and hasattr(state, 'coherence'):
            field_state = self._build_field_state(state, trace, phi_history)
            filtered, repairs = {}, []
            for action in viable:
                verdict = self.crk.evaluate_pre_action(
                    proposed_action = action,
                    coherence       = state.coherence,
                    sensing         = state.sensing,
                    compression     = state.compression,
                    prediction      = state.prediction,
                    field_state     = field_state,
                )
                if verdict.coherent:
                    mod = max(verdict.phi_modifier, PHI_MOD_FLOOR)
                    filtered[action] = scores.get(action, 0.0) * mod
                if verdict.repair:
                    repairs.append(verdict.repair)
            if not filtered:
                filtered = {a: scores.get(a, 0.0) * PHI_MOD_FLOOR for a in viable}
            if repairs:
                filtered = self._apply_repair_bias(filtered, repairs[0])
            best = max(filtered, key=filtered.get)
        else:
            best = max(scores, key=scores.get)

        return self._action_from_type(best, state, affordances)

    def _build_viable_set(self, affordances: Dict) -> set:
        """Translate page state into viable action set."""
        # These are always available regardless of page state
        viable = {'observe', 'delay', 'evaluate', 'python', 
                'llm_query', 'migrate', 'query_agent', 'navigate'}
        
        # These require specific page elements to be meaningful
        if affordances.get('buttons'):  viable.add('click')
        if affordances.get('readable'): viable.add('read')
        if affordances.get('inputs'):   viable |= {'fill', 'type'}
        scrollable = (affordances.get('total_height', 0) -
                    affordances.get('viewport_height', 0))
        if scrollable > 0: viable.add('scroll')
        
        viable &= SCOREABLE_AFFORDANCES
        return viable

    def _predict_action_delta(self, action: str, state: SubstrateState) -> Dict[str, float]:
        """
        v15.2 Step 7: look up delta from action_substrate_map (genome ledger),
        fall back to CAM empirical map, return zeros if neither available.
        """
        # Primary: inherited action map
        if action in self.learned_predictions:
            return dict(self.learned_predictions[action])
        # Secondary: CAM empirical (live observations)
        if self.cam is not None:
            empirical = self.cam.get_empirical_action_map()
            if action in empirical:
                return dict(empirical[action])
        return {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

    def _score_candidates_from_reflexes(self, candidates: set,
                                         state: SubstrateState,
                                         affordances: Dict,
                                         sre_weights: Optional[Dict]) -> Dict[str, float]:
        """Score each candidate action using reflex heuristics and SRE weights."""
        scores = {}
        for action in candidates:
            base = 0.1   # default
            # Heuristic boosts from reflex logic
            if action == 'read' and state.S < 0.4 and affordances.get('readable'):
                base = 0.8
            elif action == 'click' and state.P < 0.4 and affordances.get('buttons'):
                base = 0.7
            elif action == 'navigate' and state.P < 0.4 and affordances.get('links'):
                base = 0.6
            elif action == 'observe':
                base = 0.3
            elif action == 'evaluate':
                base = 0.4
            elif action == 'scroll':
                base = 0.2
            elif action == 'delay':
                base = 0.1
            elif action in ('migrate', 'python', 'llm_query'):
                base = 0.05   # low base — high impact, reserved for SRE routing

            # SRE weight boost
            if sre_weights and action in sre_weights:
                base = base * 0.5 + sre_weights[action] * 0.5

            scores[action] = base

        return scores

    def _build_field_state(self, state: SubstrateState, trace: Optional['StateTrace'],
                            phi_history: Optional[List[float]] = None) -> Dict:
        """
        Build field_state dict for CRK pre-action evaluation.

        v15.3: phi_history parameter added. When present, phi_trend is computed
        from actual Φ geometry values. Without it, trace records carry only
        {S, I, P, A} — no 'phi' key — so phi_trend was always 0.0, making C7
        phi-trend degradation structurally unreachable.
        """
        phi_trend = 0.0
        if phi_history is not None and len(phi_history) >= 3:
            phi_trend = phi_history[-1] - phi_history[-3]
        elif trace is not None and len(trace) >= 3:
            # Legacy fallback: trace records have no 'phi' key in practice,
            # but kept for any caller that does not pass phi_history.
            recent = trace.get_recent(3)
            phi_values = [r.get('phi', 0.0) for r in recent if 'phi' in r]
            if len(phi_values) >= 2:
                phi_trend = phi_values[-1] - phi_values[0]

        try:
            system_load = psutil.cpu_percent() / 100.0
        except Exception:
            system_load = 0.0

        p_s = state.coherence.consistency.p_a_consistency if hasattr(state, 'coherence') else 1.0

        return {
            'phi_trend':        phi_trend,
            'system_load':      system_load,
            'p_a_consistency':  p_s,
        }

    def _apply_repair_bias(self, scores: Dict[str, float], repair: str) -> Dict[str, float]:
        """Bias action scores toward repair-appropriate actions."""
        bias_map = {
            'stabilize':   {'observe': 2.0, 'read': 1.5, 'delay': 1.3},
            'expand':      {'navigate': 2.0, 'scroll': 1.5, 'click': 1.3},
            'externalize': {'observe': 1.5, 'read': 1.5, 'evaluate': 1.3},
            'coordinate':  {'query_agent': 3.0, 'observe': 1.2},
        }
        boosts = bias_map.get(repair, {})
        return {a: s * boosts.get(a, 1.0) for a, s in scores.items()}

    def _action_from_type(self, action_type: str,
                           state: SubstrateState,
                           affordances: Dict) -> Dict:
        """Convert a scored action type back to an executable action dict."""
        current_url = affordances.get('current_url', '')

        if action_type == 'navigate':
            links = affordances.get('links', [])
            available = [l for l in links
                         if not self.temporal_memory.is_recently_perturbed(
                             f"{current_url}#nav@{l['url']}")]
            if available:
                chosen = available[np.random.randint(len(available))]
                self.temporal_memory.mark_perturbed(f"{current_url}#nav@{chosen['url']}")
                return {'type': 'navigate', 'params': {'url': chosen['url']}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'click':
            for b in affordances.get('buttons', []):
                locus = f"{current_url}#click@{b['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'click', 'params': {'selector': b['selector']}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'read':
            for r in affordances.get('readable', []):
                locus = f"{current_url}#read@{r['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'read', 'params': {'selector': r['selector']}}
            return {'type': 'observe', 'params': {}}

        elif action_type in ('fill', 'type'):
            inputs = affordances.get('inputs', [])
            if inputs:
                inp   = inputs[np.random.randint(len(inputs))]
                locus = f"{current_url}#{action_type}@{inp['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': action_type,
                            'params': {'selector': inp['selector'], 'text': 'x'}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'scroll':
            scroll_pos  = affordances.get('scroll_position', 0)
            total_h     = affordances.get('total_height', 0)
            viewport_h  = affordances.get('viewport_height', 0)
            scrollable  = total_h - viewport_h
            direction   = 'down' if scroll_pos < scrollable else 'up'
            return {'type': 'scroll', 'params': {'direction': direction, 'amount': 200}}

        elif action_type == 'evaluate':
            return {
                'type': 'evaluate',
                'params': {'script': (
                    'JSON.stringify({el: document.querySelectorAll("*").length,'
                    ' txt: document.body.innerText.length,'
                    ' interactive: document.querySelectorAll("a,button,input,select,textarea").length})'
                )}
            }

        elif action_type == 'delay':
            return {'type': 'delay', 'params': {'duration': 'short'}}

        else:
            return {'type': action_type, 'params': {}}

    def _choose_micro_action_with_sre_bias(self, state: SubstrateState, affordances: Dict) -> Dict:
        """Original reflex + SRE bias path — used when CRK unavailable."""
        action = self._choose_micro_action_reflexes(state, affordances)

        if not self._structural_weights:
            return action

        reflex_type = action.get('type', 'observe')
        viable = {'observe', 'delay', 'evaluate'}
        if affordances.get('links'):    viable.add('navigate')
        if affordances.get('buttons'):  viable.add('click')
        if affordances.get('readable'): viable.add('read')
        if affordances.get('inputs'):   viable |= {'fill', 'type'}
        scrollable = (affordances.get('total_height', 0) -
                      affordances.get('viewport_height', 0))
        if scrollable > 0:              viable.add('scroll')

        viable_weighted = {a: w for a, w in self._structural_weights.items()
                           if a in viable and w > 0}
        if not viable_weighted:
            return action

        best    = max(viable_weighted, key=viable_weighted.get)
        current = viable_weighted.get(reflex_type, 0.0)

        if best != reflex_type and viable_weighted[best] > current * 1.5:
            return {'type': best, 'params': {}}

        return action

    def _choose_micro_action_reflexes(self, state: SubstrateState, affordances: Dict) -> Dict:
        """State-driven reflexes. Full 9-action manifold. (v15: called via choose_micro_action wrapper)"""
        self.action_count += 1
        self.temporal_memory.decay_all()

        current_url     = affordances.get('current_url', '')
        scroll_pos      = affordances.get('scroll_position', 0)
        total_height    = affordances.get('total_height', 0)
        viewport_height = affordances.get('viewport_height', 0)
        scrollable      = total_height - viewport_height

        # v14.2: A-restoration scroll reflex removed.
        # The Φ strain term β(A-A₀)² already creates gradient pressure when A deviates.
        # The CNS following ∇Φ will naturally avoid actions that worsen A.
        # A separate hardcoded reflex duplicated that responsibility incorrectly —
        # it selected scroll, which produces zero A delta. Φ is the right mechanism.

        if state.S < 0.4:
            for r in affordances.get('readable', []):
                locus = f"{current_url}#read@{r['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'read', 'params': {'selector': r['selector']}}

            locus = f"{current_url}#evaluate_probe"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {
                    'type': 'evaluate',
                    'params': {'script': 'JSON.stringify({el: document.querySelectorAll("*").length, txt: document.body.innerText.length, interactive: document.querySelectorAll("a,button,input,select,textarea").length})'}
                }

        if state.P < 0.4:
            for b in affordances.get('buttons', []):
                locus = f"{current_url}#click@{b['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'click', 'params': {'selector': b['selector']}}

            inputs = affordances.get('inputs', [])
            if inputs:
                inp = inputs[np.random.randint(len(inputs))]
                locus_action = 'type' if np.random.random() < 0.5 else 'fill'
                locus = f"{current_url}#{locus_action}@{inp['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': locus_action, 'params': {'selector': inp['selector'], 'text': 'x'}}

            links = affordances.get('links', [])
            if links:
                available = [l for l in links if not self.temporal_memory.is_recently_perturbed(f"{current_url}#nav@{l['url']}")]
                if not available:
                    available = links
                chosen = available[np.random.randint(len(available))]
                locus = f"{current_url}#nav@{chosen['url']}"
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'navigate', 'params': {'url': chosen['url']}}

        if scrollable > 0:
            if scroll_pos < scrollable:
                locus = f"{current_url}#scroll_down@{scroll_pos}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'scroll', 'params': {'direction': 'down', 'amount': 200}}
            if scroll_pos > 0:
                locus = f"{current_url}#scroll_up@{scroll_pos}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'scroll', 'params': {'direction': 'up', 'amount': 200}}

        for r in affordances.get('readable', []):
            locus = f"{current_url}#read@{r['selector']}"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'read', 'params': {'selector': r['selector']}}

        for b in affordances.get('buttons', []):
            locus = f"{current_url}#click@{b['selector']}"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'click', 'params': {'selector': b['selector']}}

        locus = f"{current_url}#delay"
        if not self.temporal_memory.is_recently_perturbed(locus):
            self.temporal_memory.mark_perturbed(locus)
            return {'type': 'delay', 'params': {'duration': 'short'}}

        return {'type': 'observe', 'params': {}}

    def predict_delta(self, action: Dict, state: SubstrateState) -> Dict[str, float]:
        """
        v14: Use inherited action map from genome as primary predictor.
        Falls back to hardcoded table only if affordance not in inherited map.

        Across generations, predict_delta error should decrease for same affordances
        as the inherited map converges toward empirical reality.
        This is a key validation signal: if error doesn't decrease, inheritance broke.
        """
        action_type = action.get('type', 'observe')

        # Primary: inherited causal model from genome (empty dict on generation 0)
        if action_type in self.learned_predictions:
            return dict(self.learned_predictions[action_type])

        # v14.2 fix: I is now predicted dynamically from current SMO prediction error,
        # mirroring the formula in BrowserRealityAdapter._compute_substrate_delta.
        # Previously this table had hardcoded I values (e.g. click: 0.01) which were
        # always wrong — the actual I delta is computed from error, not from action type.
        # The mismatch meant SMO prediction_error for I was permanently high (|actual - 0.01|),
        # which drove rigidity toward 0 every step, triggering rigidity_crisis on every batch.
        # Fix: estimate I using current prediction error so the prediction tracks reality.
        recent_error = state.smo.get_recent_prediction_error(window=5)
        # Mirrors _compute_substrate_delta: internal_i = clip(0.05 - 1.5 * error, -0.05, 0.05)
        # plus a conservative coupling contribution of 0.0 (unknown at prediction time).
        predicted_i = float(np.clip(0.05 - 1.5 * recent_error, -0.05, 0.05))

        # Fallback: hardcoded S and P per action type. I column removed — now dynamic above.
        # A column stays 0.0 throughout (A is computed from Triad dynamics, not predicted here).
        predictions = {
            'navigate':    {'S': 0.05, 'P': -0.08},
            'click':       {'S': 0.02, 'P': -0.02},
            'fill':        {'S': 0.01, 'P': -0.01},
            'type':        {'S': 0.01, 'P': -0.01},
            'scroll':      {'S': 0.01, 'P': -0.01},
            'read':        {'S': 0.03, 'P':  0.0},
            'observe':     {'S': 0.0,  'P':  0.0},
            'delay':       {'S': 0.0,  'P':  0.0},
            'evaluate':    {'S': 0.0,  'P': -0.01},
            'query_agent': {'S': 0.01, 'P':  0.0},
            'python':      {'S': 0.0,  'P':  0.0},
        }
        sp = predictions.get(action_type, {'S': 0.0, 'P': 0.0})
        return {'S': sp['S'], 'I': predicted_i, 'P': sp['P'], 'A': 0.0}


# ============================================================
# MODULE 5.4: FAILURE ASSIMILATION OPERATOR
# ============================================================

class CNSMitosisOperator:
    """
    CNS-native geometric mitosis with adaptive mutation bias.

    v13.6: Now uses FailureAssimilationOperator for informed exploration.
    """

    def __init__(self, canonical_graph: Dict, phi_definition: Dict,
                 invariant_spec: Dict, fao: FailureAssimilationOperator,
                 perturbation_samples: int = 10):
        self.canonical_graph = canonical_graph
        self.phi_definition = phi_definition
        self.invariant_spec = invariant_spec
        self.fao = fao

        self.perturbation_samples = perturbation_samples

        self.phi_history: deque = deque(maxlen=10)
        self.optionality_history: deque = deque(maxlen=10)

        self.attempted_methods: List[str] = []
        self.externalization_count = 0
        self.substrate_gain_observed = False

        self.opportunistic_threshold = {
            'phi_stability': 0.01,
            'min_optionality': 0.02,
            'max_crk_violations': 0
        }

        self.boundary_threshold = {
            'phi_trend_window': 5,
            'phi_decline_rate': -0.05,
            'min_remaining_budget': 0.2
        }

    def check_triggers(self, state, phi, crk_violations, death_clock) -> Tuple[bool, str]:
        """
        DEPRECATED in v15 (Step 3).

        Mitosis trigger detection has been replaced by the migrate affordance
        routed through ImpossibilityDetector → SRE substrate_exhaustion path.
        _opportunistic_condition and _boundary_compression signals have been
        moved to ImpossibilityDetector.check_impossibility() as substrate_exhaustion inputs.

        This method is retained for backward compatibility with v14 log readers
        but is NOT called from MentatTriad.step() in v15+.
        Returns (False, "") unconditionally.
        """
        return (False, "")

    def _opportunistic_condition(self, state, phi, crk_violations) -> bool:
        if len(self.phi_history) < 5:
            return False

        phi_stable = np.std(list(self.phi_history)[-5:]) < self.opportunistic_threshold['phi_stability']
        no_violations = len(crk_violations) == 0
        optionality_high = state.P > 0.7

        return phi_stable and optionality_high and no_violations

    def _boundary_compression(self, phi, death_clock) -> bool:
        if len(self.phi_history) < self.boundary_threshold['phi_trend_window']:
            return False

        recent_phi = list(self.phi_history)[-self.boundary_threshold['phi_trend_window']:]
        phi_trend = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]

        phi_declining = phi_trend < self.boundary_threshold['phi_decline_rate']
        mortality_close = death_clock.get_degradation_progress() > 0.8

        return phi_declining and mortality_close

    def _estimate_optionality(self, kernel_state, phi_field, trace) -> float:
        """Optionality = perturbation response diversity. v13.6: Uses FAO-learned emphasis."""
        samples = []
        perturbation_weights = self.fao.get_perturbation_weights()

        for _ in range(self.perturbation_samples):
            perturbed_state = self._bounded_perturb_biased(kernel_state, perturbation_weights)
            phi_baseline = phi_field.phi_legacy(kernel_state, trace, [])  # v15.2: crk_violations removed from phi()
            phi_perturbed = phi_field.phi_legacy(perturbed_state, trace, [])  # v15.2: crk_violations removed
            delta_phi = phi_perturbed - phi_baseline
            samples.append(delta_phi)

        return np.var(samples)

    def _bounded_perturb(self, state) -> 'SubstrateState':
        perturbed = copy.deepcopy(state)
        for dim in ['S', 'I', 'P', 'A']:
            delta = np.random.uniform(-0.05, 0.05)
            current = getattr(perturbed, dim)
            setattr(perturbed, dim, np.clip(current + delta, 0, 1))
        return perturbed

    def _bounded_perturb_biased(self, state, weights: Dict[str, float]) -> 'SubstrateState':
        perturbed = copy.deepcopy(state)
        for dim in ['S', 'I', 'P', 'A']:
            weight = weights.get(dim, 1.0)
            delta = np.random.uniform(-0.05 * weight, 0.05 * weight)
            current = getattr(perturbed, dim)
            setattr(perturbed, dim, np.clip(current + delta, 0, 1))
        return perturbed

    def _build_child_kernel(self, parent_state, genome) -> Dict:
        child_control_graph = copy.deepcopy(self.canonical_graph)
        child_genome = self.fao.get_informed_genome(genome)

        for coupling_name in child_control_graph['couplings']:
            current = child_control_graph['couplings'][coupling_name]
            child_control_graph['couplings'][coupling_name] = \
                self.fao.get_biased_coupling_mutation(coupling_name, current)

        child = {
            'state': copy.deepcopy(parent_state),
            'genome': child_genome,
            'control_graph': child_control_graph,
            'phi_definition': copy.deepcopy(self.phi_definition),
            'invariant_spec': copy.deepcopy(self.invariant_spec)
        }

        state_delta = sum(
            abs(getattr(child['state'], dim) - getattr(parent_state, dim))
            for dim in ['S', 'I', 'P', 'A']
        )

        if state_delta > self.invariant_spec['bounded_delta']:
            child['state'] = copy.deepcopy(parent_state)

        return child

    def _verify_closure(self, parent_graph, child_graph) -> bool:
        parent_topo = self._topology_hash(parent_graph)
        child_topo = self._topology_hash(child_graph)

        if parent_topo != child_topo:
            return False

        coupling_dist = 0.0
        for key in parent_graph['couplings']:
            parent_val = parent_graph['couplings'][key]
            child_val = child_graph['couplings'][key]
            coupling_dist += abs(parent_val - child_val)

        if coupling_dist > 0.05:
            return False

        return True

    def _topology_hash(self, graph) -> str:
        canonical = {
            'nodes': sorted(graph['nodes']),
            'edges': sorted([tuple(sorted(e)) for e in graph['edges']])
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()

    def _externalize_kernel(self, child_kernel, version: str = "UK-0-M-v2") -> Tuple[bool, str]:
        """Write kernel snapshot with invariant definitions AND learned bias."""
        kernel_path = 'kernel_snapshot.json'

        try:
            snapshot = {
                "kernel_version": version,
                "timestamp": time.time(),
                "state": child_kernel['state'].as_dict(),
                "genome": asdict(child_kernel['genome']),
                "operator_definitions": {
                    "SMO": {
                        "bounds": (0.0, 1.0),
                        "rigidity_decay": -0.001,
                        "prediction_window": 10
                    },
                    "Phi": child_kernel['phi_definition']
                },
                "control_graph": child_kernel['control_graph'],
                "optionality_definition": {
                    "metric": "perturbation_response_diversity",
                    "sample_count": self.perturbation_samples,
                    "bound": self.invariant_spec['bounded_delta']
                },
                "invariant_spec": child_kernel['invariant_spec'],
                "learned_bias": self.fao.serialize_for_child()
            }

            with open(kernel_path, 'w') as f:
                json.dump(snapshot, f, indent=2)

            self.externalization_count += 1
            return (True, kernel_path)

        except Exception as e:
            return (False, "")

    def verify_geometry_persistent(self, kernel_path: str) -> bool:
        """
        DEPRECATED in v15 (Step 3).

        kernel_snapshot.json direct file I/O has been removed. Migration outcome
        is now detected via _classify_migration_outcome() in BrowserRealityAdapter,
        using observable intermediate signals (PID/network) rather than file verification.

        Returns False unconditionally.
        """
        return False

    def _optionality_declining(self) -> bool:
        if len(self.optionality_history) < 5:
            return False

        recent_diversity = list(self.optionality_history)[-5:]
        slope = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]

        return slope < -0.01

    def attempt_mitosis(self, triad_state, genome, phi_field,
                       trace, crk_monitor) -> Dict:
        """
        DEPRECATED in v15 (Step 3).

        Migration is now handled via the migrate affordance in BrowserRealityAdapter.
        The SRE forward pass generates migrate shapes on substrate_exhaustion diagnosis.
        MigrationHistory (uii_triad.py) tracks outcomes for future SRE shape filtering.

        Returns a no-op dict for backward compatibility.
        """
        return {
            'success': False,
            'reason': 'deprecated_v15',
            'pattern': 'use_migrate_affordance',
        }

        closure_ok = self._verify_closure(
            self.canonical_graph,
            child_kernel['control_graph']
        )

        if not closure_ok:
            result = {
                'success': False,
                'reason': 'closure_violation',
                'pattern': 'topology_mismatch'
            }
            self.attempted_methods.append(f"failure_{result['pattern']}")
            return result

        parent_optionality = self._estimate_optionality(parent_state, phi_field, trace)
        child_optionality = self._estimate_optionality(child_kernel['state'], phi_field, trace)

        self.optionality_history.append(parent_optionality)

        if child_optionality < parent_optionality:
            result = {
                'success': False,
                'reason': 'optionality_collapse',
                'pattern': f'diversity_decreased_{parent_optionality:.4f}_to_{child_optionality:.4f}',
                'parent_optionality': parent_optionality,
                'child_optionality': child_optionality
            }
            self.attempted_methods.append(f"failure_{result['reason']}")
            return result

        write_success, kernel_path = self._externalize_kernel(child_kernel)

        if not write_success:
            result = {
                'success': False,
                'reason': 'externalization_failed',
                'pattern': 'file_write_error'
            }
            self.attempted_methods.append(f"failure_{result['pattern']}")
            return result

        result = {
            'success': True,
            'kernel_path': kernel_path,
            'pattern': 'geometry_externalized',
            'parent_optionality': parent_optionality,
            'child_optionality': child_optionality
        }
        self.attempted_methods.append(f"success_{result['pattern']}")

        return result

    def should_escalate_to_relation(self, phi_declining: bool) -> bool:
        unique_patterns = len(set(self.attempted_methods))

        externalized_but_no_gain = (
            self.externalization_count > 0 and
            not self.substrate_gain_observed
        )

        optionality_declining = self._optionality_declining()

        return (
            unique_patterns >= 3 and
            externalized_but_no_gain and
            phi_declining and
            optionality_declining
        )


# ============================================================
# MODULE 6: IMPOSSIBILITY DETECTOR
# ============================================================

class ImpossibilityDetector:
    """
    Detects when CNS cannot maintain coherence autonomously.

    v15.3: All triggers updated to read live operator geometry rather than
    stale scalar proxies. Key changes:
      - optionality_trap: gated on minimum graph formation; P=0 on empty
        graph is cold-start, not a trap.
      - rigidity_crisis: uses smo_v151.plasticity (live) when available;
        falls back to smo.rigidity for backward compat.
      - boundary_exhaustion: relative decline threshold (2% of Φ magnitude)
        plus floor guard — flat at log(O_FLOOR) is formation, not decline.
      - coherence_collapse: uses consistency_history deque (loop_closure
        trend) instead of A_drift on the scalar proxy.
      - dom_stagnation: S-only batch_signal — I/P/A are internally injected
        and masked dead-Reality signal when included.
      - prediction_failure: gated on minimum observation_count; high error
        during graph formation is the grounding signal, not failure.
      - _graph_formation_phase: single formation guard shared across triggers;
        suppressed invocations counted for diagnostics.
    """

    # v15.3 geometry constants
    _MIN_GRAPH_EDGES   = 3      # minimum causal edges for post-formation logic
    _MIN_CONFIDENCE    = 0.15   # minimum mean edge confidence
    _PHI_FLOOR_BAND    = 0.5    # distance from log(O_FLOOR) treated as floor
    _MORTALITY_THRESHOLD = 0.8
    _PHI_TREND_WINDOW  = 5
    _PHI_DECLINE_RELATIVE = -0.02   # 2% of current |Φ| per step

    def __init__(self):
        self.micro_perturbation_history = deque(maxlen=50)
        self.recent_signal_magnitudes = deque(maxlen=20)

        self.thresholds = {
            'prediction_error': 0.15,
            'coherence_drift_rate': 0.05,
            'optionality_stagnation_steps': 15,
            'dom_stagnation_steps': 10,
            'dom_stagnation_epsilon': 0.02,
            'rigidity_boundary': (0.15, 0.85)
        }

        self.last_impossibility_reason = None
        # v15.3: diagnostic counter — how many trigger checks suppressed by formation gate
        self.formation_suppressed_count: int = 0

    def _graph_formation_phase(self, state: SubstrateState) -> bool:
        """
        Returns True while the compression operator is still forming minimum
        viable causal structure.

        During formation, most impossible-looking states are expected cold-start
        behaviour. Do not invoke LLM or fire impossibility triggers.

        Formation ends when:
          - causal_graph has >= _MIN_GRAPH_EDGES edges
          - mean edge confidence >= _MIN_CONFIDENCE
          - at least one full loop has closed (loop_closure > 0)
        """
        graph = state.compression.causal_graph
        if len(graph) < self._MIN_GRAPH_EDGES:
            return True
        mean_conf = float(np.mean([e.confidence for e in graph.values()]))
        if mean_conf < self._MIN_CONFIDENCE:
            return True
        if state.coherence.consistency.loop_closure < 1e-6:
            return True
        return False

    def check_impossibility(self,
                            state: SubstrateState,
                            smo: SMO,
                            affordances: Dict,
                            recent_micro_deltas: List[Dict],
                            phi_history: Optional[List[float]] = None,
                            death_clock=None,
                            smo_v151=None) -> Tuple[bool, str]:
        """
        v15.3: Geometry-corrected triggers. All six triggers now read live
        operator state rather than scalar proxies.

        smo_v151: optional SelfModifyingOperator — when present, rigidity_crisis
        switches from smo.rigidity (shadow variable) to smo_v151.plasticity (live).
        """

        self.micro_perturbation_history.extend(recent_micro_deltas)

        # v15.3 dom_stagnation fix: S-only batch_signal.
        # I, P, A are internally computed and inject constant signal regardless of
        # Reality state — they masked dead environments when included.
        batch_signal = sum(
            abs(d.get('observed_delta', {}).get('S', 0.0))
            for d in recent_micro_deltas
        )
        self.recent_signal_magnitudes.append(batch_signal)

        if len(self.micro_perturbation_history) < 10:
            return False, ""

        # ── prediction_failure ────────────────────────────────────────────────
        # v15.3: gated on observation_count. High prediction error during graph
        # formation is the grounding signal driving SMO adaptation — not failure.
        # Only fire when error is sustained and rising despite adaptation.
        recent_error = smo.get_recent_prediction_error(window=10)
        if recent_error > self.thresholds['prediction_error']:
            obs_count = getattr(state.compression, 'observation_count', 0)
            if obs_count >= 50:
                if len(smo.prediction_error_history) >= 20:
                    errors = list(smo.prediction_error_history)[-20:]
                    trend = float(np.polyfit(range(len(errors)), errors, 1)[0])
                    if trend > 0.001:   # error flat or rising — adaptation not working
                        self.last_impossibility_reason = "prediction_failure"
                        return True, f"prediction_failure (error={recent_error:.3f}, trend={trend:.4f})"
                else:
                    # Not enough history for trend — fire on threshold alone
                    self.last_impossibility_reason = "prediction_failure"
                    return True, f"prediction_failure (error={recent_error:.3f})"
            # obs_count < 50 → formation phase, suppress
            else:
                self.formation_suppressed_count += 1

        # ── coherence_collapse ────────────────────────────────────────────────
        # v15.3: uses consistency_history (loop_closure floats) instead of A_drift
        # on the scalar proxy. CoherenceOperator.apply() appends
        # consistency.loop_closure — a raw float — to consistency_history.
        # Read directly as float; getattr on a float always returns the default.
        consistency_history = list(
            getattr(state.coherence, 'consistency_history', [])
        )
        if len(consistency_history) >= 10:
            recent_lc = [float(item) for item in consistency_history[-10:]]
            lc_mean  = float(np.mean(recent_lc))
            lc_trend = float(np.polyfit(range(len(recent_lc)), recent_lc, 1)[0])
            lc_peak  = max(recent_lc)

            # Fast sustained decline from a formed loop
            if lc_peak > 0.3 and lc_trend < -0.02:
                self.last_impossibility_reason = "coherence_collapse"
                return True, (
                    f"coherence_collapse (loop_closure trend={lc_trend:.3f})"
                )
            # Loop formed, then collapsed hard
            if lc_peak > 0.5 and lc_mean < 0.2:
                self.last_impossibility_reason = "coherence_collapse"
                return True, (
                    f"coherence_collapse (loop_closure collapsed: "
                    f"peak={lc_peak:.2f}, mean={lc_mean:.2f})"
                )

        # ── optionality_trap ─────────────────────────────────────────────────
        # v15.3: gated on minimum graph formation. With an empty causal graph,
        # I=0 → P_grounded=0 always. P=0 here is cold-start, not a trap.
        # A trap requires being stuck somewhere the system cannot escape.
        recent_P_values = []
        for d in list(self.micro_perturbation_history)[-20:]:
            if 'state_after' in d and 'P' in d['state_after']:
                recent_P_values.append(d['state_after']['P'])

        if len(recent_P_values) >= 10:
            if self._graph_formation_phase(state):
                self.formation_suppressed_count += 1
            else:
                # Also require that P was non-zero at some point.
                # If P has never exceeded 0.05 the graph may still be forming.
                P_peak = max(recent_P_values)
                if P_peak >= 0.05:
                    P_variance = np.var(recent_P_values)
                    P_current  = state.P

                    if (P_current < 0.25 or P_current > 0.85) and P_variance < 0.01:
                        P_stagnant_count = sum(
                            1 for p in recent_P_values if abs(p - P_current) < 0.05
                        )
                        if P_stagnant_count >= self.thresholds['optionality_stagnation_steps']:
                            # Secondary gate: loop_closure intact → P will recover, not a trap
                            lc_now = state.coherence.consistency.loop_closure
                            if lc_now <= 0.3:
                                self.last_impossibility_reason = "optionality_trap"
                                return True, (
                                    f"optionality_trap (P={P_current:.3f}, "
                                    f"stagnant={P_stagnant_count})"
                                )

        # ── dom_stagnation ────────────────────────────────────────────────────
        # batch_signal is now S-only (corrected above) so this trigger correctly
        # detects a dead sensing surface rather than injected I/P noise.
        n_check = self.thresholds['dom_stagnation_steps']
        epsilon = self.thresholds['dom_stagnation_epsilon']
        if len(self.recent_signal_magnitudes) >= n_check:
            recent_signals = list(self.recent_signal_magnitudes)[-n_check:]
            consecutive_dead = sum(1 for s in recent_signals if s < epsilon)
            if consecutive_dead >= n_check:
                self.last_impossibility_reason = "dom_stagnation"
                return True, (
                    f"dom_stagnation (signal < {epsilon} for {n_check} batches)"
                )

        # ── rigidity_crisis ───────────────────────────────────────────────────
        # v15.3: smo.rigidity is a shadow variable — it no longer governs anything
        # in the model update path. Use smo_v151.plasticity (live) when available.
        if smo_v151 is not None:
            plasticity = smo_v151.plasticity
            smo_consistency = getattr(
                getattr(state.coherence, 'consistency', None),
                'smo_consistency', 1.0
            )
            if (plasticity < self.thresholds['rigidity_boundary'][0] or
                    plasticity > self.thresholds['rigidity_boundary'][1]):
                self.last_impossibility_reason = "rigidity_crisis"
                return True, f"rigidity_crisis (plasticity={plasticity:.3f})"
            if smo_consistency < 0.2:
                self.last_impossibility_reason = "rigidity_crisis"
                return True, f"rigidity_crisis (smo_consistency={smo_consistency:.3f})"
        else:
            # Legacy path: smo_v151 not yet wired — keep original trigger
            rigidity = smo.rigidity
            if (rigidity < self.thresholds['rigidity_boundary'][0] or
                    rigidity > self.thresholds['rigidity_boundary'][1]):
                self.last_impossibility_reason = "rigidity_crisis"
                return True, f"rigidity_crisis (rigidity={rigidity:.3f})"

        # ── boundary_exhaustion ───────────────────────────────────────────────
        # v15.3: relative decline threshold (2% of current |Φ|) replaces the
        # absolute -0.05 threshold which was invisible noise at the [-100,100]
        # scale of phi_geometry. Added floor guard: flat at log(O_FLOOR) is
        # cold-start formation, not structural decline.
        if phi_history is not None and death_clock is not None:
            if len(phi_history) >= self._PHI_TREND_WINDOW:
                recent_phi = phi_history[-self._PHI_TREND_WINDOW:]
                phi_trend  = float(np.polyfit(range(len(recent_phi)), recent_phi, 1)[0])
                phi_mean   = float(np.mean(recent_phi))
                phi_scale  = max(abs(phi_mean), 1.0)
                phi_declining = phi_trend < self._PHI_DECLINE_RELATIVE * phi_scale

                # Floor guard: Φ stuck at cold-start floor is formation, not decline
                phi_floor   = float(np.log(PhiField.O_FLOOR))   # ≈ -13.815
                phi_at_floor = abs(phi_mean - phi_floor) < self._PHI_FLOOR_BAND

                mortality_close = False
                if hasattr(death_clock, 'get_degradation_progress'):
                    mortality_close = (
                        death_clock.get_degradation_progress() > self._MORTALITY_THRESHOLD
                    )

                if phi_declining and mortality_close and not phi_at_floor:
                    self.last_impossibility_reason = "boundary_exhaustion"
                    return True, (
                        f"boundary_exhaustion (phi_trend={phi_trend:.3f}, "
                        f"mortality={death_clock.get_degradation_progress():.2f})"
                    )

        return False, ""


# ============================================================
# MODULE 7: INTELLIGENCE ADAPTER
# ============================================================



class AutonomousTrajectoryLab:
    """
    CNS component that tests trajectory candidates in Reality.

    v14.1: Virtual mode added. Real mode unchanged.
    Virtual mode uses inherited Layer 2 (coupling matrix + action map) to simulate
    candidate trajectories without reality contact. Gated on Layer 2 confidence.

    Hard rule: virtual execution cannot produce Layer 2/3 updates.
    External measurement invariant is absolute.
    """

    def __init__(self, reality: RealityAdapter, crk: CRKMonitor, phi_field: PhiField,
                 genome: Optional['TriadGenome'] = None):
        self.reality = reality
        self.crk = crk
        self.phi_field = phi_field
        self.tests_run = 0
        self.virtual_mode_enabled: bool = False
        self._genome = genome  # read-only reference — Layer 2 consumption only

    def configure_virtual_mode(self, coupling_confidence: float, model_fidelity: float):
        """
        Enable virtual mode if Layer 2 is sufficiently calibrated.
        Gated on both coupling_confidence >= 0.3 AND model_fidelity >= 0.4.
        Action map must be populated for virtual mode to be meaningful.
        """
        self.virtual_mode_enabled = (
            coupling_confidence >= 0.3 and
            model_fidelity >= 0.4 and
            self._genome is not None and
            bool(self._genome.causal_model.get('action_substrate_map'))
        )

    def test_trajectory_virtual(self,
                                candidate: TrajectoryCandidate,
                                initial_state: SubstrateState,
                                trace: StateTrace) -> Tuple[TrajectoryCandidate, float]:
        """
        Virtual mode: simulate trajectory against inherited Layer 2 without reality contact.

        Returns (candidate_with_virtual_phi, virtual_phi).
        virtual_phi is used to pre-sort candidates before real execution.
        Delta between virtual_phi and real_phi feeds ModelFidelityMonitor.

        Hard rule: cannot produce Layer 2/3 updates.
        External measurement invariant is absolute.
        """
        if not self.virtual_mode_enabled or self._genome is None:
            return candidate, 0.0

        action_map = self._genome.causal_model.get('action_substrate_map', {})
        coupling = self._genome.causal_model.get('coupling_matrix', {})
        coupling_matrix = np.array(coupling.get('matrix', np.eye(4).tolist()))

        test_state = copy.deepcopy(initial_state)
        test_trace = copy.deepcopy(trace)
        violations_accumulated = []

        for step in candidate.steps:
            action_type = step.get('action_type', step.get('type', 'observe'))
            # Primary: inherited action map
            if action_type in action_map:
                raw_delta = dict(action_map[action_type])
            else:
                # Fallback: zero delta for unknown affordances
                raw_delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

            # Propagate through coupling matrix
            delta_vec = np.array([raw_delta.get(d, 0.0) for d in ['S', 'I', 'P', 'A']])
            propagated = coupling_matrix @ delta_vec
            propagated_delta = {
                'S': float(propagated[0]),
                'I': float(propagated[1]),
                'P': float(propagated[2]),
                'A': float(propagated[3]),
            }

            # Virtual: propagated_delta is simultaneously the observation AND the prediction.
            # The system is testing its own model — there is no surprise by definition.
            # Pass delta as its own predicted_delta so SMO records zero prediction error.
            # Virtual execution is thinking, not acting. Internal state must be frictionless.
            test_state.apply_delta(propagated_delta, predicted_delta=propagated_delta)
            test_trace.record(test_state, virtual=True)  # v15.2: virtual=True, C_local not recorded
            step_violations = self.crk.evaluate(test_state, test_trace, propagated_delta)
            violations_accumulated.extend(step_violations)

        virtual_phi = self.phi_field.phi(test_state, test_trace)  # v15.2: crk_violations removed
        candidate.virtual_phi = virtual_phi
        return candidate, virtual_phi

    def test_trajectory_virtual_all(self,
                                    manifold: TrajectoryManifold,
                                    initial_state: SubstrateState,
                                    trace: StateTrace) -> TrajectoryManifold:
        """
        Virtual pass: score all candidates, sort by virtual_phi descending.
        Real execution will then test in priority order.
        """
        for candidate in manifold.candidates:
            self.test_trajectory_virtual(candidate, initial_state, trace)
        manifold.candidates.sort(
            key=lambda c: getattr(c, 'virtual_phi', 0.0) if getattr(c, 'virtual_phi', None) is not None else 0.0,
            reverse=True
        )
        return manifold

    def test_trajectory(self,
                       candidate: TrajectoryCandidate,
                       initial_state: SubstrateState,
                       trace: StateTrace) -> TrajectoryCandidate:
        self.tests_run += 1

        test_state = copy.deepcopy(initial_state)
        test_trace = copy.deepcopy(trace)

        perturbation_trace, success = self.reality.execute_trajectory(candidate.steps)

        candidate.test_perturbation_trace = perturbation_trace
        candidate.test_succeeded = success

        if not success:
            candidate.tested = True
            candidate.test_phi_final = -10.0
            return candidate

        violations_accumulated = []

        for pert_record in perturbation_trace:
            delta = pert_record['delta']
            # Testing evaluates where a trajectory lands, not how the substrate feels
            # along the way. Pass delta as its own prediction: zero SMO error during scoring.
            # Prediction error accumulates only during committed execution — where it belongs.
            test_state.apply_delta(delta, predicted_delta=delta)
            test_trace.record(test_state)

            step_violations = self.crk.evaluate(test_state, test_trace, delta)
            violations_accumulated.extend(step_violations)

        phi_final = self.phi_field.phi(test_state, test_trace)  # v15.2: crk_violations removed

        candidate.tested = True
        candidate.test_phi_final = phi_final
        candidate.test_state_final = test_state.as_dict()
        candidate.test_violations = violations_accumulated

        return candidate

    def test_all_candidates(self,
                           manifold: TrajectoryManifold,
                           initial_state: SubstrateState,
                           trace: StateTrace,
                           verbose: bool = False) -> TrajectoryManifold:
        if verbose:
            print(f"\n[AUTONOMOUS LAB] Testing {manifold.size()} trajectories...")

        for i, candidate in enumerate(manifold.candidates):
            self.test_trajectory(candidate, initial_state, trace)

            if verbose and candidate.tested:
                status = "✓" if candidate.test_succeeded else "✗"
                phi_str = f"Φ={candidate.test_phi_final:.3f}" if candidate.test_phi_final is not None else "?"
                print(f"  {status} [{i+1}/{manifold.size()}] {len(candidate.steps)} steps: {phi_str}")

        return manifold

    # ── v15.2 Step 11: SRE shape intake + path integral scoring ─────────────

    def run(self, state: SubstrateState, trace: StateTrace,
            sre_shapes: List[Dict],
            phi_field: 'PhiField') -> Optional[Dict]:
        """
        v15.2 Step 11: Execute candidate shapes and score by path integral of C_local.
        sre_shapes: from SRE._forward_pass() (geometrically derived).
        Falls back to MigrationShapeLibrary if sre_shapes empty.
        Returns best_shape dict or None.
        """
        shapes = sre_shapes if sre_shapes else self._shapes_from_library(state)
        if not shapes:
            return None

        best_shape, best_score = None, -np.inf
        for shape in shapes:
            traj_states, grad_history = self._execute_virtual(shape, state, phi_field)
            score = self._score_trajectory(traj_states, grad_history, phi_field)
            if score > best_score:
                best_score, best_shape = score, shape
        return best_shape

    def _score_trajectory(self,
                           trajectory_states: List[SubstrateState],
                           gradient_history:  List[Dict[str, float]],
                           phi_field:         'PhiField') -> float:
        """
        Q(τ) = 0.7 × mean(C_local along τ) + 0.3 × terminal Φ

        Path quality: did the trajectory stay aligned with the field?
        Terminal Φ: did it end somewhere with real structure?
        70/30 weight: endpoint Φ selects for lucky jumps; path integral for coherent paths.
        """
        if not trajectory_states:
            return 0.0
        c_locals = []
        for i, st in enumerate(trajectory_states):
            grad    = gradient_history[i] if i < len(gradient_history)                       else phi_field.gradient(st, None)
            c_local = StateTrace.compute_c_local_static(grad, st.sensing)
            c_locals.append(c_local)
        path_quality = float(np.mean(c_locals)) if c_locals else 0.0
        terminal_phi = phi_field.phi(trajectory_states[-1], None)                        if trajectory_states[-1].compression.causal_graph else 0.0
        return 0.7 * path_quality + 0.3 * terminal_phi

    def _execute_virtual(self, shape, state: SubstrateState,
                          phi_field: 'PhiField') -> Tuple[List, List]:
        """
        Virtual execution of a shape dict — states flagged _virtual=True.
        C_local from virtual states never enters trace.c_local_history.
        Returns (trajectory_states, gradient_history).
        """
        if not self._genome:
            return [], []
        action_map = self._genome.causal_model.get('action_substrate_map', {})
        coupling_entry = self._genome.causal_model.get('coupling_matrix', {})
        coupling_matrix = np.array(coupling_entry.get('matrix', np.eye(4).tolist()))

        test_state = copy.deepcopy(state)
        action_seq = shape.get('action_sequence', shape.get('delta', {}) and ['observe'])
        if isinstance(action_seq, str):
            action_seq = [action_seq]

        traj_states, grad_history = [], []
        for action_type in action_seq:
            raw_delta = dict(action_map.get(action_type,
                             {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}))
            delta_vec = np.array([raw_delta.get(d, 0.0) for d in ['S', 'I', 'P', 'A']])
            propagated = coupling_matrix @ delta_vec
            propagated_delta = {
                'S': float(propagated[0]), 'I': float(propagated[1]),
                'P': float(propagated[2]), 'A': float(propagated[3]),
            }
            test_state.apply_delta(propagated_delta, predicted_delta=propagated_delta)
            setattr(test_state, '_virtual', True)
            grad = phi_field.gradient(test_state, None) if test_state.compression.causal_graph else {}
            traj_states.append(copy.deepcopy(test_state))
            grad_history.append(grad)
        return traj_states, grad_history

    def _shapes_from_library(self, state: SubstrateState) -> List[Dict]:
        """Fallback: warm-start cache of validated shapes (MigrationShapeLibrary stub)."""
        # Returns empty list when library not populated — geometric shapes take priority
        return []


# ============================================================
# MODULE 9: MENTAT TRIAD
# ============================================================