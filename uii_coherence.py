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

    def check_activation(self, smo: SMO, trace: StateTrace) -> bool:
        """ENO activates when prediction error is low."""
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
    """Selects highest-control pattern cluster from discovered structure."""

    def __init__(self):
        self.cluster_history: deque = deque(maxlen=20)
        self.zero_control_counter = 0
        self.zero_control_threshold = 3

    def discover_and_select_pattern(self,
                                    eno: ExteriorNecessitationOperator,
                                    cam: ControlAsymmetryMeasure) -> Tuple[Set[str], List[Set[str]], Dict]:
        """Discover pattern structure and select best cluster."""
        viable = eno.get_viable_affordances()

        graph = cam.build_covariance_graph(viable)

        clusters = cam.extract_pattern_clusters(graph, threshold=0.01)

        if not clusters:
            clusters = [viable]

        cluster_controls = {}
        for i, cluster in enumerate(clusters):
            control = cam.measure_cluster_control(cluster)
            cluster_controls[i] = {
                'cluster': cluster,
                'control': control
            }

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
        """Boundary pressure as environmental constant. Fixed at 0.85."""
        return 0.85

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


class ContinuousRealityEngine:
    """CNS-driven micro-perturbation system."""

    def __init__(self, reality: RealityAdapter, inherited_action_map: Optional[Dict] = None):
        self.reality = reality
        self.action_count = 0

        self.temporal_memory = TemporalPerturbationMemory(
            window_steps=5,
            capacity=20
        )
        # v14: Inherited action map from genome (primary predictor).
        # Falls back to hardcoded table only if affordance not present.
        # On generation 0 this is empty — hardcoded table used for everything.
        self.learned_predictions: Dict = inherited_action_map or {}

    def choose_micro_action(self, state: SubstrateState, affordances: Dict) -> Dict:
        """State-driven reflexes. Full 9-action manifold."""
        self.action_count += 1
        self.temporal_memory.decay_all()

        if affordances.get('bootstrap_state', False):
            return {'type': 'observe', 'params': {}}

        current_url     = affordances.get('current_url', '')
        scroll_pos      = affordances.get('scroll_position', 0)
        total_height    = affordances.get('total_height', 0)
        viewport_height = affordances.get('viewport_height', 0)
        scrollable      = total_height - viewport_height

        if state.P > 0.7:
            return {'type': 'observe', 'params': {}}

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

        # Fallback: hardcoded table (v14.2: A column zeroed throughout, query_agent + python added)
        # Actions do not predict A delta — A is computed from Triad dynamics in the step loop.
        # Non-zero A predictions here produced permanent structural prediction error,
        # contaminating I quality (which now uses prediction error as its primary signal).
        # query_agent and python were previously absent — fell through to all-zeros default,
        # generating prediction error on every invocation.
        predictions = {
            'navigate':    {'S': 0.05, 'I': 0.03, 'P': -0.08, 'A': 0.0},
            'click':       {'S': 0.02, 'I': 0.01, 'P': -0.02, 'A': 0.0},
            'fill':        {'S': 0.01, 'I': 0.02, 'P': -0.01, 'A': 0.0},
            'type':        {'S': 0.01, 'I': 0.02, 'P': -0.01, 'A': 0.0},
            'scroll':      {'S': 0.01, 'I': 0.0,  'P': -0.01, 'A': 0.0},
            'read':        {'S': 0.03, 'I': 0.02, 'P':  0.0,  'A': 0.0},
            'observe':     {'S': 0.0,  'I': 0.0,  'P':  0.0,  'A': 0.0},
            'delay':       {'S': 0.0,  'I': 0.0,  'P':  0.0,  'A': 0.0},
            'evaluate':    {'S': 0.0,  'I': 0.01, 'P': -0.01, 'A': 0.0},
            'query_agent': {'S': 0.01, 'I': 0.0,  'P':  0.0,  'A': 0.0},
            'python':      {'S': 0.0,  'I': 0.02, 'P':  0.0,  'A': 0.0},
        }
        return predictions.get(action_type, {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0})


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
        self.phi_history.append(phi)

        if self._opportunistic_condition(state, phi, crk_violations):
            return (True, "opportunistic_high_coherence")

        if self._boundary_compression(phi, death_clock):
            return (True, "boundary_compression_survival")

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
            phi_baseline = phi_field.phi(kernel_state, trace, [])
            phi_perturbed = phi_field.phi(perturbed_state, trace, [])
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
        if not Path(kernel_path).exists():
            return False

        try:
            with open(kernel_path) as f:
                loaded = json.load(f)

            loaded_topo_hash = self._topology_hash(loaded['control_graph'])
            parent_topo_hash = self._topology_hash(self.canonical_graph)

            if loaded_topo_hash != parent_topo_hash:
                return False

            if loaded.get('operator_definitions', {}).get('Phi') != self.phi_definition:
                return False

            self.substrate_gain_observed = True
            return True

        except Exception as e:
            return False

    def _optionality_declining(self) -> bool:
        if len(self.optionality_history) < 5:
            return False

        recent_diversity = list(self.optionality_history)[-5:]
        slope = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]

        return slope < -0.01

    def attempt_mitosis(self, triad_state, genome, phi_field,
                       trace, crk_monitor) -> Dict:
        parent_state = triad_state['substrate_state']
        child_kernel = self._build_child_kernel(parent_state, genome)

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
    """Detects when CNS cannot maintain coherence autonomously."""

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

    def check_impossibility(self,
                           state: SubstrateState,
                           smo: SMO,
                           affordances: Dict,
                           recent_micro_deltas: List[Dict]) -> Tuple[bool, str]:
        if affordances.get('bootstrap_state', False):
            return True, "bootstrap_migration"

        self.micro_perturbation_history.extend(recent_micro_deltas)

        batch_signal = sum(
            sum(abs(d.get('observed_delta', {}).get(dim, 0.0)) for dim in ['S', 'I', 'P', 'A'])
            for d in recent_micro_deltas
        )
        self.recent_signal_magnitudes.append(batch_signal)

        if len(self.micro_perturbation_history) < 10:
            return False, ""

        recent_error = smo.get_recent_prediction_error(window=10)
        if recent_error > self.thresholds['prediction_error']:
            self.last_impossibility_reason = "prediction_failure"
            return True, f"prediction_failure (error={recent_error:.3f})"

        recent_states = [d.get('state_after') for d in list(self.micro_perturbation_history)[-10:] if 'state_after' in d]
        if len(recent_states) >= 5:
            A_values = [s['A'] for s in recent_states if 'A' in s]
            if A_values:
                A_drift = np.std(A_values)
                if A_drift > self.thresholds['coherence_drift_rate']:
                    self.last_impossibility_reason = "coherence_collapse"
                    return True, f"coherence_collapse (A_drift={A_drift:.3f})"

        P_stagnant_count = 0
        recent_P_values = []
        for d in list(self.micro_perturbation_history)[-20:]:
            if 'state_after' in d and 'P' in d['state_after']:
                recent_P_values.append(d['state_after']['P'])

        if len(recent_P_values) >= 10:
            P_variance = np.var(recent_P_values)
            P_current = state.P

            if (P_current < 0.25 or P_current > 0.85) and P_variance < 0.01:
                P_stagnant_count = sum(1 for p in recent_P_values if abs(p - P_current) < 0.05)
                if P_stagnant_count >= self.thresholds['optionality_stagnation_steps']:
                    self.last_impossibility_reason = "optionality_trap"
                    return True, f"optionality_trap (P={P_current:.3f}, stagnant={P_stagnant_count})"

        n_check = self.thresholds['dom_stagnation_steps']
        epsilon = self.thresholds['dom_stagnation_epsilon']
        if len(self.recent_signal_magnitudes) >= n_check:
            recent_signals = list(self.recent_signal_magnitudes)[-n_check:]
            consecutive_dead = sum(1 for s in recent_signals if s < epsilon)
            if consecutive_dead >= n_check:
                self.last_impossibility_reason = "dom_stagnation"
                return True, f"dom_stagnation (signal < {epsilon} for {n_check} batches)"

        rigidity = smo.rigidity
        if rigidity < self.thresholds['rigidity_boundary'][0] or rigidity > self.thresholds['rigidity_boundary'][1]:
            self.last_impossibility_reason = "rigidity_crisis"
            return True, f"rigidity_crisis (rigidity={rigidity:.3f})"

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
            test_trace.record(test_state)
            step_violations = self.crk.evaluate(test_state, test_trace, propagated_delta)
            violations_accumulated.extend(step_violations)

        virtual_phi = self.phi_field.phi(test_state, test_trace, violations_accumulated)
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

        phi_final = self.phi_field.phi(test_state, test_trace, violations_accumulated)

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


# ============================================================
# MODULE 9: MENTAT TRIAD
# ============================================================