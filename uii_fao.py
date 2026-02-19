from __future__ import annotations  # lazy annotations -- breaks circular dep with uii_coherence

"""
UII v14.1 — uii_fao.py
Failure Assimilation Operator & Residual Learning

Role: FAO is the legal mechanism by which Relation failures inform CNS search geometry.
It operates on the mutation distribution, not on CNS behavior.
Input: observable signals only (Φ delta, CRK violation types, trajectory outcomes).
Output: mutation distribution shape. Never action selection.

Also contains the residual learning stack that feeds the FAO and genome:
  - ResidualTracker (prediction errors + context signals per perturbation)
  - ResidualExplainer (escalation ladder: coupling → nonlinear → lag → candidate)
  - AxisAdmissionTest (4-condition gate: predictive/compression/shuffle/phi)
  - classify_relation_failure (observable-signal-only failure classification)
  - FailureAssimilationOperator (mutation bias + distill_to_genome)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import copy
import time
from collections import deque

from uii_types import (
    LAYER1_PARAMS, SUBSTRATE_DIMS,
    INTERFACE_COUPLED_SIGNALS, POTENTIALLY_INVARIANT_SIGNALS,
    SubstrateState, TrajectoryCandidate,
)
from uii_genome import TriadGenome, ProvisionalAxisManager
from uii_reality import CouplingMatrixEstimator

class ResidualTracker:
    """
    Tracks prediction error with context signals per perturbation step.

    Records (predicted_delta, observed_delta, context_signals) each step.
    Context signals are pre-classified at extraction:
    - POTENTIALLY_INVARIANT signals: allowed to proceed to axis admission
    - INTERFACE_COUPLED signals: blocked here, never recorded

    This classification happens at extraction, not in AxisAdmissionTest.
    Interface-coupled signals are excluded before they can accumulate evidence.
    """

    def __init__(self, maxlen: int = 200):
        self.records: deque = deque(maxlen=maxlen)

    def record(self, predicted: Dict, observed: Dict, reality_context: Dict):
        """Record one step's prediction error with pre-filtered context signals."""
        residual = {
            dim: observed.get(dim, 0.0) - predicted.get(dim, 0.0)
            for dim in SUBSTRATE_DIMS
        }
        before = reality_context.get('before', {})
        after = reality_context.get('after', {})
        context_signals = self._extract_invariant_signals(before, after, reality_context)
        self.records.append({
            'residual': residual,
            'predicted': predicted,
            'observed': observed,
            'context': context_signals,
        })

    def _extract_invariant_signals(self, before: Dict, after: Dict,
                                    full_context: Dict) -> Dict[str, float]:
        """
        Extract context signals that may survive substrate change.
        Interface-coupled signals (DOM artifacts) are deliberately excluded.
        Test: would this signal exist if we changed browsers? If not, exclude it.
        """
        signals = {}

        # Content entropy: text per element (information density, not raw count)
        text_after = after.get('text_length', 0)
        elem_after = max(after.get('element_count', 1), 1)
        signals['content_entropy'] = text_after / elem_after

        # Surface change rate: normalized delta in interactive surface
        ia = after.get('interactive_count', 0)
        ib = max(before.get('interactive_count', 1), 1)
        signals['surface_change_rate'] = abs(ia - ib) / ib

        # Interaction density: interactive fraction of total elements
        signals['interaction_density'] = ia / elem_after

        # Response latency: network reality (present if timing was recorded)
        if 'response_latency_ms' in full_context:
            signals['response_latency'] = full_context['response_latency_ms']

        return signals

    def get_recent(self, n: int) -> List[Dict]:
        records = list(self.records)
        return records[-n:] if len(records) >= n else records

    def get_residuals_array(self, dim: str, n: int = None) -> np.ndarray:
        records = list(self.records) if n is None else list(self.records)[-n:]
        return np.array([r['residual'].get(dim, 0.0) for r in records])

    def get_signal_array(self, signal: str, n: int = None) -> np.ndarray:
        records = list(self.records) if n is None else list(self.records)[-n:]
        return np.array([r['context'].get(signal, 0.0) for r in records])

    def __len__(self):
        return len(self.records)


# ============================================================
# NEW: ResidualExplainer
# ============================================================
class ResidualExplainer:
    """
    Attempts to explain structured residual WITHOUT adding new dimensions.
    New axis is LAST resort, not first response.

    Escalation order (strict — each level must fail before escalating):
    1. Coupling refinement — can better coupling resolution explain it?
    2. Nonlinear term — does a quadratic interaction explain it?
    3. Lag memory — does the previous step's delta explain it?
    4. Candidate axis — only if ALL above fail, returns candidate for AxisAdmissionTest

    Structured residual does NOT mean new dimension. It usually means:
    - Coupling matrix underfit (try level 1 first)
    - Nonlinearity within existing axes (try level 2)
    - Temporal lag not modeled (try level 3)
    - Only if none of these: candidate new axis (level 4)
    """

    REFINEMENT_THRESHOLD = 0.08
    NONLINEAR_THRESHOLD = 0.08
    LAG_THRESHOLD = 0.08
    MIN_RECORDS_FOR_ANALYSIS = 50

    def explain(self, tracker: ResidualTracker,
                coupling_estimator: CouplingMatrixEstimator) -> Dict:
        """
        Run escalation ladder. Returns recommended action.
        Only returns 'candidate_axis' if all lower levels fail.
        """
        if len(tracker) < self.MIN_RECORDS_FOR_ANALYSIS:
            return {'action': 'insufficient_data', 'records': len(tracker)}

        gain = self._try_coupling_refinement(tracker, coupling_estimator)
        if gain > self.REFINEMENT_THRESHOLD:
            return {'action': 'refine_coupling', 'gain': gain}

        gain = self._try_nonlinear_term(tracker)
        if gain > self.NONLINEAR_THRESHOLD:
            return {'action': 'add_nonlinear_term', 'gain': gain}

        gain = self._try_lag_memory(tracker)
        if gain > self.LAG_THRESHOLD:
            return {'action': 'add_lag', 'gain': gain}

        candidate = self._find_candidate_signal(tracker)
        if candidate:
            return {'action': 'candidate_axis', 'candidate': candidate,
                    'requires_admission': True}

        return {'action': 'no_structure_found'}

    def _r_squared(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    def _try_coupling_refinement(self, tracker, coupling_estimator) -> float:
        """Can a locally-fitted coupling matrix explain more residual variance?"""
        records = tracker.get_recent(100)
        if len(records) < 20:
            return 0.0
        gains = []
        for target_dim in SUBSTRATE_DIMS:
            target_idx = SUBSTRATE_DIMS.index(target_dim)
            y = np.array([r['residual'].get(target_dim, 0.0) for r in records])
            if np.std(y) < 1e-8:
                continue
            current_predictions = []
            for r in records:
                obs = r['observed']
                pred = sum(
                    coupling_estimator.matrix[target_idx][j] * obs.get(SUBSTRATE_DIMS[j], 0.0)
                    for j in range(4) if j != target_idx
                )
                current_predictions.append(pred)
            r2_current = self._r_squared(y, np.array(current_predictions))
            X = np.array([[r['observed'].get(d, 0.0) for d in SUBSTRATE_DIMS
                           if d != target_dim] for r in records])
            if X.shape[0] < X.shape[1] + 1:
                continue
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                r2_local = self._r_squared(y, X @ coeffs)
                gains.append(max(0.0, r2_local - r2_current))
            except Exception:
                continue
        return np.mean(gains) if gains else 0.0

    def _try_nonlinear_term(self, tracker) -> float:
        """Does adding quadratic interaction terms explain residual variance?"""
        records = tracker.get_recent(100)
        if len(records) < 20:
            return 0.0
        gains = []
        for target_dim in SUBSTRATE_DIMS:
            y = np.array([r['residual'].get(target_dim, 0.0) for r in records])
            if np.std(y) < 1e-8:
                continue
            X_linear = np.array([[r['observed'].get(d, 0.0) for d in SUBSTRATE_DIMS
                                   if d != target_dim] for r in records])
            obs_vals = [[r['observed'].get(d, 0.0) for d in SUBSTRATE_DIMS] for r in records]
            X_quad = np.array([[row[i] * row[j] for i in range(4) for j in range(i, 4)]
                                for row in obs_vals])
            X_full = np.hstack([X_linear, X_quad])
            if X_full.shape[0] < X_full.shape[1] + 1:
                continue
            try:
                c_lin, _, _, _ = np.linalg.lstsq(X_linear, y, rcond=None)
                r2_lin = self._r_squared(y, X_linear @ c_lin)
                c_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
                r2_full = self._r_squared(y, X_full @ c_full)
                gains.append(max(0.0, r2_full - r2_lin))
            except Exception:
                continue
        return np.mean(gains) if gains else 0.0

    def _try_lag_memory(self, tracker) -> float:
        """Does the previous step's delta predict the current step's residual?"""
        records = tracker.get_recent(100)
        if len(records) < 20:
            return 0.0
        gains = []
        for target_dim in SUBSTRATE_DIMS:
            y = np.array([r['residual'].get(target_dim, 0.0) for r in records[1:]])
            if np.std(y) < 1e-8:
                continue
            X_lag = np.array([[r['observed'].get(d, 0.0) for d in SUBSTRATE_DIMS]
                               for r in records[:-1]])
            if X_lag.shape[0] < X_lag.shape[1] + 1:
                continue
            try:
                c, _, _, _ = np.linalg.lstsq(X_lag, y, rcond=None)
                r2 = self._r_squared(y, X_lag @ c)
                gains.append(max(0.0, r2))
            except Exception:
                continue
        return np.mean(gains) if gains else 0.0

    def _find_candidate_signal(self, tracker) -> Optional[Dict]:
        """
        Find the POTENTIALLY_INVARIANT context signal with highest correlation
        to residual. Returns None if nothing exceeds conservative threshold (0.3).
        INTERFACE_COUPLED signals are never present in tracker records
        (filtered at extraction), so they cannot appear here.
        """
        records = tracker.get_recent(len(tracker))
        if len(records) < 50:
            return None
        best_candidate = None
        best_correlation = 0.3  # Conservative minimum threshold
        for signal in POTENTIALLY_INVARIANT_SIGNALS:
            signal_vals = np.array([r['context'].get(signal, 0.0) for r in records])
            if np.std(signal_vals) < 1e-8:
                continue
            for target_dim in SUBSTRATE_DIMS:
                residuals = np.array([r['residual'].get(target_dim, 0.0) for r in records])
                if np.std(residuals) < 1e-8:
                    continue
                corr = np.corrcoef(signal_vals, residuals)[0, 1]
                if not np.isfinite(corr):
                    continue
                if abs(corr) > best_correlation:
                    best_correlation = abs(corr)
                    best_candidate = {
                        'signal': signal,
                        'affects': target_dim,
                        'correlation': corr,
                        'evidence': len(records)
                    }
        return best_candidate


# ============================================================
# NEW: AxisAdmissionTest
# ============================================================
class AxisAdmissionTest:
    """
    Four-condition gate for new substrate dimension admission.
    ALL four conditions required. No exceptions.

    A. Out-of-sample predictive gain > 0.15 (both window directions, take minimum)
    B. Compression gain: bits saved > bits added (ratio > 1.0, AIC-style)
    C. Shuffle invariance: correlation must NOT survive temporal destruction (< 0.3)
    D. Phi survival relevance: Phi stability must measurably improve (> 0.05)

    Interface invariance filter applied first (before any condition):
    - INTERFACE_COUPLED signals: blocked regardless of statistical performance
    - Unknown signals: blocked pending cross-generational stability

    The compression law is binding:
    Every dimension must justify itself by reducing predictive residual per parameter.
    Decorative modeling (improves prediction but not Phi) does not survive.
    """

    PREDICTIVE_GAIN_THRESHOLD = 0.15
    COMPRESSION_RATIO_THRESHOLD = 1.0
    SHUFFLE_CORRELATION_MAX = 0.3
    PHI_IMPROVEMENT_THRESHOLD = 0.05
    MIN_EVIDENCE = 75

    def evaluate(self, candidate: Dict, tracker: ResidualTracker,
                 phi_history: List[float]) -> Tuple[bool, Dict]:
        """Run full admission test. Returns (admitted, detailed_results)."""
        results = {}

        # Pre-check: interface invariance filter
        signal = candidate.get('signal', '')
        if signal in INTERFACE_COUPLED_SIGNALS:
            results['blocked'] = 'interface_coupled_signal'
            return False, results
        if signal not in POTENTIALLY_INVARIANT_SIGNALS:
            results['blocked'] = 'unknown_signal_requires_cross_generational_stability'
            return False, results
        if candidate.get('evidence', 0) < self.MIN_EVIDENCE:
            results['blocked'] = f"insufficient_evidence_{candidate.get('evidence', 0)}"
            return False, results

        records = list(tracker.records)
        if len(records) < 60:
            results['blocked'] = 'insufficient_tracker_history'
            return False, results

        target_dim = candidate['affects']
        mid = len(records) // 2
        w1, w2 = records[:mid], records[mid:]

        # A: Out-of-sample predictive gain (both directions, conservative minimum)
        gain_1v2 = self._out_of_sample_gain(candidate, w1, w2, target_dim)
        gain_2v1 = self._out_of_sample_gain(candidate, w2, w1, target_dim)
        results['predictive_gain_1v2'] = gain_1v2
        results['predictive_gain_2v1'] = gain_2v1
        results['predictive_gain_conservative'] = min(gain_1v2, gain_2v1)
        results['passes_predictive'] = results['predictive_gain_conservative'] > self.PREDICTIVE_GAIN_THRESHOLD

        # B: Compression gain (AIC-style)
        bits_added = self._parameter_cost()
        bits_saved = self._residual_entropy_reduction(candidate, records, target_dim)
        results['bits_added'] = bits_added
        results['bits_saved'] = bits_saved
        results['compression_ratio'] = bits_saved / max(bits_added, 1e-6)
        results['passes_compression'] = results['compression_ratio'] > self.COMPRESSION_RATIO_THRESHOLD

        # C: Shuffle invariance (destroy temporal structure, recheck correlation)
        signal_vals = np.array([r['context'].get(signal, 0.0) for r in records])
        residuals = np.array([r['residual'].get(target_dim, 0.0) for r in records])
        shuffled = signal_vals.copy()
        np.random.shuffle(shuffled)
        shuffle_corr = abs(np.corrcoef(shuffled, residuals)[0, 1]) \
            if np.std(shuffled) > 1e-8 and np.std(residuals) > 1e-8 else 0.0
        results['shuffle_correlation'] = shuffle_corr
        results['passes_shuffle'] = shuffle_corr < self.SHUFFLE_CORRELATION_MAX

        # D: Phi survival relevance
        phi_improvement = self._phi_stability_gain(phi_history)
        results['phi_improvement'] = phi_improvement
        results['passes_survival'] = phi_improvement > self.PHI_IMPROVEMENT_THRESHOLD

        # All four must pass
        admitted = all([
            results['passes_predictive'],
            results['passes_compression'],
            results['passes_shuffle'],
            results['passes_survival']
        ])
        results['admitted'] = admitted
        return admitted, results

    def _out_of_sample_gain(self, candidate, train, test, target_dim) -> float:
        signal = candidate['signal']
        if len(train) < 10 or len(test) < 10:
            return 0.0
        train_signal = np.array([r['context'].get(signal, 0.0) for r in train])
        train_residual = np.array([r['residual'].get(target_dim, 0.0) for r in train])
        if np.std(train_signal) < 1e-8:
            return 0.0
        try:
            coeffs = np.polyfit(train_signal, train_residual, 1)
        except Exception:
            return 0.0
        test_signal = np.array([r['context'].get(signal, 0.0) for r in test])
        test_residual = np.array([r['residual'].get(target_dim, 0.0) for r in test])
        mse_baseline = np.mean(test_residual ** 2)
        if mse_baseline < 1e-10:
            return 0.0
        mse_model = np.mean((test_residual - np.polyval(coeffs, test_signal)) ** 2)
        return max(0.0, (mse_baseline - mse_model) / mse_baseline)

    def _parameter_cost(self) -> float:
        """Cost of one new parameter in bits (AIC-style)."""
        return 2.0

    def _residual_entropy_reduction(self, candidate, records, target_dim) -> float:
        signal = candidate['signal']
        signal_vals = np.array([r['context'].get(signal, 0.0) for r in records])
        residuals = np.array([r['residual'].get(target_dim, 0.0) for r in records])
        if np.std(signal_vals) < 1e-8 or np.std(residuals) < 1e-8:
            return 0.0
        try:
            coeffs = np.polyfit(signal_vals, residuals, 1)
            residual_after = residuals - np.polyval(coeffs, signal_vals)
            var_before, var_after = np.var(residuals), np.var(residual_after)
            if var_before < 1e-10:
                return 0.0
            variance_reduction = (var_before - var_after) / var_before
            return max(0.0, -0.5 * np.log2(max(1 - variance_reduction, 1e-6)))
        except Exception:
            return 0.0

    def _phi_stability_gain(self, phi_history) -> float:
        """Has Phi variance decreased in the second half of history vs the first?"""
        if len(phi_history) < 20:
            return 0.0
        mid = len(phi_history) // 2
        var_early = np.var(phi_history[:mid])
        var_recent = np.var(phi_history[mid:])
        if var_early < 1e-10:
            return 0.0
        return max(0.0, (var_early - var_recent) / var_early)


# ============================================================
# MODULE 0.5: ATTRACTOR MONITORING
# ============================================================


# ============================================================
# FAILURE CLASSIFICATION
# ============================================================

def classify_relation_failure(trajectories: List, committed_success: bool,
                              violations: List, phi_delta: float) -> Dict:
    """Extract semantic failure pattern from Relation enumeration."""
    if committed_success:
        return None

    failure_type = "unknown"
    severity = 1.0

    if not trajectories or len(trajectories) == 0:
        failure_type = "enumeration_failed"
        severity = 2.0

    elif violations and len(violations) > 0:
        violation_types = [v[0] for v in violations]

        if 'C2_optionality' in violation_types:
            failure_type = "optionality_collapse"
            severity = 1.5
        elif 'C4_reality_contact' in violation_types:
            failure_type = "state_instability"
            severity = 1.3
        elif 'C7_global_coherence' in violation_types:
            failure_type = "coherence_drift"
            severity = 1.4
        else:
            failure_type = "closure_violation"
            severity = 1.2

    elif phi_delta < -0.2:
        failure_type = "coherence_drift"
        severity = 1.6

    else:
        failure_type = "serialization_failed"
        severity = 1.0

    return {
        'type': failure_type,
        'severity': severity,
        'phi_delta': phi_delta,
        'violation_count': len(violations) if violations else 0,
        'timestamp': time.time()
    }




class FailureAssimilationOperator:
    """
    Translates semantic Relation failures into geometric mutation bias.

    Learning that crosses CNS/Relation boundary WITHOUT breaking conservation.
    v13.6: Key to adaptive acceleration across existential boundaries.
    v14: Gains distill_to_genome method — compresses session learning into heritable genome.
    """

    def __init__(self, memory_decay: float = 0.95, inheritance_noise: float = 0.1):
        self.failure_history: deque = deque(maxlen=50)
        self.memory_decay = memory_decay
        self.inheritance_noise = inheritance_noise

        self.mutation_bias = {
            'genome_sigma': {
                'S_bias': 0.1,
                'I_bias': 0.1,
                'P_bias': 0.1,
                'A_bias': 0.1,
                'rigidity_init': 0.1,
                'phi_coherence_weight': 0.1
            },
            'coupling_weights': {
                'smo_rigidity': 1.0,
                'phi_alpha': 1.0,
                'phi_beta': 1.0,
                'phi_gamma': 1.0,
                'phi_A0': 1.0
            },
            'perturbation_emphasis': {
                'S': 1.0,
                'I': 1.0,
                'P': 1.0,
                'A': 1.0
            }
        }

        self.min_sigma = 0.01
        self.max_sigma = 0.5
        self.min_weight = 0.5
        self.max_weight = 3.0

        self.total_failures_assimilated = 0
        self.bias_update_count = 0

    def assimilate_relation_failure(self, failure_record: Dict):
        """Extract geometric pressures from semantic failure."""
        self.failure_history.append(failure_record)
        self.total_failures_assimilated += 1

        failure_type = failure_record.get('type', '')
        severity = failure_record.get('severity', 1.0)

        if failure_type == 'state_instability':
            self.mutation_bias['coupling_weights']['phi_beta'] *= (1.0 + 0.1 * severity)
            self.mutation_bias['perturbation_emphasis']['I'] *= (1.0 + 0.2 * severity)
            self.mutation_bias['genome_sigma']['I_bias'] *= (1.0 + 0.15 * severity)
            self.bias_update_count += 1

        elif failure_type == 'optionality_collapse':
            self.mutation_bias['genome_sigma']['P_bias'] *= (1.0 + 0.15 * severity)
            self.mutation_bias['perturbation_emphasis']['P'] *= (1.0 + 0.3 * severity)
            self.bias_update_count += 1

        elif failure_type == 'coherence_drift':
            self.mutation_bias['genome_sigma']['A_bias'] *= (1.0 + 0.1 * severity)
            self.mutation_bias['coupling_weights']['phi_A0'] *= (1.0 + 0.2 * severity)
            self.mutation_bias['perturbation_emphasis']['A'] *= (1.0 + 0.15 * severity)
            self.bias_update_count += 1

        elif failure_type == 'closure_violation':
            for coupling in self.mutation_bias['coupling_weights']:
                current = self.mutation_bias['coupling_weights'][coupling]
                self.mutation_bias['coupling_weights'][coupling] = \
                    current * 0.9 + 1.0 * 0.1
            self.bias_update_count += 1

        elif failure_type == 'serialization_failed':
            self.mutation_bias['genome_sigma']['rigidity_init'] *= (1.0 + 0.2 * severity)
            self.bias_update_count += 1

        elif failure_type == 'boundary_exhaustion':
            for param in self.mutation_bias['genome_sigma']:
                self.mutation_bias['genome_sigma'][param] *= (1.0 + 0.05 * severity)
            self.bias_update_count += 1

        self._apply_decay()
        self._enforce_bounds()

    def _apply_decay(self):
        for param in self.mutation_bias['genome_sigma']:
            current = self.mutation_bias['genome_sigma'][param]
            baseline = 0.1
            self.mutation_bias['genome_sigma'][param] = \
                baseline + (current - baseline) * self.memory_decay

        for coupling in self.mutation_bias['coupling_weights']:
            current = self.mutation_bias['coupling_weights'][coupling]
            baseline = 1.0
            self.mutation_bias['coupling_weights'][coupling] = \
                baseline + (current - baseline) * self.memory_decay

        for dim in self.mutation_bias['perturbation_emphasis']:
            current = self.mutation_bias['perturbation_emphasis'][dim]
            baseline = 1.0
            self.mutation_bias['perturbation_emphasis'][dim] = \
                baseline + (current - baseline) * self.memory_decay

    def _enforce_bounds(self):
        for param in self.mutation_bias['genome_sigma']:
            self.mutation_bias['genome_sigma'][param] = np.clip(
                self.mutation_bias['genome_sigma'][param],
                self.min_sigma,
                self.max_sigma
            )

        for coupling in self.mutation_bias['coupling_weights']:
            self.mutation_bias['coupling_weights'][coupling] = np.clip(
                self.mutation_bias['coupling_weights'][coupling],
                self.min_weight,
                self.max_weight
            )

        for dim in self.mutation_bias['perturbation_emphasis']:
            self.mutation_bias['perturbation_emphasis'][dim] = np.clip(
                self.mutation_bias['perturbation_emphasis'][dim],
                self.min_weight,
                self.max_weight
            )

    def get_informed_genome(self, parent_genome: 'TriadGenome') -> 'TriadGenome':
        """Mutate genome with learned bias. Layers 2/3 pass through intact."""
        child = copy.deepcopy(parent_genome)
        child.generation += 1

        for field_name in ['S_bias', 'I_bias', 'P_bias', 'A_bias',
                      'rigidity_init', 'phi_coherence_weight']:
            current = getattr(parent_genome, field_name)
            sigma = self.mutation_bias['genome_sigma'][field_name]
            noise = np.random.normal(0, sigma)
            setattr(child, field_name, np.clip(current + noise, 0, 1))

        return child

    def get_biased_coupling_mutation(self, coupling_name: str, current_value: float) -> float:
        """Apply learned bias to coupling mutation."""
        weight = self.mutation_bias['coupling_weights'].get(coupling_name, 1.0)
        base_sigma = 0.01
        sigma = base_sigma * weight
        noise = np.random.normal(0, sigma)
        mutated = current_value + noise
        return np.clip(
            mutated,
            max(0, current_value - 0.05),
            min(1, current_value + 0.05)
        )

    def get_perturbation_weights(self) -> Dict[str, float]:
        """Return learned perturbation emphasis for optionality sampling."""
        return self.mutation_bias['perturbation_emphasis'].copy()

    def serialize_for_child(self) -> Dict:
        """Child inherits parent's learned bias WITH NOISE."""
        noisy_bias = copy.deepcopy(self.mutation_bias)

        for param in noisy_bias['genome_sigma']:
            noise_factor = np.random.normal(1.0, self.inheritance_noise)
            noisy_bias['genome_sigma'][param] *= noise_factor
            noisy_bias['genome_sigma'][param] = np.clip(
                noisy_bias['genome_sigma'][param],
                self.min_sigma,
                self.max_sigma
            )

        for coupling in noisy_bias['coupling_weights']:
            noise_factor = np.random.normal(1.0, self.inheritance_noise)
            noisy_bias['coupling_weights'][coupling] *= noise_factor
            noisy_bias['coupling_weights'][coupling] = np.clip(
                noisy_bias['coupling_weights'][coupling],
                self.min_weight,
                self.max_weight
            )

        return {
            'mutation_bias': noisy_bias,
            'failure_count': len(self.failure_history),
            'total_assimilated': self.total_failures_assimilated,
            'bias_updates': self.bias_update_count,
            'memory_decay': self.memory_decay,
            'inheritance_noise': self.inheritance_noise
        }

    @classmethod
    def from_serialized(cls, serialized: Dict) -> 'FailureAssimilationOperator':
        """Reconstruct FAO from serialized learned bias."""
        fao = cls(
            memory_decay=serialized.get('memory_decay', 0.95),
            inheritance_noise=serialized.get('inheritance_noise', 0.1)
        )
        fao.mutation_bias = serialized['mutation_bias']
        fao.total_failures_assimilated = serialized.get('total_assimilated', 0)
        fao.bias_update_count = serialized.get('bias_updates', 0)
        return fao

    def should_reset_bias(self, phi_current: float, phi_history: List[float]) -> bool:
        """Stochastic bias reset if Φ declining despite learning."""
        if len(phi_history) < 10:
            return False
        recent_phi = phi_history[-10:]
        slope = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]
        if slope < -0.1 and phi_current < 0.3:
            return np.random.random() < 0.2
        return False

    def reset_to_baseline(self):
        """Reset mutation bias to isotropic baseline."""
        self.mutation_bias = {
            'genome_sigma': {k: 0.1 for k in self.mutation_bias['genome_sigma']},
            'coupling_weights': {k: 1.0 for k in self.mutation_bias['coupling_weights']},
            'perturbation_emphasis': {k: 1.0 for k in self.mutation_bias['perturbation_emphasis']}
        }
        self.failure_history.clear()

    def distill_to_genome(self,
                           coupling_estimator: CouplingMatrixEstimator,
                           cam: ControlAsymmetryMeasure,
                           residual_tracker: ResidualTracker,
                           residual_explainer: ResidualExplainer,
                           axis_admission: AxisAdmissionTest,
                           phi_history: List[float],
                           parent_genome: TriadGenome,
                           session_length: int = 100) -> TriadGenome:
        """
        SESSION → GENOME BRIDGE. v14.1: uses ProvisionalAxisManager for Layer 3 decay.

        Layer 1: Mutated via get_informed_genome (existing FAO mechanism, unchanged).
        Layer 2: Coupling matrix merged if >= 50 session observations.
                 Action map written for any affordance with >= 10 observations.
                 Merged with parent values, session weighted by evidence.
        Layer 3: ProvisionalAxisManager two-tier lifecycle:
                 - Admitted: 0.8x decay, floor 20 (unchanged from v14)
                 - Provisional: 0.5x decay (0.6x if session < 30 steps), floor 5
                 - New: failed-gate but escalation-passed axes → provisional (not discarded)
                 - Promotion: provisional → admitted via AxisAdmissionTest on real residuals
        Layer 4: lineage_history passes through intact (managed by extract_genome_v14_1.py).
        Returns child TriadGenome.
        """
        child = self.get_informed_genome(parent_genome)

        # === LAYER 2: LEARNED CAUSAL MODEL ===
        causal = dict(parent_genome.causal_model)

        # Coupling matrix: merge if sufficient session evidence
        if coupling_estimator.observation_count >= 50:
            parent_coupling_entry = causal.get('coupling_matrix', {
                'matrix': np.eye(4).tolist(),
                'observations': 0,
                'confidence': 0.0
            })
            merged = CouplingMatrixEstimator.merge(parent_coupling_entry, coupling_estimator)
            causal['coupling_matrix'] = merged.to_genome_entry()

        # Action substrate map: merge with parent, weighted by evidence
        empirical_map = cam.get_empirical_action_map()
        if empirical_map:
            parent_map = causal.get('action_substrate_map', {})
            merged_map = dict(parent_map)
            for affordance, new_delta in empirical_map.items():
                if affordance in merged_map:
                    obs_count = len(cam.affordance_deltas.get(affordance, []))
                    session_weight = min(0.5, obs_count / 100.0)
                    for dim in SUBSTRATE_DIMS:
                        merged_map[affordance][dim] = (
                            (1 - session_weight) * merged_map[affordance].get(dim, 0.0) +
                            session_weight * new_delta.get(dim, 0.0)
                        )
                else:
                    merged_map[affordance] = new_delta
            causal['action_substrate_map'] = merged_map

        # === LAYER 3: DISCOVERED STRUCTURE (v14.1: two-tier via ProvisionalAxisManager) ===
        discovered = dict(parent_genome.discovered_structure)
        pam = ProvisionalAxisManager()

        # Attempt promotion of existing provisional axes before running escalation
        for key, entry in list(discovered.items()):
            if entry.get('status') == 'provisional':
                candidate_for_promo = {
                    'signal': entry.get('signal'),
                    'affects': entry.get('affects'),
                    'correlation': entry.get('correlation', 0.0),
                    'evidence': entry.get('total_evidence', 0.0),
                }
                discovered[key] = pam.try_promote(
                    key, entry, candidate_for_promo, residual_tracker, phi_history, axis_admission
                )

        # Run escalation ladder — reaches candidate_axis only if levels 1-3 fail
        explanation = residual_explainer.explain(residual_tracker, coupling_estimator)

        if explanation['action'] == 'candidate_axis' and explanation.get('requires_admission'):
            candidate = explanation.get('candidate')
            if candidate:
                admitted, admission_results = axis_admission.evaluate(
                    candidate, residual_tracker, phi_history
                )
                dim_key = f"{candidate['signal']}_affects_{candidate['affects']}"
                if admitted:
                    if dim_key in discovered:
                        discovered[dim_key]['total_evidence'] += candidate['evidence']
                        discovered[dim_key]['correlation'] = (
                            0.7 * discovered[dim_key]['correlation'] +
                            0.3 * candidate['correlation']
                        )
                        discovered[dim_key]['generations_seen'] = \
                            discovered[dim_key].get('generations_seen', 0) + 1
                        # Promote if currently provisional
                        if discovered[dim_key].get('status') == 'provisional':
                            discovered[dim_key]['status'] = 'admitted'
                            discovered[dim_key]['admission_results'] = admission_results
                    else:
                        # New axis: admitted directly (passed full gate)
                        discovered[dim_key] = {
                            **candidate,
                            'total_evidence': candidate['evidence'],
                            'generations_seen': 1,
                            'admission_results': admission_results,
                            'status': 'admitted',
                        }
                else:
                    # Failed AxisAdmissionTest but passed escalation ladder
                    # → admit as provisional for priority observation next session
                    if dim_key not in discovered:
                        discovered[dim_key] = {
                            **candidate,
                            'total_evidence': candidate['evidence'],
                            'generations_seen': 1,
                            'admission_results': admission_results,
                            'status': 'provisional',
                        }

        # v14.1: Two-tier decay via ProvisionalAxisManager
        discovered = pam.decay_and_prune(discovered, session_length)

        # Layer 4: lineage_history passes through unchanged
        # (extract_genome_v14_1.py appends the new generation snapshot)
        child.causal_model = causal
        child.discovered_structure = discovered
        child.lineage_history = list(parent_genome.lineage_history)
        return child


# ============================================================
# MODULE 5.5: CNS GEOMETRIC MITOSIS
# ============================================================
