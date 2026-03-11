from __future__ import annotations

"""
UII v16 — uii_fao.py
Failure Assimilation Operator & Residual Learning

Role: FAO is the legal mechanism by which session learning informs the Ledger's
causal model. It operates on ledger.causal_model and ledger.discovered_structure.
Input: observable signals only (Φ delta, CRK violation types, trajectory outcomes).
Output: updated TriadLedger. Never action selection.

v16 changes vs v15.3:
  - Imports: uii_geometry (replaces uii_types); uii_ledger.TriadLedger (replaces
    uii_genome.TriadGenome). ProvisionalAxisManager moves here from uii_genome.py.
  - distill_to_ledger() replaces distill_to_genome():
      · No genome parameter — TriadLedger passed in directly.
      · Layer 1 (scalar velocity targets) eliminated entirely.
      · Layer 1b (operator geometry scalars) eliminated — live in operator_snapshot,
        written by PeakOptionalityTracker.update() during the run, not here.
      · CAM parameter eliminated — reads coupling_estimator.affordance_deltas directly.
      · Does NOT touch ledger.hessian_snapshot or ledger.operator_snapshot.
        Those are written exclusively by PeakOptionalityTracker.update() during step().
      · Layer 2, 3 logic unchanged from v15.3.
  - get_informed_ledger() replaces get_informed_genome():
      · No Layer 1 mutation. Returns ledger copy with updated causal_model and
        discovered_structure only.
  - FailureAssimilationOperator.mutation_bias: genome_sigma key removed (Layer 1 dead).
    coupling_weights and perturbation_emphasis retained.

Also contains (unchanged from v15.3):
  - ResidualTracker
  - ResidualExplainer
  - AxisAdmissionTest
  - ProvisionalAxisManager (moved here from uii_genome.py — its only user)
  - classify_relation_failure
  - FailureAssimilationOperator
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import copy
import time
from collections import deque

from uii_geometry import (
    SUBSTRATE_DIMS,
    INTERFACE_COUPLED_SIGNALS,
    POTENTIALLY_INVARIANT_SIGNALS,
    SubstrateState,
    TrajectoryCandidate,
)
from uii_ledger import TriadLedger
from uii_reality import CouplingMatrixEstimator


# ──────────────────────────────────────────────────────────────────────────────
# ProvisionalAxisManager — moved from uii_genome.py (its only user)
# Logic unchanged from v15.3.
# ──────────────────────────────────────────────────────────────────────────────

class ProvisionalAxisManager:
    """
    Two-tier Layer 3 management: provisional and admitted.

    Provisional axes:
      - Evidence floor: 5 (vs 20 for admitted)
      - Decay: 0.5x per session (vs 0.8x for admitted)
      - Short-session variant: 0.6x if session_length < 30 steps
      - Watched by ResidualTracker from session start (priority observation)

    Promotion: provisional → admitted when AxisAdmissionTest passes on real residuals.
    Removal: automatic via faster decay.

    Framing:
      Cheap to test (provisional costs almost nothing)
      Cheap to remove (0.5x decay clears in ~3 sessions)
      Expensive to keep (full AxisAdmissionTest required for promotion)
      Continuously re-validated (decay never stops, even for admitted)
    """

    PROVISIONAL_EVIDENCE_FLOOR:    float = 5.0
    ADMITTED_EVIDENCE_FLOOR:       float = 20.0
    PROVISIONAL_DECAY:             float = 0.5
    PROVISIONAL_DECAY_SHORT_SESSION: float = 0.6
    ADMITTED_DECAY:                float = 0.8
    SHORT_SESSION_THRESHOLD:       int   = 30

    def decay_and_prune(self,
                        discovered_structure: Dict,
                        session_length:       int) -> Dict:
        """
        Apply per-tier decay to all Layer 3 entries. Remove below floor.
        Called by distill_to_ledger() before returning updated ledger.
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
                    key:              str,
                    entry:            Dict,
                    candidate:        Dict,
                    residual_tracker,
                    phi_history:      List[float],
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


# ──────────────────────────────────────────────────────────────────────────────
# ResidualTracker — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

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
        after  = reality_context.get('after',  {})
        context_signals = self._extract_invariant_signals(before, after, reality_context)
        self.records.append({
            'residual':   residual,
            'predicted':  predicted,
            'observed':   observed,
            'context':    context_signals,
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

        # Response latency: network reality
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


# ──────────────────────────────────────────────────────────────────────────────
# ResidualExplainer — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

class ResidualExplainer:
    """
    Attempts to explain structured residual WITHOUT adding new dimensions.
    New axis is LAST resort, not first response.

    Escalation order (strict — each level must fail before escalating):
    1. Coupling refinement — can better coupling resolution explain it?
    2. Nonlinear term — does a quadratic interaction explain it?
    3. Lag memory — does the previous step's delta explain it?
    4. Candidate axis — only if ALL above fail, returns candidate for AxisAdmissionTest
    """

    REFINEMENT_THRESHOLD       = 0.08
    NONLINEAR_THRESHOLD        = 0.08
    LAG_THRESHOLD              = 0.08
    MIN_RECORDS_FOR_ANALYSIS   = 50

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

    def _try_coupling_refinement(self, tracker: ResidualTracker,
                                  coupling_estimator: CouplingMatrixEstimator) -> float:
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
                obs  = r['observed']
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
        return float(np.mean(gains)) if gains else 0.0

    def _try_nonlinear_term(self, tracker: ResidualTracker) -> float:
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
            obs_vals = [[r['observed'].get(d, 0.0) for d in SUBSTRATE_DIMS]
                        for r in records]
            X_quad = np.array([[row[i] * row[j] for i in range(4) for j in range(i, 4)]
                                for row in obs_vals])
            X_full = np.hstack([X_linear, X_quad])
            if X_full.shape[0] < X_full.shape[1] + 1:
                continue
            try:
                c_lin,  _, _, _ = np.linalg.lstsq(X_linear, y, rcond=None)
                r2_lin          = self._r_squared(y, X_linear @ c_lin)
                c_full, _, _, _ = np.linalg.lstsq(X_full,   y, rcond=None)
                r2_full         = self._r_squared(y, X_full   @ c_full)
                gains.append(max(0.0, r2_full - r2_lin))
            except Exception:
                continue
        return float(np.mean(gains)) if gains else 0.0

    def _try_lag_memory(self, tracker: ResidualTracker) -> float:
        """Does the previous step's delta predict the current step's residual?"""
        records = tracker.get_recent(100)
        if len(records) < 20:
            return 0.0
        gains = []
        for target_dim in SUBSTRATE_DIMS:
            y     = np.array([r['residual'].get(target_dim, 0.0) for r in records[1:]])
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
        return float(np.mean(gains)) if gains else 0.0

    def _find_candidate_signal(self, tracker: ResidualTracker) -> Optional[Dict]:
        """
        Find the POTENTIALLY_INVARIANT context signal with highest correlation
        to residual. Returns None if nothing exceeds conservative threshold (0.3).
        INTERFACE_COUPLED signals are never present in tracker records
        (filtered at extraction), so they cannot appear here.
        """
        records = tracker.get_recent(len(tracker))
        if len(records) < 50:
            return None
        best_candidate   = None
        best_correlation = 0.3  # conservative minimum
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
                    best_candidate   = {
                        'signal':      signal,
                        'affects':     target_dim,
                        'correlation': corr,
                        'evidence':    len(records),
                    }
        return best_candidate


# ──────────────────────────────────────────────────────────────────────────────
# AxisAdmissionTest — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

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
    """

    PREDICTIVE_GAIN_THRESHOLD  = 0.15
    COMPRESSION_RATIO_THRESHOLD = 1.0
    SHUFFLE_CORRELATION_MAX    = 0.3
    PHI_IMPROVEMENT_THRESHOLD  = 0.05
    MIN_EVIDENCE               = 75

    def evaluate(self, candidate: Dict, tracker: ResidualTracker,
                 phi_history: List[float]) -> Tuple[bool, Dict]:
        """Run full admission test. Returns (admitted, detailed_results)."""
        results = {}

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
        mid        = len(records) // 2
        w1, w2     = records[:mid], records[mid:]

        # A: Out-of-sample predictive gain (both directions, conservative minimum)
        gain_1v2 = self._out_of_sample_gain(candidate, w1, w2, target_dim)
        gain_2v1 = self._out_of_sample_gain(candidate, w2, w1, target_dim)
        results['predictive_gain_1v2']         = gain_1v2
        results['predictive_gain_2v1']         = gain_2v1
        results['predictive_gain_conservative'] = min(gain_1v2, gain_2v1)
        results['passes_predictive'] = results['predictive_gain_conservative'] > self.PREDICTIVE_GAIN_THRESHOLD

        # B: Compression gain (AIC-style)
        bits_added = self._parameter_cost()
        bits_saved = self._residual_entropy_reduction(candidate, records, target_dim)
        results['bits_added']          = bits_added
        results['bits_saved']          = bits_saved
        results['compression_ratio']   = bits_saved / max(bits_added, 1e-6)
        results['passes_compression']  = results['compression_ratio'] > self.COMPRESSION_RATIO_THRESHOLD

        # C: Shuffle invariance
        signal_vals = np.array([r['context'].get(signal, 0.0) for r in records])
        residuals   = np.array([r['residual'].get(target_dim, 0.0) for r in records])
        shuffled    = signal_vals.copy()
        np.random.shuffle(shuffled)
        shuffle_corr = (
            abs(np.corrcoef(shuffled, residuals)[0, 1])
            if np.std(shuffled) > 1e-8 and np.std(residuals) > 1e-8
            else 0.0
        )
        results['shuffle_correlation'] = shuffle_corr
        results['passes_shuffle']      = shuffle_corr < self.SHUFFLE_CORRELATION_MAX

        # D: Phi survival relevance
        phi_improvement = self._phi_stability_gain(phi_history)
        results['phi_improvement']  = phi_improvement
        results['passes_survival']  = phi_improvement > self.PHI_IMPROVEMENT_THRESHOLD

        admitted = all([
            results['passes_predictive'],
            results['passes_compression'],
            results['passes_shuffle'],
            results['passes_survival'],
        ])
        results['admitted'] = admitted
        return admitted, results

    def _out_of_sample_gain(self, candidate: Dict, train: List, test: List,
                             target_dim: str) -> float:
        signal = candidate['signal']
        if len(train) < 10 or len(test) < 10:
            return 0.0
        train_signal   = np.array([r['context'].get(signal, 0.0) for r in train])
        train_residual = np.array([r['residual'].get(target_dim, 0.0) for r in train])
        if np.std(train_signal) < 1e-8:
            return 0.0
        try:
            coeffs = np.polyfit(train_signal, train_residual, 1)
        except Exception:
            return 0.0
        test_signal   = np.array([r['context'].get(signal, 0.0) for r in test])
        test_residual = np.array([r['residual'].get(target_dim, 0.0) for r in test])
        mse_baseline  = np.mean(test_residual ** 2)
        if mse_baseline < 1e-10:
            return 0.0
        mse_model = np.mean((test_residual - np.polyval(coeffs, test_signal)) ** 2)
        return max(0.0, (mse_baseline - mse_model) / mse_baseline)

    def _parameter_cost(self) -> float:
        """Cost of one new parameter in bits (AIC-style)."""
        return 2.0

    def _residual_entropy_reduction(self, candidate: Dict, records: List,
                                     target_dim: str) -> float:
        signal      = candidate['signal']
        signal_vals = np.array([r['context'].get(signal, 0.0) for r in records])
        residuals   = np.array([r['residual'].get(target_dim, 0.0) for r in records])
        if np.std(signal_vals) < 1e-8 or np.std(residuals) < 1e-8:
            return 0.0
        try:
            coeffs        = np.polyfit(signal_vals, residuals, 1)
            residual_after = residuals - np.polyval(coeffs, signal_vals)
            var_before, var_after = np.var(residuals), np.var(residual_after)
            if var_before < 1e-10:
                return 0.0
            variance_reduction = (var_before - var_after) / var_before
            return max(0.0, -0.5 * np.log2(max(1 - variance_reduction, 1e-6)))
        except Exception:
            return 0.0

    def _phi_stability_gain(self, phi_history: List[float]) -> float:
        """Has Phi variance decreased in the second half vs first?"""
        if len(phi_history) < 20:
            return 0.0
        mid       = len(phi_history) // 2
        var_early  = np.var(phi_history[:mid])
        var_recent = np.var(phi_history[mid:])
        if var_early < 1e-10:
            return 0.0
        return max(0.0, (var_early - var_recent) / var_early)


# ──────────────────────────────────────────────────────────────────────────────
# classify_relation_failure — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

def classify_relation_failure(trajectories: List,
                               committed_success: bool,
                               violations:        List,
                               phi_delta:         float) -> Optional[Dict]:
    """Extract semantic failure pattern from Relation enumeration."""
    if committed_success:
        return None

    failure_type = "unknown"
    severity     = 1.0

    if not trajectories or len(trajectories) == 0:
        failure_type = "enumeration_failed"
        severity     = 2.0

    elif violations and len(violations) > 0:
        violation_types = [v[0] for v in violations]
        if 'C2_optionality' in violation_types:
            failure_type, severity = "optionality_collapse", 1.5
        elif 'C4_reality_contact' in violation_types:
            failure_type, severity = "state_instability", 1.3
        elif 'C7_global_coherence' in violation_types:
            failure_type, severity = "coherence_drift", 1.4
        else:
            failure_type, severity = "closure_violation", 1.2

    elif phi_delta < -0.2:
        failure_type, severity = "coherence_drift", 1.6

    else:
        failure_type, severity = "serialization_failed", 1.0

    return {
        'type':            failure_type,
        'severity':        severity,
        'phi_delta':       phi_delta,
        'violation_count': len(violations) if violations else 0,
        'timestamp':       time.time(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FailureAssimilationOperator — v16
# ──────────────────────────────────────────────────────────────────────────────

class FailureAssimilationOperator:
    """
    Translates session learning into Ledger geometry.

    v16 changes:
      - genome_sigma removed from mutation_bias (Layer 1 dead).
      - get_informed_ledger() replaces get_informed_genome() —
        no Layer 1 scalar mutation; copies ledger with updated causal_model
        and discovered_structure only.
      - distill_to_ledger() replaces distill_to_genome() — accepts TriadLedger
        directly, no genome parameter, no CAM parameter.
      - assimilate_relation_failure() retained as no-op (backward compat).
      - coupling_weights and perturbation_emphasis unchanged.
    """

    def __init__(self, memory_decay: float = 0.95, inheritance_noise: float = 0.1):
        self.failure_history: deque = deque(maxlen=50)
        self.memory_decay      = memory_decay
        self.inheritance_noise = inheritance_noise

        # genome_sigma removed — Layer 1 scalars dead.
        # coupling_weights and perturbation_emphasis retained.
        self.mutation_bias = {
            'coupling_weights': {
                'smo_rigidity':        1.0,
                'phi_alpha':           1.0,
                'phi_beta':            1.0,
                'phi_gamma':           1.0,
                'phi_A0':              1.0,
            },
            'perturbation_emphasis': {
                'S': 1.0,
                'I': 1.0,
                'P': 1.0,
                'A': 1.0,
            },
        }

        self.min_weight = 0.5
        self.max_weight = 3.0

        self.total_failures_assimilated = 0
        self.bias_update_count          = 0

    def assimilate_relation_failure(self, failure_record: Dict):
        """
        No-op. Retained for backward compatibility with v14 log readers.
        In v16 session outcomes feed CouplingMatrixEstimator directly via update().
        """
        pass

    def _apply_decay(self):
        for coupling in self.mutation_bias['coupling_weights']:
            current  = self.mutation_bias['coupling_weights'][coupling]
            baseline = 1.0
            self.mutation_bias['coupling_weights'][coupling] = (
                baseline + (current - baseline) * self.memory_decay
            )
        for dim in self.mutation_bias['perturbation_emphasis']:
            current  = self.mutation_bias['perturbation_emphasis'][dim]
            baseline = 1.0
            self.mutation_bias['perturbation_emphasis'][dim] = (
                baseline + (current - baseline) * self.memory_decay
            )

    def _enforce_bounds(self):
        for coupling in self.mutation_bias['coupling_weights']:
            self.mutation_bias['coupling_weights'][coupling] = float(np.clip(
                self.mutation_bias['coupling_weights'][coupling],
                self.min_weight, self.max_weight
            ))
        for dim in self.mutation_bias['perturbation_emphasis']:
            self.mutation_bias['perturbation_emphasis'][dim] = float(np.clip(
                self.mutation_bias['perturbation_emphasis'][dim],
                self.min_weight, self.max_weight
            ))

    def get_informed_ledger(self, ledger: TriadLedger) -> TriadLedger:
        """
        Return a copy of ledger. No Layer 1 scalar mutation.
        causal_model and discovered_structure are updated by distill_to_ledger().
        hessian_snapshot and operator_snapshot are NOT touched — written
        exclusively by PeakOptionalityTracker.update() during step().

        Called by distill_to_ledger() internally; exposed for callers that
        previously used get_informed_genome().
        """
        return copy.deepcopy(ledger)

    def get_biased_coupling_mutation(self, coupling_name: str,
                                      current_value: float) -> float:
        """Apply learned bias to coupling mutation. Unchanged from v15.3."""
        weight    = self.mutation_bias['coupling_weights'].get(coupling_name, 1.0)
        base_sigma = 0.01
        sigma     = base_sigma * weight
        noise     = np.random.normal(0, sigma)
        mutated   = current_value + noise
        return float(np.clip(mutated,
                              max(0, current_value - 0.05),
                              min(1, current_value + 0.05)))

    def get_perturbation_weights(self) -> Dict[str, float]:
        """Return learned perturbation emphasis for optionality sampling."""
        return self.mutation_bias['perturbation_emphasis'].copy()

    def serialize_for_child(self) -> Dict:
        """FAO state persistence — child inherits learned bias WITH NOISE."""
        noisy_bias = copy.deepcopy(self.mutation_bias)
        for coupling in noisy_bias['coupling_weights']:
            noise_factor = np.random.normal(1.0, self.inheritance_noise)
            noisy_bias['coupling_weights'][coupling] = float(np.clip(
                noisy_bias['coupling_weights'][coupling] * noise_factor,
                self.min_weight, self.max_weight
            ))
        return {
            'mutation_bias':      noisy_bias,
            'failure_count':      len(self.failure_history),
            'total_assimilated':  self.total_failures_assimilated,
            'bias_updates':       self.bias_update_count,
            'memory_decay':       self.memory_decay,
            'inheritance_noise':  self.inheritance_noise,
        }

    @classmethod
    def from_serialized(cls, serialized: Dict) -> 'FailureAssimilationOperator':
        """Reconstruct FAO from serialized learned bias."""
        fao = cls(
            memory_decay      = serialized.get('memory_decay',       0.95),
            inheritance_noise = serialized.get('inheritance_noise',   0.1),
        )
        fao.mutation_bias               = serialized['mutation_bias']
        fao.total_failures_assimilated  = serialized.get('total_assimilated', 0)
        fao.bias_update_count           = serialized.get('bias_updates',       0)
        return fao

    def should_reset_bias(self, phi_current: float,
                           phi_history: List[float]) -> bool:
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
            'coupling_weights':     {k: 1.0 for k in self.mutation_bias['coupling_weights']},
            'perturbation_emphasis': {k: 1.0 for k in self.mutation_bias['perturbation_emphasis']},
        }
        self.failure_history.clear()

    # ── distill_to_ledger — REPLACES distill_to_genome ───────────────────────

    def distill_to_ledger(self,
                           coupling_estimator:  CouplingMatrixEstimator,
                           residual_tracker:    ResidualTracker,
                           residual_explainer:  ResidualExplainer,
                           axis_admission:      AxisAdmissionTest,
                           phi_history:         List[float],
                           ledger:              TriadLedger,
                           session_length:      int = 100,
                           migration_history:   Optional[List] = None,
                           ) -> TriadLedger:
        """
        SESSION → LEDGER BRIDGE.

        Updates ledger.causal_model and ledger.discovered_structure.
        Does NOT touch ledger.hessian_snapshot or ledger.operator_snapshot —
        those are written exclusively by PeakOptionalityTracker.update()
        during step() at peak Vol_opt. Session end may be degraded.

        Layer 1 (scalar velocity targets): ELIMINATED.
        Layer 1b (operator geometry): NOT written here — already in operator_snapshot.

        Layer 2 (causal model): Unchanged merge logic vs v15.3.
          - Coupling matrix: merge session estimator into ledger.causal_model
            ['coupling_matrix']. Session weighted by evidence.
          - action_substrate_map: merged from coupling_estimator.affordance_deltas
            (replaces CAM). Mean delta per action for actions with >= 5 obs.
          - migration_geometry: unchanged merge logic.

        Layer 3 (discovered structure): Unchanged.
          ProvisionalAxisManager two-tier decay lifecycle. AxisAdmissionTest.

        Layer 4 (lineage_history): ELIMINATED.
        """
        child = self.get_informed_ledger(ledger)

        # === LAYER 2: LEARNED CAUSAL MODEL ===
        causal = dict(ledger.causal_model)

        # Coupling matrix merge
        terminal_entry = coupling_estimator.to_ledger_entry()

        if coupling_estimator.observation_count >= 50:
            parent_coupling_entry = causal.get('coupling_matrix', {
                'matrix':       np.eye(4).tolist(),
                'observations': 0,
                'confidence':   0.0,
            })
            merged = CouplingMatrixEstimator.merge(parent_coupling_entry, coupling_estimator)
            causal['coupling_matrix'] = merged.to_ledger_entry()

        # Action substrate map: read from coupling_estimator.affordance_deltas directly
        # (replaces CAM.get_empirical_action_map())
        empirical_map = coupling_estimator.get_empirical_action_map()
        if empirical_map:
            parent_map  = causal.get('action_substrate_map', {})
            merged_map  = dict(parent_map)
            for affordance, new_delta in empirical_map.items():
                if affordance in merged_map:
                    obs_count      = len(coupling_estimator.affordance_deltas.get(affordance, []))
                    session_weight = min(0.5, obs_count / 100.0)
                    for dim in SUBSTRATE_DIMS:
                        merged_map[affordance][dim] = (
                            (1 - session_weight) * merged_map[affordance].get(dim, 0.0)
                            + session_weight * new_delta.get(dim, 0.0)
                        )
                else:
                    merged_map[affordance] = new_delta
            causal['action_substrate_map'] = merged_map

        # Migration geometry merge (unchanged logic from v15.3)
        if migration_history:
            parent_geo = causal.get('migration_geometry', {
                'bad_hashes':                 [],
                'successful_coupling_states': [],
                'outcome_counts':             {},
                'total_attempts':             0,
            })

            session_bad_hashes:     set              = set()
            session_good_states:    List             = []
            session_outcome_counts: Dict[str, int]   = {}

            for attempt in migration_history:
                outcome   = attempt.get('outcome')   if isinstance(attempt, dict) else attempt.outcome
                code_hash = attempt.get('code_hash') if isinstance(attempt, dict) else attempt.code_hash
                cs        = attempt.get('coupling_state') if isinstance(attempt, dict) else attempt.coupling_state

                session_outcome_counts[outcome] = session_outcome_counts.get(outcome, 0) + 1

                if outcome in ('coherence_loss', 'serialized_only'):
                    if code_hash:
                        session_bad_hashes.add(code_hash)
                elif outcome in ('spawn_attempted', 'handshake_received'):
                    if cs is not None:
                        session_good_states.append(cs)

            merged_bad  = set(parent_geo.get('bad_hashes', [])) | session_bad_hashes
            merged_good = list(parent_geo.get('successful_coupling_states', []))
            for new_cs in session_good_states:
                new_arr = np.array(new_cs)
                is_dup  = any(
                    np.linalg.norm(new_arr - np.array(existing), 'fro') < 0.1
                    for existing in merged_good
                )
                if not is_dup:
                    merged_good.append(new_cs)

            merged_outcomes = dict(parent_geo.get('outcome_counts', {}))
            for k, v in session_outcome_counts.items():
                merged_outcomes[k] = merged_outcomes.get(k, 0) + v

            causal['migration_geometry'] = {
                'bad_hashes':                 list(merged_bad),
                'successful_coupling_states': merged_good,
                'outcome_counts':             merged_outcomes,
                'total_attempts':             parent_geo.get('total_attempts', 0) + len(migration_history),
            }

        # === LAYER 3: DISCOVERED STRUCTURE ===
        discovered = dict(ledger.discovered_structure)
        pam        = ProvisionalAxisManager()

        # Attempt promotion of existing provisional axes first
        for key, entry in list(discovered.items()):
            if entry.get('status') == 'provisional':
                candidate_for_promo = {
                    'signal':      entry.get('signal'),
                    'affects':     entry.get('affects'),
                    'correlation': entry.get('correlation', 0.0),
                    'evidence':    entry.get('total_evidence', 0.0),
                }
                discovered[key] = pam.try_promote(
                    key, entry, candidate_for_promo,
                    residual_tracker, phi_history, axis_admission
                )

        # Run escalation ladder
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
                        discovered[dim_key]['total_evidence']   += candidate['evidence']
                        discovered[dim_key]['correlation']       = (
                            0.7 * discovered[dim_key]['correlation']
                            + 0.3 * candidate['correlation']
                        )
                        discovered[dim_key]['sessions_seen'] = (
                            discovered[dim_key].get('sessions_seen', 0) + 1
                        )
                        if discovered[dim_key].get('status') == 'provisional':
                            discovered[dim_key]['status']            = 'admitted'
                            discovered[dim_key]['admission_results'] = admission_results
                    else:
                        discovered[dim_key] = {
                            **candidate,
                            'total_evidence':   candidate['evidence'],
                            'sessions_seen':    1,
                            'admission_results': admission_results,
                            'status':           'admitted',
                        }
                else:
                    # Failed gate but passed escalation → provisional
                    if dim_key not in discovered:
                        discovered[dim_key] = {
                            **candidate,
                            'total_evidence':   candidate['evidence'],
                            'sessions_seen':    1,
                            'admission_results': admission_results,
                            'status':           'provisional',
                        }

        # Two-tier decay
        discovered = pam.decay_and_prune(discovered, session_length)

        child.causal_model       = causal
        child.discovered_structure = discovered
        # hessian_snapshot and operator_snapshot: NOT touched. Written by
        # PeakOptionalityTracker.update() during step(). Session end is
        # potentially degraded — never overwrite peak basin geometry.
        return child
