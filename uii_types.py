"""
UII v14.1 — uii_types.py
Shared Foundation Types

Role: Provides all data structures, abstract interfaces, and constants shared
across the other UII modules. Everything else imports from here.
No UII module depends on this one — it only uses stdlib + numpy.

Contents:
  - Constants (affordances, signals, dimensions)
  - SMO (Self-Modifying Operator)
  - SubstrateState, StateTrace
  - PhiField (Φ information potential field)
  - CRKMonitor (Constraint Recognition Kernel)
  - TrajectoryCandidate, TrajectoryManifold
  - AgentHandler, UserAgentHandler, AVAILABLE_AGENTS
  - RealityAdapter (ABC), IntelligenceAdapter (ABC)
"""

from dataclasses import dataclass, field
import dataclasses
from typing import Dict, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
import numpy as np
import copy
import time
from collections import deque

# v15.1: operator layer — imported here for SubstrateState and CRK
from uii_operators import (
    SensingOperator, CompressionOperator, PredictionOperator,
    CoherenceOperator, OperatorConsistencyCheck, DEFAULT_CHANNELS,
    SensingChannel, CausalEdge, ChannelPrediction, SMOUpdate,
)

BASE_AFFORDANCES = {
    'navigate', 'click', 'fill', 'type', 'read',
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',
    'python',
    'llm_query',   # v15: symbol grounding affordance — outcome feeds CouplingMatrixEstimator
    'migrate',     # v15: substrate migration — exits current environment, opens new causal surface
}

# Interface-coupled signals — artifacts of HTML/browser rendering, not reality structure.
# BLOCKED from becoming discovered genome axes.
# A signal that vanishes when you change browsers is modeling the interface, not the world.
INTERFACE_COUPLED_SIGNALS = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity'
}

# v14.1: Named constant for Layer 1 parameter list — used by velocity system throughout
# v15.3: Operator geometry replaces scalar DASS proxies in Layer 1.
# Each param is a scalar summary derived from the corresponding operator's geometry —
# earned through reality contact, not designer-initialized.
#
# s_coverage_mean: mean coverage of active SensingOperator channels at peak Vol_opt
# p_horizon_norm:  PredictionOperator.realized_horizon / 50.0 at peak Vol_opt
# a_loop_closure:  CoherenceOperator loop_signature-derived closure at peak Vol_opt
# rigidity_init:   SMO plasticity seed (search param — not a DASS proxy)
# phi_coherence_weight: Φ geometry weight (search param — not a DASS proxy)
#
# I_bias removed: compression geometry is L2 (coupling_matrix) — no L1 scalar needed.
LAYER1_PARAMS = ['s_coverage_mean', 'p_horizon_norm', 'a_loop_closure',
                 'rigidity_init', 'phi_coherence_weight']

# v14.1: Explicit mapping from Layer 1 param name to its velocity field name
VELOCITY_FIELD = {
    's_coverage_mean':      's_coverage_mean_velocity',
    'p_horizon_norm':       'p_horizon_norm_velocity',
    'a_loop_closure':       'a_loop_closure_velocity',
    'rigidity_init':        'rigidity_init_velocity',
    'phi_coherence_weight': 'phi_coherence_velocity',
}

# Potentially invariant signals — may survive substrate change.
# ALLOWED to proceed to the full AxisAdmissionTest.
POTENTIALLY_INVARIANT_SIGNALS = {
    'response_latency',    # network reality (not DOM)
    'content_entropy',     # information-theoretic (text/element ratio)
    'surface_change_rate', # how much interactive surface changed (normalized)
    'interaction_density'  # interactive fraction of total (normalized)
}

SUBSTRATE_DIMS = ['S', 'I', 'P', 'A']



class SMO:
    """Self-Modifying Operator - bounded, reversible substrate updates."""

    def __init__(self, bounds: Tuple[float, float] = (0.0, 1.0), history_depth: int = 10):
        self.bounds = bounds
        self.prediction_error_history: deque = deque(maxlen=100)
        self.rigidity: float = 0.5
        self.state_history: deque = deque(maxlen=history_depth)
        self.rollback_available: bool = False

    def apply(self, current: float, observed_delta: float, predicted_delta: float = 0.0) -> float:
        self.state_history.append(current)
        self.rollback_available = True

        prediction_error = abs(observed_delta - predicted_delta)
        self.prediction_error_history.append(prediction_error)

        rigidity_change = 0.01 if prediction_error < 0.02 else -0.02
        rigidity_decay = -0.001
        self.rigidity = np.clip(self.rigidity + rigidity_change + rigidity_decay, 0.0, 1.0)

        modulated_delta = observed_delta * (1.0 - 0.3 * self.rigidity)
        new_value = np.clip(current + modulated_delta, *self.bounds)
        return new_value

    def get_recent_prediction_error(self, window: int = 10) -> float:
        if len(self.prediction_error_history) < window:
            return 0.0
        recent = list(self.prediction_error_history)[-window:]
        return np.mean(recent)

    def reverse(self) -> Optional[float]:
        if self.state_history:
            previous = self.state_history.pop()
            self.rollback_available = len(self.state_history) > 0
            return previous
        return None

    def can_reverse(self) -> bool:
        return self.rollback_available and len(self.state_history) > 0


class SubstrateState:
    """
    v15.1: Four-dimensional information processing geometry backed by DASS operators.

    S, I, P, A are properties derived from operators — all existing code that
    reads state.S, state.I, state.P, state.A continues to work unchanged.

    The scalar SMO (self.smo) is kept for backward compat:
    - predict_delta() in ContinuousRealityEngine reads smo.prediction_error_history
    - apply_delta() updates smo for error tracking while operators update separately
    - smo.rigidity is still logged for parallel validation against SelfModifyingOperator
    """

    def __init__(self,
                 sensing:     SensingOperator,
                 compression: CompressionOperator,
                 prediction:  PredictionOperator,
                 coherence:   CoherenceOperator):
        self.sensing     = sensing
        self.compression = compression
        self.prediction  = prediction
        self.coherence   = coherence
        # Backward-compat scalar SMO — kept for predict_delta error tracking
        self.smo = SMO(history_depth=10)

    # ── Scalar proxies — derived from operators ───────────────────────────────

    @property
    def S(self) -> float:
        return self.sensing.to_scalar_proxy()

    @property
    def I(self) -> float:
        return self.compression.to_scalar_proxy()

    @property
    def P(self) -> float:
        return self.prediction.to_grounded_proxy(self.sensing, self.compression)

    @property
    def A(self) -> float:
        return self.coherence.to_scalar_proxy()

    @A.setter
    def A(self, value: float):
        # Backward compat: MentatTriad._compute_a() writes self.state.A directly.
        # With operator-backed A this is a no-op — A is derived from coherence operator.
        # _compute_a() continues to run for its side-effects but the value is ignored here.
        # The coherence operator is the authoritative source.
        pass

    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}

    def apply_delta(self, observed_delta: Dict[str, float], predicted_delta: Dict[str, float] = None):
        """
        Backward-compat apply_delta: updates the scalar SMO for error tracking.
        Operator updates happen separately in MentatTriad's micro-perturbation loop.

        The scalar SMO still runs here so predict_delta()'s error estimates remain valid
        and log output (smo.rigidity) stays consistent during parallel validation.
        """
        if predicted_delta is None:
            predicted_delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

        # Update scalar SMO for each dimension — this populates prediction_error_history
        # which predict_delta() reads for I delta estimation.
        self.smo.apply(self.S, observed_delta.get('S', 0), predicted_delta.get('S', 0))
        self.smo.apply(self.I, observed_delta.get('I', 0), predicted_delta.get('I', 0))
        self.smo.apply(self.P, observed_delta.get('P', 0), predicted_delta.get('P', 0))

    def rollback(self) -> bool:
        """Backward compat stub — v15.1 rollback is handled by MentatTriad._restore_from_snapshot()."""
        return False


class StateTrace:
    """Ordered history of substrate states for field calculations."""

    def __init__(self, max_length: int = 1000):
        self.history: deque = deque(maxlen=max_length)
        # v15.2: C_local trajectory-field alignment tracking
        self.c_local_history: deque = deque(maxlen=100)
        self.c_global:        float = 0.0
        self._last_gradient:  Dict[str, float] = {}  # most recent ∇Φ — SRE/EGD access

    def compute_c_local(self, gradient: Dict[str, float],
                        sensing: 'SensingOperator') -> float:
        """
        C_local = ⟨∇Φ, ẋ⟩ / (‖∇Φ‖ · ‖ẋ‖)
        ẋ = last_delta across active channels.
        Returns 0.0 during bootstrap (gradient near-zero) — not gradient-lost.
        """
        x_dot  = {cid: ch.last_delta for cid, ch in sensing.channels.items()
                   if ch.active and cid in gradient}
        common = set(gradient.keys()) & set(x_dot.keys())
        if not common:
            return 0.0
        g_vec = np.array([gradient[k] for k in common])
        x_vec = np.array([x_dot[k]    for k in common])
        ng, nx = np.linalg.norm(g_vec), np.linalg.norm(x_vec)
        if ng < 1e-8 or nx < 1e-8:
            return 0.0   # bootstrap — undefined, not gradient-lost
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    @staticmethod
    def compute_c_local_static(gradient: Dict[str, float],
                                sensing: 'SensingOperator') -> float:
        """Static version for use in virtual trajectory scoring."""
        x_dot  = {cid: ch.last_delta for cid, ch in sensing.channels.items()
                   if ch.active and cid in gradient}
        common = set(gradient.keys()) & set(x_dot.keys())
        if not common:
            return 0.0
        g_vec = np.array([gradient[k] for k in common])
        x_vec = np.array([x_dot[k]    for k in common])
        ng, nx = np.linalg.norm(g_vec), np.linalg.norm(x_vec)
        if ng < 1e-8 or nx < 1e-8:
            return 0.0
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    def record(self, state: SubstrateState, gradient: Dict[str, float] = None,
               virtual: bool = False):
        """
        v15.2: gradient parameter added. Computes C_local internally every step.
        virtual=True: state enters history but C_local is NOT recorded (lab trajectories).
        gradient defaults to None — uses _last_gradient when not provided.
        """
        self.history.append(state.as_dict())
        if gradient is not None:
            self._last_gradient = gradient
        if not virtual and self._last_gradient:
            c_local = self.compute_c_local(self._last_gradient, state.sensing)
            self.c_local_history.append(c_local)
            self.c_global = float(np.mean(self.c_local_history))                             if self.c_local_history else 0.0

    def get_recent(self, n: int) -> List[Dict]:
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]

    def __len__(self) -> int:
        return len(self.history)


class PhiField:
    """
    Information Potential Field (Φ-field) — v15.3.

    Three-term potential over operator space:
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)

    α, β, γ are structural weights on the three components.
    They are not designer tuning of the dynamics — they scale
    compression quality, viable future volume, and attractor
    proximity relative to each other.
    """

    O_FLOOR = 1e-6   # prevents log(0); keeps O term active during bootstrap

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 A0: float = 0.7, alpha_crk: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.A0 = A0          # retained for phi_legacy only
        self.alpha_crk = alpha_crk  # retained for phi_legacy only

    # ── C(x) — unchanged from v15.2 ──────────────────────────────────────────

    def _compute_C(self, state) -> float:
        """
        C(x) = Σ_{edges} |w(e)|·conf(e)·cov(src)·cov(tgt)  /  (2·|edges|)
        Unchanged from v15.2 phi_geometry.
        """
        graph = state.compression.causal_graph
        if not graph:
            return 0.0
        total = 0.0
        for (src, tgt), edge in graph.items():
            src_cov = state.sensing.channels.get(
                src, SensingChannel(src, False, 0, 0, 0)).coverage
            tgt_cov = state.sensing.channels.get(
                tgt, SensingChannel(tgt, False, 0, 0, 0)).coverage
            total += abs(edge.weight) * edge.confidence * src_cov * tgt_cov
        max_possible = 2.0 * len(graph)
        return float(np.clip(total / max(max_possible, 1.0), 0.0, 1.0))

    # ── O(x) — NEW v15.3 ─────────────────────────────────────────────────────

    def _build_sigma_p(self, state):
        """
        Build Σ_P over active channels from causal graph.

        Two components:
          Diagonal:     Σ_P[i,i] = coverage_i
            Each active channel contributes its own viable future volume
            proportional to how much of its domain is currently reachable.
            Non-zero even before causal graph forms → O > O_FLOOR at bootstrap.
            Makes ∂O/∂coverage_i ≠ 0 via finite difference in gradient().

          Off-diagonal: Σ_P[i,j] = w(i→j) × conf(i→j) × coverage_i
            Causal edge contribution weighted by source channel coverage.
            Incorporating coverage_src makes the finite diff in gradient()
            sensitive to coverage changes — without this, g_O = 0 always.

        Returns (sigma_p, active_channel_list).
        sigma_p is symmetrized before return.
        """
        channels = state.sensing.channels
        active = [cid for cid, ch in channels.items() if ch.active]
        n = len(active)
        if n == 0:
            return np.zeros((0, 0)), []

        idx = {cid: i for i, cid in enumerate(active)}
        sigma_p = np.zeros((n, n))

        # Coverage diagonal: self-prediction variance = sensing surface reachability.
        # Every active channel with coverage > 0 IS a viable future direction —
        # we can observe changes there, even without a causal model of what causes them.
        for i, cid in enumerate(active):
            sigma_p[i, i] = channels[cid].coverage

        # Off-diagonal: causal co-prediction, weighted by source coverage.
        # += accumulates on top of diagonal, preserving the coverage signal.
        # coverage_src weighting ensures ∂Σ_P/∂coverage_src ≠ 0 → g_O ≠ 0.
        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                sigma_p[idx[src], idx[tgt]] += (
                    edge.weight * edge.confidence * channels[src].coverage
                )

        # Symmetrize — prediction covariance is symmetric
        sigma_p = (sigma_p + sigma_p.T) / 2.0
        return sigma_p, active

    def _direction_crk_viable(self, channel_delta: Dict[str, float], state) -> bool:
        """
        Check whether moving in a channel direction is CRK-viable.

        A direction is non-viable if it is predominantly ungrounded —
        i.e. it activates channels not represented in the causal graph,
        meaning the system would move into territory its compression
        operator has no model of.

        Non-viable conditions:
          1. > 70% of delta magnitude falls on channels absent from causal graph
          2. Direction predominantly reduces coverage of channels that anchor
             the causal graph (graph_channels with high total edge weight)

        Forming guard: when causal_graph is empty, no model exists yet.
        All directions are viable — the system has maximum optionality
        and must not reject sensing directions before any observations.
        Consistent with CRK C2 forming guard.
        """
        # Forming guard — no constraints before any causal structure exists
        if not state.compression.causal_graph:
            return True

        graph_channels = set()
        for (src, tgt) in state.compression.causal_graph:
            graph_channels.add(src)
            graph_channels.add(tgt)

        total_delta = sum(abs(v) for v in channel_delta.values())
        if total_delta < 1e-8:
            return True

        ungrounded = sum(abs(v) for cid, v in channel_delta.items()
                         if cid not in graph_channels)
        if ungrounded / total_delta > 0.7:
            return False

        # Direction that reduces high-weight channels is non-viable
        channel_weights = {}
        for (src, tgt), edge in state.compression.causal_graph.items():
            w = abs(edge.weight) * edge.confidence
            channel_weights[src] = channel_weights.get(src, 0.0) + w
            channel_weights[tgt] = channel_weights.get(tgt, 0.0) + w

        if channel_weights:
            max_w = max(channel_weights.values())
            threshold = max_w * 0.5
            anchors = {cid for cid, w in channel_weights.items() if w >= threshold}
            anchor_reduction = sum(
                -v for cid, v in channel_delta.items()
                if cid in anchors and v < 0
            )
            if anchor_reduction / total_delta > 0.5:
                return False

        return True

    def _compute_O(self, state) -> float:
        """
        O(x) = Σ_{λ_i > 0} λ_i  of  Σ_P filtered to CRK-viable directions.

        1. Build Σ_P over active channels (coverage diagonal + causal edges).
        2. Eigendecompose.
        3. For each eigenvector with positive eigenvalue, check CRK viability.
        4. Sum viable positive eigenvalues.

        Returns O_FLOOR if no active channels exist.
        During bootstrap (empty causal graph), Σ_P has coverage diagonal entries
        and forming guard makes all directions CRK-viable → O reflects sensing
        surface availability rather than collapsing to O_FLOOR.
        """
        sigma_p, active = self._build_sigma_p(state)
        if len(active) == 0:
            return self.O_FLOOR

        eigenvalues, eigenvectors = np.linalg.eigh(sigma_p)

        vol_opt = 0.0
        for k, lam in enumerate(eigenvalues):
            if lam <= 0:
                continue
            v = eigenvectors[:, k]
            channel_delta = {active[j]: float(v[j]) for j in range(len(active))}
            if self._direction_crk_viable(channel_delta, state):
                vol_opt += lam

        return float(max(vol_opt, self.O_FLOOR))

    # ── K(x) — NEW v15.3 ─────────────────────────────────────────────────────

    def _compute_K(self, state, peak_snapshot: Optional[Dict] = None) -> float:
        """
        K(x) = -(‖C_current - C_peak‖_F²  +  ‖cov_current - cov_peak‖²)

        peak_snapshot: dict with keys 'coupling_matrix' and 'coverage_vector'
                       from genome L2 peak optionality snapshot.
                       None → K = 0 (cold start; genome L2 not yet implemented).

        K = 0 at the attractor.
        K decreases as system drifts from peak.
        γ·K(x) < 0 always — drag term that grows as drift grows.
        """
        if peak_snapshot is None:
            return 0.0

        # Coupling matrix distance
        coupling_dist_sq = 0.0
        peak_coupling = peak_snapshot.get('coupling_matrix')
        if peak_coupling is not None:
            try:
                current_mat = state.compression.to_coupling_matrix()
                peak_mat = np.array(peak_coupling)
                diff = current_mat - peak_mat
                coupling_dist_sq = float(np.sum(diff ** 2))
            except Exception:
                pass

        # Coverage vector distance
        cov_dist_sq = 0.0
        peak_coverage = peak_snapshot.get('coverage_vector')
        if peak_coverage is not None:
            try:
                current_cov = np.array([
                    ch.coverage for ch in state.sensing.channels.values()
                    if ch.active
                ])
                peak_cov = np.array(peak_coverage)
                min_len = min(len(current_cov), len(peak_cov))
                if min_len > 0:
                    cov_dist_sq = float(np.sum(
                        (current_cov[:min_len] - peak_cov[:min_len]) ** 2
                    ))
            except Exception:
                pass

        return -(coupling_dist_sq + cov_dist_sq)

    # ── Φ(x) — unified v15.3 ─────────────────────────────────────────────────

    def phi(self, state, trace, peak_snapshot: Optional[Dict] = None) -> float:
        """
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)

        peak_snapshot: from genome L2 peak optionality snapshot.
                       Pass None until peak optionality tracker is implemented.
                       γ term drops out cleanly when None.

        All three terms grounded in operators:
          C(x) — CompressionOperator causal graph geometry
          O(x) — PredictionOperator viable future volume
          K(x) — CoherenceOperator distance from peak attractor state

        S enters all three as the sensing surface that makes them computable:
          no active channels → C = 0, O = O_FLOOR, K coverage term = 0
        """
        C = self._compute_C(state)
        O = self._compute_O(state)
        K = self._compute_K(state, peak_snapshot)

        phi_val = (self.alpha * C
                   + self.beta * np.log(max(O, self.O_FLOOR))
                   + self.gamma * K)

        # Normalize: log(O) can be negative (when O < 1) and K is always ≤ 0.
        # Keep Φ unbounded below — collapse is visible.
        # Clip only the upper end for numerical stability.
        return float(np.clip(phi_val, -100.0, 100.0))

    # ── ∇Φ — updated v15.3 ───────────────────────────────────────────────────

    def gradient(self, state, trace,
                 peak_snapshot: Optional[Dict] = None) -> Dict[str, float]:
        """
        ∇Φ(x) per channel, normalized to unit vector.

        Three contributions:
          ∂C/∂channel_i  — closed form (unchanged from v15.2)
          ∂O/∂channel_i  — finite difference (eigenvalues of Σ_P via coverage)
          ∂K/∂channel_i  = -2·(cov_i - cov_peak_i)  for coverage contribution

        β/O(x) weighting on ∂O term means gradient explodes as viable futures
        collapse — correct behavior, pulls system toward recovery automatically.
        """
        channels = state.sensing.channels
        grad = {}

        O = self._compute_O(state)

        # Coverage vector for K gradient
        peak_coverage = {}
        if peak_snapshot is not None:
            peak_cov_list = peak_snapshot.get('coverage_vector', [])
            active_ids = [cid for cid, ch in channels.items() if ch.active]
            for i, cid in enumerate(active_ids):
                if i < len(peak_cov_list):
                    peak_coverage[cid] = peak_cov_list[i]

        graph = state.compression.causal_graph

        for cid, ch in channels.items():
            if not ch.active:
                continue

            # ∂C/∂channel_i (closed form)
            g_C = 0.0
            for (src, tgt), edge in graph.items():
                if src == cid:
                    tgt_cov = channels.get(
                        tgt, SensingChannel(tgt, False, 0, 0, 0)).coverage
                    g_C += edge.weight * edge.confidence * tgt_cov
                elif tgt == cid:
                    src_cov = channels.get(
                        src, SensingChannel(src, False, 0, 0, 0)).coverage
                    g_C += edge.weight * edge.confidence * src_cov
            # Normalize by max_possible (same as phi computation)
            max_possible = 2.0 * max(len(graph), 1)
            g_C /= max_possible

            # ∂O/∂channel_i (finite difference)
            # Perturb coverage of channel_i by ε, recompute O
            g_O = 0.0
            eps = 1e-4
            try:
                perturbed_channels = dict(channels)
                perturbed_ch = dataclasses.replace(
                    ch, coverage=float(np.clip(ch.coverage + eps, 0.0, 1.0))
                )
                perturbed_channels[cid] = perturbed_ch

                from uii_operators import SensingOperator as _SO
                ps_sensing = _SO(channels=perturbed_channels)

                class _PState:
                    pass
                ps = _PState()
                ps.sensing = ps_sensing
                ps.compression = state.compression

                O_perturbed = self._compute_O(ps)
                g_O = (O_perturbed - O) / eps
            except Exception:
                g_O = 0.0

            # ∂K/∂channel_i = -2·(cov_i - cov_peak_i)
            g_K = 0.0
            if cid in peak_coverage:
                g_K = -2.0 * (ch.coverage - peak_coverage[cid])

            # Combine with term weights
            # β/O weighting on ∂O is automatic via chain rule
            grad[cid] = (self.alpha * g_C
                         + self.beta / max(O, self.O_FLOOR) * g_O
                         + self.gamma * g_K)

        # Normalize to unit vector
        norm = float(np.sqrt(sum(v ** 2 for v in grad.values())))
        if norm > 1e-8:
            grad = {k: v / norm for k, v in grad.items()}
        return grad

    # ── phi_legacy — retained for validation ─────────────────────────────────

    @staticmethod
    def si_capacity(S: float, I: float) -> float:
        """
        Single source of truth for the SI_capacity grounding formula.
        Retained for phi_legacy and backward compat callers.
        """
        return min(S, I) * 0.5 + (S + I) / 4.0

    def phi_legacy(self, state, trace,
                   crk_violations=None) -> float:
        """
        v15.2 legacy formula retained for parallel validation.
        α·log(1+P) - β·(A-A₀)² - γ·curvature - CRK_penalty
        Remove after phi_geometry convergence confirmed.
        """
        cap = self.si_capacity(state.S, state.I)
        effective_P = min(state.P, cap * 2.0)
        opt = np.log(1.0 + max(effective_P, 0.0))
        strain = (state.A - self.A0) ** 2

        recent = trace.get_recent(3)
        curv = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2 * h1[k] + h0[k])

        phi_raw = self.alpha * opt - self.beta * strain - self.gamma * curv
        crk_penalty = 0.0
        if crk_violations:
            crk_penalty = self.alpha_crk * sum(
                severity for _, severity in crk_violations
            )
        return phi_raw - crk_penalty



# ──────────────────────────────────────────────────────────────────────────────
# v15.1: CRK shared structures — pre/post action evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CRKEvaluation:
    constraint:  str     # 'C1'–'C7'
    phase:       str     # 'pre_action' | 'post_action'
    status:      str     # 'satisfied' | 'degraded' | 'violated'
    risk:        float   # [0,1]
    attribution: str     # 'internal' | 'external' | 'mixed'
    blocks:      bool    # violated constraints block action or roll back SMO
    signal:      str     # what triggered this


@dataclass
class CRKVerdict:
    phase:        str                    # 'pre_action' | 'post_action'
    action:       str
    evaluations:  List[CRKEvaluation]
    coherent:     bool
    repair:       Optional[str]          # routing signal, not forced action
    phi_modifier: float                  # [0,1] — pre_action: scales Φ score
    smo_permitted: bool                  # post_action: whether SMO update is allowed


class CRKMonitor:
    """
    Constraint Recognition Kernel (CRK) — v15.1.

    Operates at two points:
      Pre-action:  evaluates quality of a choice before execution.
                   Filters action manifold. Returns phi_modifier.
                   Violated constraints block the action.
      Post-action: evaluates quality of SMO update after Reality responds.
                   Gates whether adaptation is grounded. Failed → rollback.

    Old evaluate() stays — run in parallel for first run to validate agreement.
    """

    # ── Shared constants ──────────────────────────────────────────────────────

    PREDICTION_ERROR_THRESHOLD = 0.05   # ε — min external mismatch to trigger SMO

    # Actions that commit to an irreversible path
    IRREVERSIBLE_ACTIONS = {'migrate', 'navigate', 'python'}
    # Actions that commit (vs. exploratory)
    COMMITTING_ACTIONS   = {'migrate', 'navigate', 'python', 'click', 'fill', 'type'}

    # ── Pre-action evaluation ─────────────────────────────────────────────────

    def evaluate_pre_action(self,
                             proposed_action: str,
                             coherence:       CoherenceOperator,
                             sensing:         SensingOperator,
                             compression:     CompressionOperator,
                             prediction:      PredictionOperator,
                             field_state:     Dict,
                             ) -> 'CRKVerdict':
        """
        Pre-action CRK: evaluates proposed action before execution.
        Returns CRKVerdict with phi_modifier and coherent flag.
        """
        crk_sig    = coherence.crk_signal()
        loop_cl    = crk_sig['loop_closure']
        sig_dev    = crk_sig['signature_deviation']
        i_p        = crk_sig['i_p_consistency']
        p_proxy    = prediction.to_grounded_proxy(sensing, compression)
        evaluations: List[CRKEvaluation] = []

        # C1: Does this conflict with what I am?
        irreversible = proposed_action in self.IRREVERSIBLE_ACTIONS
        c1_blocks    = loop_cl < 0.3 and irreversible and sig_dev > 0.3

        # v15.2 Step 12: δ²Φ stability check — additive to loop_closure/sig_dev
        # Negative stability = action pushes into amplifying direction
        c1_stability_risk = 0.0
        try:
            coupling   = compression.to_coupling_matrix()
            pred_delta = self._get_predicted_delta(proposed_action, compression)
            delta_vec  = np.array([pred_delta.get(d, 0.0) for d in ['S', 'I', 'P', 'A']])
            stability  = float(0.5 * delta_vec @ coupling @ delta_vec)
            # Only degrades C1 combined with low loop_closure — does not block independently
            c1_stability_risk = float(np.clip(-stability, 0.0, 1.0)) if stability < -0.1 else 0.0
        except Exception:
            pass  # to_coupling_matrix not yet available — skip gracefully

        base_c1_risk = float(np.clip(1.0 - loop_cl + sig_dev + c1_stability_risk * 0.3, 0.0, 1.0))
        evaluations.append(CRKEvaluation(
            constraint  = 'C1',
            phase       = 'pre_action',
            status      = 'violated' if c1_blocks else ('degraded' if loop_cl < 0.5 else 'satisfied'),
            risk        = base_c1_risk,
            attribution = 'internal',
            blocks      = c1_blocks,
            signal      = (f'loop_closure={loop_cl:.2f}, sig_dev={sig_dev:.2f}, '
                           f'irreversible={irreversible}, stability_risk={c1_stability_risk:.3f}'),
        ))

        # C2: Does this restrict future state space?
        # Formation guard: empty causal_graph means no observations have been
        # integrated yet. p_proxy = 0 here is maximum openness, not collapse —
        # there are no constraints yet to violate. C2 blocking committing actions
        # before any observations exist prevents the geometry from ever forming.
        committing    = proposed_action in self.COMMITTING_ACTIONS
        forming       = len(compression.causal_graph) == 0
        c2_blocks     = p_proxy < 0.15 and committing and not forming
        evaluations.append(CRKEvaluation(
            constraint  = 'C2',
            phase       = 'pre_action',
            status      = 'satisfied' if forming else (
                          'violated' if c2_blocks else (
                          'degraded' if p_proxy < 0.3 else 'satisfied')),
            risk        = 0.0 if forming else (
                          float(np.clip(0.3 - p_proxy, 0.0, 1.0)) if p_proxy < 0.3 else 0.0),
            attribution = 'internal',
            blocks      = c2_blocks,
            signal      = f'p_proxy={p_proxy:.2f}, committing={committing}, forming={forming}',
        ))

        # C3: Am I internalizing failure distorting this choice?
        c3_blocks = i_p < 0.3
        evaluations.append(CRKEvaluation(
            constraint  = 'C3',
            phase       = 'pre_action',
            status      = 'violated' if c3_blocks else ('degraded' if i_p < 0.5 else 'satisfied'),
            risk        = float(np.clip(0.5 - i_p, 0.0, 1.0)),
            attribution = 'internal',
            blocks      = c3_blocks,
            signal      = f'i_p_consistency={i_p:.2f}',
        ))

        # C6: Am I assuming sole agency? (never blocks — degrades score)
        system_load = field_state.get('system_load', 0.0)
        evaluations.append(CRKEvaluation(
            constraint  = 'C6',
            phase       = 'pre_action',
            status      = 'degraded' if system_load > 0.7 else 'satisfied',
            risk        = float(np.clip(system_load - 0.7, 0.0, 1.0)) if system_load > 0.7 else 0.0,
            attribution = 'external',
            blocks      = False,   # C6 never blocks
            signal      = f'system_load={system_load:.2f}',
        ))

        # C7: Does this destabilize the field?
        phi_trend  = field_state.get('phi_trend', 0.0)
        c7_blocks  = proposed_action == 'migrate' and loop_cl < 0.5
        evaluations.append(CRKEvaluation(
            constraint  = 'C7',
            phase       = 'pre_action',
            status      = 'violated' if c7_blocks else ('degraded' if phi_trend < -0.1 else 'satisfied'),
            risk        = float(np.clip(0.5 - loop_cl, 0.0, 1.0)) if c7_blocks else 0.0,
            attribution = 'mixed',
            blocks      = c7_blocks,
            signal      = f'migrate+loop_cl={loop_cl:.2f}' if c7_blocks else f'phi_trend={phi_trend:.2f}',
        ))

        # phi_modifier
        phi_modifier = float(np.prod([
            1.0 - (e.risk * (1.0 if e.status == 'degraded' else 2.0))
            for e in evaluations
        ]))
        phi_modifier = float(np.clip(phi_modifier, 0.0, 1.0))

        coherent  = not any(e.blocks for e in evaluations)

        # Repair routing
        repair = None
        violated = [e for e in evaluations if e.status == 'violated']
        if violated:
            repair = {
                'C1': 'stabilize',
                'C2': 'expand',
                'C3': 'reattribute',
                'C7': 'stabilize',
            }.get(violated[0].constraint)
        elif any(e.constraint == 'C6' and e.status == 'degraded' for e in evaluations):
            repair = 'coordinate'

        return CRKVerdict(
            phase        = 'pre_action',
            action       = proposed_action,
            evaluations  = evaluations,
            coherent     = coherent,
            repair       = repair,
            phi_modifier = phi_modifier,
            smo_permitted = True,   # pre-action doesn't gate SMO
        )

    def _get_predicted_delta(self, action: str, compression: 'CompressionOperator') -> Dict[str, float]:
        """
        v15.2 Step 12: look up predicted SIPA delta for action from compression graph.
        Falls back to zero vector if action unknown.
        """
        # Attempt to read from action_substrate_map if available on compression
        action_map = getattr(compression, 'action_substrate_map', {}) or {}
        if action in action_map:
            return dict(action_map[action])
        # Hardcoded fallback — same table as ContinuousRealityEngine.predict_delta
        fallback = {
            'navigate': {'S': 0.05, 'I': 0.0, 'P': -0.08, 'A': 0.0},
            'click':    {'S': 0.02, 'I': 0.0, 'P': -0.02, 'A': 0.0},
            'read':     {'S': 0.03, 'I': 0.0, 'P':  0.0,  'A': 0.0},
            'observe':  {'S': 0.0,  'I': 0.0, 'P':  0.0,  'A': 0.0},
        }
        return fallback.get(action, {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0})

    # ── Post-action evaluation ────────────────────────────────────────────────

    def evaluate_post_action(self,
                              proposed_smo_update: Dict,
                              observed_delta:      Dict,
                              predicted_delta:     Dict,
                              sensing:             SensingOperator,
                              compression:         CompressionOperator,
                              prediction:          PredictionOperator,
                              coherence:           CoherenceOperator,
                              prior_compression:   CompressionOperator,
                              smo_plasticity:      float = 0.5,
                              ) -> 'CRKVerdict':
        """
        Post-action CRK: evaluates whether SMO update is grounded.
        Returns CRKVerdict with smo_permitted flag.

        smo_plasticity: current plasticity from SelfModifyingOperator.
                        Used by C3 (Non-Internalization) to detect plasticity
                        surging in response to external error signals.
        """
        evaluations: List[CRKEvaluation] = []

        evaluations.append(self._c4_post(observed_delta, predicted_delta, sensing, compression))
        evaluations.append(self._c5_post(observed_delta, predicted_delta, coherence))
        evaluations.append(self._c1_post(proposed_smo_update, prior_compression, compression))
        evaluations.append(self._c3_post(proposed_smo_update, smo_plasticity))

        smo_permitted = not any(e.blocks for e in evaluations)

        # Repair routing
        repair = None
        if not smo_permitted:
            c4 = next((e for e in evaluations if e.constraint == 'C4' and e.blocks), None)
            c1 = next((e for e in evaluations if e.constraint == 'C1' and e.blocks), None)
            c5 = next((e for e in evaluations if e.constraint == 'C5' and e.status in ('degraded', 'violated')), None)
            if c4:
                repair = None   # SMO blocked — no update, continue with current model
            elif c1:
                repair = 'rollback'
            elif c5:
                repair = 'reattribute'

        # C3 repair fires even when SMO is permitted — reattribute without blocking
        if repair is None:
            c3_violated = next((e for e in evaluations
                                if e.constraint == 'C3' and e.status == 'violated'), None)
            if c3_violated:
                repair = 'reattribute'

        return CRKVerdict(
            phase         = 'post_action',
            action        = str(proposed_smo_update),
            evaluations   = evaluations,
            coherent      = smo_permitted,
            repair        = repair,
            phi_modifier  = 1.0,   # post-action doesn't modify phi
            smo_permitted = smo_permitted,
        )

    def _c3_post(self, prediction_errors: Dict,
                 smo_plasticity: float) -> 'CRKEvaluation':
        """
        C3 post-action: Non-Internalization.
        High external error + plasticity surge = internalizing external
        constraint as self-failure. Repair: reattribute. Does not block.
        """
        mean_error = float(np.mean(list(prediction_errors.values()))) \
                     if prediction_errors else 0.0

        if mean_error > 0.15 and smo_plasticity > 0.75:
            return CRKEvaluation(
                'C3', 'post_action', 'violated',
                smo_plasticity, 'internal', False,
                f'mean_error={mean_error:.3f} (external), '
                f'plasticity={smo_plasticity:.2f} — internalising external resistance'
            )
        if mean_error > 0.10 and smo_plasticity > 0.65:
            return CRKEvaluation(
                'C3', 'post_action', 'degraded',
                smo_plasticity * 0.5, 'internal', False,
                f'mean_error={mean_error:.3f}, plasticity={smo_plasticity:.2f} — watch'
            )
        return CRKEvaluation('C3', 'post_action', 'satisfied',
                             0.0, 'internal', False,
                             f'mean_error={mean_error:.3f}, plasticity={smo_plasticity:.2f}')

    def _c4_post(self, observed: Dict, predicted: Dict,
                 sensing: SensingOperator,
                 compression: 'CompressionOperator') -> 'CRKEvaluation':
        """
        C4 post-action: Is prediction error from external Reality or internal noise?
        Only external mismatch grounds SMO adaptation.

        Formation guard: empty causal_graph means no model exists yet.
        Any observation is external mismatch by definition — C4 satisfied.

        Channel key fix: sensing.channels uses ids like 'browser', 'resource_cpu'.
        observed_delta and predicted_delta use SIPA keys 'S','I','P','A'.
        Per-channel lookup always returns 0 for both — mean_error = 0 — which
        incorrectly fires C4 violated. Fall through to SIPA scalar comparison
        directly when no channel-level data is present in the delta dicts.
        """
        # Formation guard: no causal graph = no model = all signal is external
        if len(compression.causal_graph) == 0:
            return CRKEvaluation('C4', 'post_action', 'satisfied',
                                 0.0, 'external', False,
                                 'forming — no model yet, all signal is external mismatch')

        per_channel_error = {}
        for channel_id, ch in sensing.channels.items():
            if not ch.active:
                continue
            obs  = observed.get(channel_id, None)
            pred = predicted.get(channel_id, None)
            # Only include if the delta dict actually carries this channel's data
            if obs is None and pred is None:
                continue
            obs  = obs  if obs  is not None else 0.0
            pred = pred if pred is not None else 0.0
            if isinstance(obs, dict):
                obs = obs.get('magnitude', 0.0)
            if isinstance(pred, dict):
                pred = pred.get('magnitude', 0.0)
            per_channel_error[channel_id] = abs(float(obs) - float(pred))

        # Fall back to SIPA scalars when delta dicts carry no channel-level data
        if not per_channel_error:
            for dim in ['S', 'I', 'P', 'A']:
                obs_val  = observed.get(dim, 0.0)
                pred_val = predicted.get(dim, 0.0)
                if isinstance(obs_val, (int, float)) and isinstance(pred_val, (int, float)):
                    per_channel_error[dim] = abs(float(obs_val) - float(pred_val))

        if not per_channel_error:
            return CRKEvaluation('C4', 'post_action', 'satisfied',
                                 0.0, 'external', False, 'no active channels')

        mean_error = float(np.mean(list(per_channel_error.values())))

        if mean_error < self.PREDICTION_ERROR_THRESHOLD:
            return CRKEvaluation(
                'C4', 'post_action', 'violated',
                1.0 - mean_error / self.PREDICTION_ERROR_THRESHOLD,
                'internal', True,
                f'mean_error={mean_error:.4f} < ε={self.PREDICTION_ERROR_THRESHOLD}'
                f' — adaptation not grounded in external mismatch'
            )

        return CRKEvaluation(
            'C4', 'post_action', 'satisfied',
            mean_error, 'external', False,
            f'mean_error={mean_error:.4f} — external mismatch confirmed'
        )

    def _c1_post(self, proposed_update: Dict,
                 prior_compression: CompressionOperator,
                 new_compression:   CompressionOperator) -> 'CRKEvaluation':
        """
        C1 post-action: Does this SMO update preserve continuity?
        SMO⁻¹ must exist — reversibility is a hard invariant.
        """
        prior_edges     = set(prior_compression.causal_graph.keys())
        new_edges       = set(new_compression.causal_graph.keys())
        edges_destroyed = prior_edges - new_edges

        if edges_destroyed:
            destruction_ratio = len(edges_destroyed) / max(len(prior_edges), 1)
            if destruction_ratio > 0.3:
                return CRKEvaluation(
                    'C1', 'post_action', 'violated',
                    destruction_ratio, 'internal', True,
                    f'{len(edges_destroyed)} edges destroyed'
                    f' ({destruction_ratio:.1%}) — historical erasure, roll back'
                )

        # Check high-confidence edges weren't zeroed
        confidence_loss = []
        for key in prior_edges & new_edges:
            prior_conf = prior_compression.causal_graph[key].confidence
            new_conf   = new_compression.causal_graph[key].confidence
            if prior_conf > 0.5 and new_conf < 0.1:
                confidence_loss.append(key)

        if len(confidence_loss) > 3:
            return CRKEvaluation(
                'C1', 'post_action', 'degraded',
                len(confidence_loss) / max(len(prior_edges), 1),
                'internal', False,
                f'{len(confidence_loss)} high-confidence edges collapsed'
            )

        return CRKEvaluation('C1', 'post_action', 'satisfied',
                             0.0, 'internal', False, 'continuity preserved')

    def _c5_post(self, observed: Dict, predicted: Dict,
                 coherence: CoherenceOperator) -> 'CRKEvaluation':
        """
        C5 post-action: Is the adaptation correctly attributed?
        If i_p is low and p_a is fine, error is likely internal compression.
        """
        i_p = coherence.consistency.i_p_consistency
        p_a = coherence.consistency.p_a_consistency

        if i_p < 0.4 and p_a > 0.6:
            return CRKEvaluation(
                'C5', 'post_action', 'degraded',
                1.0 - i_p, 'internal', False,
                f'i_p={i_p:.2f} low, p_a={p_a:.2f} fine — '
                f'error source is internal compression, not external Reality. '
                f'Reattribute accurately before SMO update.'
            )

        return CRKEvaluation('C5', 'post_action', 'satisfied',
                             0.0, 'external', False, 'attribution clear')

    # ── Legacy evaluate() — kept for parallel validation ─────────────────────

    def evaluate(self, state, trace: 'StateTrace',
                 reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Original scalar CRK — kept for parallel validation against new pre/post split.
        Run both, log both, compare agreement. Do not remove until validated.
        """
        violations = []

        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            jump = sum(abs(prev[k] - getattr(state, k)) for k in ["S", "I", "P", "A"])
            if jump > 0.3:
                violations.append(("C1_Continuity", jump - 0.3))

        if state.P < 0.35:
            violations.append(("C2_Optionality", 0.35 - state.P))

        if len(trace) >= 5:
            recent_5 = trace.get_recent(5)
            conf_values = [r["S"] + r["I"] for r in recent_5]
            conf_monotone_declining = all(
                conf_values[i] >= conf_values[i+1]
                for i in range(len(conf_values) - 1)
            )
            externally_driven = (
                reality_delta is not None and
                sum(abs(v) for v in reality_delta.values() if isinstance(v, (int, float))) > 0.05
            )
            if conf_monotone_declining and not externally_driven:
                severity = conf_values[0] - conf_values[-1]
                if severity > 0.1:
                    violations.append(("C3_NonInternalization", severity))

        if reality_delta and len(trace) >= 3:
            feedback_magnitude = sum(abs(v) for v in reality_delta.values()
                                     if isinstance(v, (int, float)))
            if feedback_magnitude < 0.01:
                violations.append(("C4_Reality", 0.01 - feedback_magnitude))

        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            prev_P    = prev["P"]
            prev_conf = prev["S"] + prev["I"]
            curr_conf = state.S + state.I

            if state.P < prev_P and curr_conf < prev_conf:
                violations.append(("C5_Attribution", min(prev_P - state.P, 1.0)))

        if state.S < 0.3:
            violations.append(("C6_Agenthood", 0.3 - state.S))

        if abs(state.A - 0.7) > 0.4:
            violations.append(("C7_GlobalCoherence", abs(state.A - 0.7) - 0.4))

        return violations


# ============================================================
# MODULE 2: TRAJECTORY MANIFOLD INFRASTRUCTURE
# ============================================================



# ============================================================
# TRAJECTORY INFRASTRUCTURE
# ============================================================

@dataclass
class TrajectoryCandidate:
    """Multi-step executable procedure with structural annotations."""
    steps: List[Dict]
    rationale: str
    estimated_coherence_preservation: float
    estimated_optionality_delta: float
    reversibility_point: int

    tested: bool = False
    test_phi_final: Optional[float] = None
    test_state_final: Optional[Dict] = None
    test_violations: Optional[List] = None
    test_perturbation_trace: Optional[List] = None
    test_succeeded: bool = False
    # v14.1: virtual mode predicted phi (set by test_trajectory_virtual before real execution)
    virtual_phi: Optional[float] = None

    def __repr__(self):
        status = "✓" if self.test_succeeded else "✗" if self.tested else "?"
        phi_str = f"Φ={self.test_phi_final:.3f}" if self.test_phi_final is not None else "untested"
        return f"{status} [{len(self.steps)} steps] {self.rationale[:50]} ({phi_str})"


@dataclass
class TrajectoryManifold:
    """Container for enumerated trajectory space"""
    candidates: List[TrajectoryCandidate]
    enumeration_context: Dict

    def size(self) -> int:
        return len(self.candidates)

    def tested_count(self) -> int:
        return sum(1 for c in self.candidates if c.tested)

    def get_best(self) -> Optional[TrajectoryCandidate]:
        """Return highest-scoring tested trajectory"""
        valid = [c for c in self.candidates if c.tested and c.test_succeeded]
        if not valid:
            return None
        return max(valid, key=lambda c: c.test_phi_final)

    def get_all_tested(self) -> List[TrajectoryCandidate]:
        """Return all tested trajectories sorted by score"""
        tested = [c for c in self.candidates if c.tested]
        return sorted(tested, key=lambda c: c.test_phi_final if c.test_phi_final is not None else -1000, reverse=True)


# ============================================================
# MODULE 3.5: AGENT INFRASTRUCTURE
# ============================================================


# ============================================================
# AGENT INFRASTRUCTURE (CRK C6: Other-Agent Existence)
# ============================================================

class AgentHandler(ABC):
    """
    Interface for agent interaction.

    Agents are other intelligences the Triad can query.
    CRK C6 (Other-Agent Existence) made concrete.
    """

    @abstractmethod
    def post_query(self, triad_id: str, query_text: str):
        """Post a query to this agent (non-blocking)"""
        pass

    @abstractmethod
    def get_response(self, triad_id: str) -> Optional[str]:
        """Check if agent has responded (returns None if still pending)"""
        pass


class UserAgentHandler(AgentHandler):
    """
    Human user as agent.

    Non-blocking: Triad posts query, continues micro-perturbations,
    integrates response when available.
    """

    def __init__(self):
        self.pending_queries: deque = deque()
        self.responses: Dict[str, str] = {}

    def post_query(self, triad_id: str, query_text: str):
        """Post query for user to see"""
        self.pending_queries.append({
            'triad_id': triad_id,
            'query': query_text,
            'timestamp': time.time()
        })

        print(f"\n{'='*70}")
        print(f"[QUERY FROM TRIAD {triad_id}]")
        print(f"{query_text}")
        print(f"{'='*70}")
        print(f"Respond with: triad.respond_to_query('{triad_id}', 'your answer')")
        print(f"Or leave pending - Triad will continue exploration")
        print(f"{'='*70}\n")

    def get_response(self, triad_id: str) -> Optional[str]:
        """Check if user has responded"""
        return self.responses.pop(triad_id, None)

    def respond(self, triad_id: str, answer: str):
        """User provides response"""
        self.responses[triad_id] = answer

    def has_pending(self) -> bool:
        """Check if any queries pending"""
        return len(self.pending_queries) > 0

    def get_pending_count(self) -> int:
        """Number of pending queries"""
        return len(self.pending_queries)


AVAILABLE_AGENTS = {
    'user': UserAgentHandler()
}



# ============================================================
# ADAPTER ABSTRACT INTERFACES
# ============================================================

class RealityAdapter(ABC):
    """Interface for environment/perturbation source."""

    @abstractmethod
    def execute(self, action: Dict) -> Tuple[Dict[str, float], Dict]:
        pass

    @abstractmethod
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        pass

    @abstractmethod
    def get_current_affordances(self) -> Dict:
        pass

    @abstractmethod
    def close(self):
        pass


class IntelligenceAdapter(ABC):
    """Interface for Relation component of Mentat Triad."""

    @abstractmethod
    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        pass

    @abstractmethod
    def record_committed_trajectory(self, trajectory: TrajectoryCandidate, phi_final: float):
        pass


# ============================================================
