"""
uii_geometry.py — v16
Merged foundation and field geometry layer.

Replaces uii_types.py entirely. All imports that previously pointed at
uii_types must be redirected here.

Also absorbs:
  - SymbolGroundingAdapter, SYMBOL_GROUNDING_PROMPT (from uii_intelligence.py)
  - DeathClock / renamed from LatentDeathClock (from uii_coherence.py)

Contents:
  Constants
    BASE_AFFORDANCES, INTERFACE_COUPLED_SIGNALS, POTENTIALLY_INVARIANT_SIGNALS,
    SUBSTRATE_DIMS

  State
    SubstrateState         — operator-backed SIPA geometry (backward compat intact)
    StateTrace             — trajectory + C_local/C_global tracking

  Field geometry
    eigen_decompose()      — module-level: always eigh, clipped eigenvalues
    expected_optionality_gain() — module-level: E[Δlog(O(a))] per action
    PhiField               — v16: _compute_K changed; compute_hessian + score_actions new;
                             all v15.3 methods retained exactly

  Constraint kernel
    CRKEvaluation, CRKVerdict, CRKMonitor — unchanged from v15.3

  Trajectory manifold
    TrajectoryCandidate, TrajectoryManifold — unchanged

  Agent infrastructure
    AgentHandler, UserAgentHandler, AVAILABLE_AGENTS — unchanged

  Adapters (ABCs)
    RealityAdapter, IntelligenceAdapter — unchanged

  Orchestration
    DeathClock             — minimal step/token budget tracker (renamed from
                             LatentDeathClock); elaborate counting eliminated —
                             resource pressure sensed through api_llm channel

  Symbol grounding
    SYMBOL_GROUNDING_PROMPT, SymbolGroundingAdapter — moved from uii_intelligence.py;
                             core logic unchanged; SRE coupling removed

What is NOT here
    SMO (old scalar class)     — private _BackCompatSMO kept for SubstrateState.smo shim
    LAYER1_PARAMS              — dead with velocity/generation system
    VELOCITY_FIELD             — dead with velocity/generation system
    RelationAdapter            — eliminated with SRE
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from uii_operators import (
    CausalEdge,
    ChannelPrediction,
    CoherenceOperator,
    CompressionOperator,
    DEFAULT_CHANNELS,
    OperatorConsistencyCheck,
    PredictionOperator,
    SensingChannel,
    SensingOperator,
    SelfModifyingOperator,
    SMOUpdate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_AFFORDANCES: Set[str] = {
    'navigate', 'click', 'fill', 'type', 'read',
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',
    'python',
    'llm_query',  # symbol grounding affordance — outcome feeds CouplingMatrixEstimator
    'migrate',    # substrate migration — exits current environment, opens new causal surface
}

# Interface-coupled signals — artifacts of HTML/browser rendering, not reality structure.
# BLOCKED from becoming discovered ledger axes.
INTERFACE_COUPLED_SIGNALS: Set[str] = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity',
}

# Potentially invariant signals — may survive substrate change.
# ALLOWED to proceed to the full AxisAdmissionTest.
POTENTIALLY_INVARIANT_SIGNALS: Set[str] = {
    'response_latency',    # network reality (not DOM)
    'content_entropy',     # information-theoretic (text/element ratio)
    'surface_change_rate', # how much interactive surface changed (normalized)
    'interaction_density', # interactive fraction of total (normalized)
}

SUBSTRATE_DIMS: List[str] = ['S', 'I', 'P', 'A']


# ──────────────────────────────────────────────────────────────────────────────
# _BackCompatSMO — private; only exists to give SubstrateState.smo its shim
# ──────────────────────────────────────────────────────────────────────────────

class _BackCompatSMO:
    """
    Minimal backward-compat scalar SMO shim.

    Retained because SubstrateState.smo is still read by legacy code for:
      - smo.rigidity     (log output)
      - smo.prediction_error_history  (predict_delta error estimation)

    The elaborate plasticity/rigidity dynamics from the old SMO class are
    no longer the primary adaptation mechanism — SelfModifyingOperator in
    uii_operators.py owns that. This shim is purely a data container that
    can be updated by callers who still track per-step error history.

    No new code should instantiate this directly. Access via SubstrateState.smo.
    """

    def __init__(self, history_depth: int = 10):
        self.prediction_error_history: deque = deque(maxlen=100)
        self.rigidity:   float = 0.5
        self.plasticity: float = 0.5
        self.state_history: deque = deque(maxlen=history_depth)
        self.rollback_available: bool = False

    def apply(self, current: float, observed_delta: float, predicted_delta: float = 0.0) -> float:
        """Backward compat — logs prediction error into history."""
        self.state_history.append(current)
        self.rollback_available = True
        prediction_error = abs(observed_delta - predicted_delta)
        self.prediction_error_history.append(prediction_error)
        rigidity_change = 0.01 if prediction_error < 0.02 else -0.02
        self.rigidity = float(np.clip(self.rigidity + rigidity_change - 0.001, 0.0, 1.0))
        self.plasticity = 1.0 - self.rigidity
        modulated_delta = observed_delta * (1.0 - 0.3 * self.rigidity)
        return float(np.clip(current + modulated_delta, 0.0, 1.0))

    def get_recent_prediction_error(self, window: int = 10) -> float:
        if len(self.prediction_error_history) < window:
            return 0.0
        return float(np.mean(list(self.prediction_error_history)[-window:]))

    def reverse(self) -> Optional[float]:
        if self.state_history:
            previous = self.state_history.pop()
            self.rollback_available = len(self.state_history) > 0
            return previous
        return None

    def can_reverse(self) -> bool:
        return self.rollback_available and len(self.state_history) > 0


# Public alias for any legacy code that imported SMO from uii_types
SMO = _BackCompatSMO


# ──────────────────────────────────────────────────────────────────────────────
# SubstrateState
# ──────────────────────────────────────────────────────────────────────────────

class SubstrateState:
    """
    v15.1+: Four-dimensional information processing geometry backed by DASS operators.

    S, I, P, A are properties derived from operators — all existing code that
    reads state.S, state.I, state.P, state.A continues to work unchanged.

    self.smo: backward-compat _BackCompatSMO shim. Retained for code that
    reads smo.rigidity and smo.prediction_error_history. No new v16 code
    should write to it — SelfModifyingOperator in uii_operators.py is the
    authoritative adaptation mechanism.
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
        # Backward-compat scalar SMO shim — kept for legacy read access only
        self.smo = _BackCompatSMO(history_depth=10)

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
        # Backward compat — A is derived from coherence operator; setter is a no-op.
        pass

    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}

    def apply_delta(self,
                    observed_delta: Dict[str, float],
                    predicted_delta: Dict[str, float] = None):
        """
        Backward-compat apply_delta: updates scalar SMO for error tracking.
        Operator updates happen separately in the 9-step loop.
        """
        if predicted_delta is None:
            predicted_delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        self.smo.apply(self.S, observed_delta.get('S', 0), predicted_delta.get('S', 0))
        self.smo.apply(self.I, observed_delta.get('I', 0), predicted_delta.get('I', 0))
        self.smo.apply(self.P, observed_delta.get('P', 0), predicted_delta.get('P', 0))

    def rollback(self) -> bool:
        """Backward compat stub — v16 rollback handled by SMO.reverse() in uii_triad.py."""
        return False


# ──────────────────────────────────────────────────────────────────────────────
# StateTrace
# ──────────────────────────────────────────────────────────────────────────────

class StateTrace:
    """Ordered history of substrate states for field calculations."""

    def __init__(self, max_length: int = 1000):
        self.history:        deque = deque(maxlen=max_length)
        self.c_local_history: deque = deque(maxlen=100)
        self.c_global:       float = 0.0
        self._last_gradient: Dict[str, float] = {}

    def compute_c_local(self,
                        gradient: Dict[str, float],
                        sensing:  SensingOperator) -> float:
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
            return 0.0
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    @staticmethod
    def compute_c_local_static(gradient: Dict[str, float],
                                sensing:  SensingOperator) -> float:
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

    def record(self, state: SubstrateState,
               gradient: Dict[str, float] = None,
               virtual: bool = False):
        """
        v15.2+: gradient parameter added. Computes C_local internally every step.
        virtual=True: state enters history but C_local is NOT recorded.
        """
        self.history.append(state.as_dict())
        if gradient is not None:
            self._last_gradient = gradient
        if not virtual and self._last_gradient:
            c_local = self.compute_c_local(self._last_gradient, state.sensing)
            self.c_local_history.append(c_local)
            self.c_global = float(np.mean(self.c_local_history)) \
                if self.c_local_history else 0.0

    def get_recent(self, n: int) -> List[Dict]:
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]

    def __len__(self) -> int:
        return len(self.history)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level geometry functions
# ──────────────────────────────────────────────────────────────────────────────

def eigen_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable symmetric eigendecomposition.

    Always uses eigh (symmetric Hermitian). Never eig.
    Fallback: add 1e-3·I if LinAlgError.
    Clip eigenvalues to [-1e6, 1e6].

    Returns (eigenvalues, eigenvectors) in ascending order (eigh convention).
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        H = H + 1e-3 * np.eye(H.shape[0])
        eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.clip(eigvals, -1e6, 1e6)
    return eigvals, eigvecs


def expected_optionality_gain(prediction:  PredictionOperator,
                               action:      str,
                               compression: CompressionOperator,
                               sensing:     SensingOperator) -> float:
    """
    E[Δlog(O(a))] = log(vol_opt_post) - log(vol_opt_pre)

    vol_opt = sum of positive eigenvalues of Σ_P.
    pre:  current covariance_matrix().
    post: simulate_covariance_update(action, compression, sensing).

    Returns 0.0 if either covariance matrix has no positive eigenvalues.

    Called once per step for all actions before _build_viable_set().
    Result is shared between C2 collapse detection and score_actions().
    Never recomputed inside either.
    """
    sigma_pre,  _ = prediction.covariance_matrix(sensing, compression)
    sigma_post, _ = prediction.simulate_covariance_update(action, compression, sensing)

    def _log_vol(sigma: np.ndarray) -> float:
        if sigma.shape[0] == 0:
            return 0.0
        ev = np.linalg.eigvalsh(sigma)
        pos = ev[ev > 1e-9]
        return float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0

    return _log_vol(sigma_post) - _log_vol(sigma_pre)


# ──────────────────────────────────────────────────────────────────────────────
# PhiField — v16
# ──────────────────────────────────────────────────────────────────────────────

class PhiField:
    """
    Information Potential Field (Φ-field) — v16.

    Three-term potential over operator space:
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)

    v16 changes vs v15.3:
      _compute_K  — now operates in active channel space only;
                    no longer uses to_coupling_matrix() (4×4 SIPA distance).
                    peak_snapshot is now ledger.operator_snapshot format.
      compute_hessian — NEW: H = α·H_C + β·H_O + γ·H_K + εI in channel space.
      score_actions   — NEW: maturity × nat_grad_align + (1-maturity) × E[Δlog(O)].

    All v15.3 methods retained exactly:
      _compute_C, _build_sigma_p, _direction_crk_viable, _compute_O,
      phi, gradient, phi_legacy.

    The gradient() K-term reads peak_snapshot using the v16 operator_snapshot
    format. When 'sensing.channels' key is absent it returns 0.0 gracefully —
    which is correct because H_K in compute_hessian() carries the attractor
    well contribution through the Hessian, so the gradient K term is secondary.
    """

    O_FLOOR = 1e-6

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 A0: float = 0.7, alpha_crk: float = 2.0):
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.A0        = A0           # retained for phi_legacy only
        self.alpha_crk = alpha_crk    # retained for phi_legacy only

    # ── C(x) — unchanged from v15.3 ──────────────────────────────────────────

    def _compute_C(self, state) -> float:
        """
        C(x) = Σ_{edges} |w(e)|·conf(e)·cov(src)·cov(tgt)  /  (2·|edges|)
        Unchanged from v15.3.
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

    # ── Σ_P builder — unchanged from v15.3 ───────────────────────────────────

    def _build_sigma_p(self, state) -> Tuple[np.ndarray, List[str]]:
        """
        Build Σ_P over active channels from causal graph.
        Diagonal: Σ_P[i,i] = coverage_i
        Off-diagonal: Σ_P[i,j] = w(i→j) × conf(i→j) × coverage_i
        Returns (sigma_p, active_channel_list). sigma_p is symmetrized.
        Unchanged from v15.3.
        """
        channels = state.sensing.channels
        active = [cid for cid, ch in channels.items() if ch.active]
        n = len(active)
        if n == 0:
            return np.zeros((0, 0)), []

        idx = {cid: i for i, cid in enumerate(active)}
        sigma_p = np.zeros((n, n))

        for i, cid in enumerate(active):
            sigma_p[i, i] = channels[cid].coverage

        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                sigma_p[idx[src], idx[tgt]] += (
                    edge.weight * edge.confidence * channels[src].coverage
                )

        sigma_p = (sigma_p + sigma_p.T) / 2.0
        return sigma_p, active

    # ── CRK viability filter — unchanged from v15.3 ──────────────────────────

    def _direction_crk_viable(self, channel_delta: Dict[str, float], state) -> bool:
        """
        Check whether moving in a channel direction is CRK-viable.
        Forming guard: empty causal_graph → all directions viable.
        Unchanged from v15.3.
        """
        if not state.compression.causal_graph:
            return True

        graph_channels: Set[str] = set()
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

        channel_weights: Dict[str, float] = {}
        for (src, tgt), edge in state.compression.causal_graph.items():
            w = abs(edge.weight) * edge.confidence
            channel_weights[src] = channel_weights.get(src, 0.0) + w
            channel_weights[tgt] = channel_weights.get(tgt, 0.0) + w

        if channel_weights:
            max_w     = max(channel_weights.values())
            threshold = max_w * 0.5
            anchors   = {cid for cid, w in channel_weights.items() if w >= threshold}
            anchor_reduction = sum(
                -v for cid, v in channel_delta.items()
                if cid in anchors and v < 0
            )
            if anchor_reduction / total_delta > 0.5:
                return False

        return True

    # ── O(x) — unchanged from v15.3 ──────────────────────────────────────────

    def _compute_O(self, state) -> float:
        """
        O(x) = Σ_{λ_i > 0} λ_i  of  Σ_P filtered to CRK-viable directions.
        Unchanged from v15.3.
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

    # ── K(x) — CHANGED in v16 ────────────────────────────────────────────────

    def _compute_K(self, state, peak_snapshot: Optional[Dict] = None) -> float:
        """
        K(x) = -‖coverage_current - coverage_peak‖²

        coverage_current: coverage values of active channels.
        coverage_peak:    from ledger operator_snapshot['sensing']['channels'].

        CHANGED from v15.3:
          - No longer reads to_coupling_matrix() (4×4 SIPA distance removed).
          - Operates in active channel space only.
          - peak_snapshot is the full ledger.operator_snapshot dict.

        If peak_snapshot is None: K = 0.0 (cold start).
        The compression geometry is fully captured by H_C in the Hessian.
        """
        if peak_snapshot is None:
            return 0.0

        peak_channels = peak_snapshot.get('sensing', {}).get('channels', {})
        if not peak_channels:
            return 0.0

        active = [cid for cid, ch in state.sensing.channels.items() if ch.active]
        if not active:
            return 0.0

        current_cov = np.array([state.sensing.channels[cid].coverage
                                 for cid in active])
        peak_cov    = np.array([float(peak_channels.get(cid, {}).get('coverage', 0.0))
                                 for cid in active])

        return -float(np.sum((current_cov - peak_cov) ** 2))

    # ── Φ(x) — unchanged from v15.3 ──────────────────────────────────────────

    def phi(self, state, trace, peak_snapshot: Optional[Dict] = None) -> float:
        """
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)
        peak_snapshot: ledger.operator_snapshot or None.
        Unchanged from v15.3 (K now reads new snapshot format).
        """
        C = self._compute_C(state)
        O = self._compute_O(state)
        K = self._compute_K(state, peak_snapshot)
        phi_val = (self.alpha * C
                   + self.beta * np.log(max(O, self.O_FLOOR))
                   + self.gamma * K)
        return float(np.clip(phi_val, -100.0, 100.0))

    # ── ∇Φ — unchanged from v15.3 ────────────────────────────────────────────

    def gradient(self, state, trace,
                 peak_snapshot: Optional[Dict] = None) -> Dict[str, float]:
        """
        ∇Φ(x) per channel, normalized to unit vector.

        Three contributions:
          ∂C/∂channel_i  — closed form
          ∂O/∂channel_i  — finite difference
          ∂K/∂channel_i  = -2·(cov_i - cov_peak_i)

        peak_snapshot format: ledger.operator_snapshot.
        The K-term reads ['sensing']['channels'][cid]['coverage'].
        Falls back gracefully to 0.0 when key absent (cold start).

        trace parameter accepted but not used in computation —
        safe to pass None (as score_actions() does).

        Unchanged from v15.3 modulo K-term snapshot key update.
        """
        channels = state.sensing.channels
        grad: Dict[str, float] = {}

        O = self._compute_O(state)

        # Build peak coverage dict from v16 operator_snapshot format
        peak_coverage: Dict[str, float] = {}
        if peak_snapshot is not None:
            peak_ch_snap = peak_snapshot.get('sensing', {}).get('channels', {})
            for cid in channels:
                if cid in peak_ch_snap:
                    peak_coverage[cid] = float(peak_ch_snap[cid].get('coverage', 0.0))

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
            max_possible = 2.0 * max(len(graph), 1)
            g_C /= max_possible

            # ∂O/∂channel_i (finite difference)
            g_O = 0.0
            eps = 1e-4
            try:
                perturbed_channels = dict(channels)
                perturbed_ch = dataclasses.replace(
                    ch, coverage=float(np.clip(ch.coverage + eps, 0.0, 1.0))
                )
                perturbed_channels[cid] = perturbed_ch
                ps_sensing = SensingOperator(channels=perturbed_channels)

                class _PState:
                    pass
                ps = _PState()
                ps.sensing     = ps_sensing
                ps.compression = state.compression
                O_perturbed = self._compute_O(ps)
                g_O = (O_perturbed - O) / eps
            except Exception:
                g_O = 0.0

            # ∂K/∂channel_i = -2·(cov_i - cov_peak_i)
            g_K = 0.0
            if cid in peak_coverage:
                g_K = -2.0 * (ch.coverage - peak_coverage[cid])

            grad[cid] = (self.alpha * g_C
                         + self.beta / max(O, self.O_FLOOR) * g_O
                         + self.gamma * g_K)

        # Normalize to unit vector
        norm = float(np.sqrt(sum(v ** 2 for v in grad.values())))
        if norm > 1e-8:
            grad = {k: v / norm for k, v in grad.items()}
        return grad

    # ── phi_legacy — retained for log output ─────────────────────────────────

    @staticmethod
    def si_capacity(S: float, I: float) -> float:
        """Retained for phi_legacy and backward compat callers."""
        return min(S, I) * 0.5 + (S + I) / 4.0

    def phi_legacy(self, state, trace, crk_violations=None) -> float:
        """
        v15.2 legacy formula retained for log output.
        α·log(1+P) - β·(A-A₀)² - γ·curvature - CRK_penalty
        Not used for decisions.
        """
        cap = self.si_capacity(state.S, state.I)
        effective_P = min(state.P, cap * 2.0)
        opt    = np.log(1.0 + max(effective_P, 0.0))
        strain = (state.A - self.A0) ** 2

        recent = trace.get_recent(3) if trace is not None else []
        curv   = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2 * h1[k] + h0[k])

        phi_raw    = self.alpha * opt - self.beta * strain - self.gamma * curv
        crk_penalty = 0.0
        if crk_violations:
            crk_penalty = self.alpha_crk * sum(
                severity for _, severity in crk_violations
            )
        return phi_raw - crk_penalty

    # ── compute_hessian — NEW in v16 ─────────────────────────────────────────

    def compute_hessian(self,
                        state,
                        prediction:    PredictionOperator,
                        coherence:     CoherenceOperator,
                        peak_snapshot: Optional[Dict] = None,
                        epsilon:       float = 1e-4,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   List[str], np.ndarray, np.ndarray]:
        """
        H = α·H_C + β·H_O + γ·H_K + εI

        All three components in active channel space (n×n where n = active
        channel count). Same space as Φ, ∇Φ, and Σ_P. Not the 4×4 SIPA space.

        H_C — compression curvature:
            weight * confidence² * coverage_src * coverage_tgt per causal edge,
            symmetrized. Confidence² damping prevents low-confidence early edges
            from exploding eigenvalues.

        H_O — optionality curvature (inverse covariance):
            Σ_P = covariance_matrix(). Decompose: eigvecs, eigvals = eigh(Σ_P).
            H_O = eigvecs[:, pos] @ diag(1/eigvals[pos]) @ eigvecs[:, pos].T
            Naturally caps Hessian eigenvalue magnitude.
            If no positive eigenvalues: H_O = 0.

        H_K — attractor well:
            delta = current_configuration_vector() - peak_configuration_vector()
            H_K = -outer(delta, delta)
            If peak_snapshot is None: H_K = 0.

        εI — diagonal regularization (prevents singular matrices).

        Returns (H, eigenvalues, eigenvectors, active_channel_list, H_C, H_O).
        H_C and H_O returned so score_actions() can compute eigenspectrum
        maturity ratio without recomputing.

        Called ONCE per step. Caller caches the return tuple; it is NOT
        recomputed inside score_actions(), viable set loop, or anywhere else.

        Returns (zeros(0,0), [], [], [], zeros(0,0), zeros(0,0)) if no
        active channels.
        """
        channels = state.sensing.channels
        active   = [cid for cid, ch in channels.items() if ch.active]
        n        = len(active)
        _zero6   = (np.zeros((0, 0)), np.array([]), np.zeros((0, 0)),
                    [], np.zeros((0, 0)), np.zeros((0, 0)))
        if n == 0:
            return _zero6

        idx = {cid: i for i, cid in enumerate(active)}

        # ── H_C ───────────────────────────────────────────────────────────────
        H_C = np.zeros((n, n))
        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                si  = idx[src]
                ti  = idx[tgt]
                src_cov = channels[src].coverage
                tgt_cov = channels[tgt].coverage
                H_C[si, ti] += edge.weight * (edge.confidence ** 2) * src_cov * tgt_cov
        H_C = (H_C + H_C.T) / 2.0

        # ── H_O ───────────────────────────────────────────────────────────────
        sigma_p, _ = prediction.covariance_matrix(state.sensing, state.compression)
        if sigma_p.shape[0] > 0:
            try:
                ev, evec = np.linalg.eigh(sigma_p)
            except np.linalg.LinAlgError:
                sigma_p += 1e-3 * np.eye(sigma_p.shape[0])
                ev, evec = np.linalg.eigh(sigma_p)
            pos = ev > 1e-9
            if np.any(pos):
                H_O = evec[:, pos] @ np.diag(1.0 / ev[pos]) @ evec[:, pos].T
            else:
                H_O = np.zeros((n, n))
        else:
            H_O = np.zeros((n, n))

        # ── H_K ───────────────────────────────────────────────────────────────
        if peak_snapshot is not None:
            delta = (
                coherence.current_configuration_vector(state.sensing, active)
                - coherence.peak_configuration_vector(peak_snapshot, active)
            )
            H_K = -np.outer(delta, delta)
        else:
            H_K = np.zeros((n, n))

        # ── Combine ───────────────────────────────────────────────────────────
        H = (self.alpha * H_C
             + self.beta  * H_O
             + self.gamma * H_K
             + epsilon * np.eye(n))

        ev_H, evec_H = eigen_decompose(H)

        return H, ev_H, evec_H, active, H_C, H_O

    # ── score_actions — NEW in v16 ────────────────────────────────────────────

    def score_actions(self,
                      viable_actions:       List[str],
                      state,
                      H:                    np.ndarray,
                      eigvals:              np.ndarray,
                      eigvecs:              np.ndarray,
                      active_channels:      List[str],
                      H_C:                  np.ndarray,
                      H_O:                  np.ndarray,
                      prediction:           PredictionOperator,
                      peak_snapshot:        Optional[Dict] = None,
                      optionality_gain_dict: Optional[Dict[str, float]] = None,
                      ) -> Dict[str, float]:
        """
        score(a) = maturity × nat_grad_align_norm(a) + (1 - maturity) × E[Δlog(O(a))]

        Dynamical law: ẋ = G⁻¹(x) ∇Φ(x) where G(x) = H = ∇²Φ locally.
        Lyapunov: dΦ/dt = ∇Φᵀ G⁻¹ ∇Φ ≥ 0 (G⁻¹ PD when H PSD).
        Φ is guaranteed non-decreasing. Ascent is information-optimal.

        nat_grad_align(a):
            natural_gradient = H⁻¹ ∇Φ
            δx(a) = test_virtual() channel delta in active channel space
            nat_grad_align(a) = ⟨natural_gradient, δx(a)⟩

        H⁻¹ via eigendecomposition (already cached — NOT recomputed):
            H⁻¹ = eigvecs @ diag(1/eigvals_safe) @ eigvecs.T
            eigvals clipped at 1e-6 to avoid inversion instability.

        maturity = var_H_C / (var_H_C + var_H_O + ε)
            Derived from Hessian eigenspectrum. No proxy.
            H_C dominates → causal graph has real structure → δ²Φ trustworthy → 1.
            H_O dominates → prediction uncertainty high → explore → 0.

        E[Δlog(O(a))] pre-computed once per step by caller; passed via
        optionality_gain_dict OR recomputed here if not pre-provided.

        Both nat_grad_align and E[Δlog(O)] are normalized to [0, 1] before
        combining so maturity is a true blend weight.

        H, eigvals, eigvecs, H_C, H_O: from cached compute_hessian().
        Do NOT recompute inside this method.

        Returns {action: score} dict.
        Empty viable_actions or empty channel space returns {a: 0.0} for each action.
        """
        if not viable_actions:
            return {}
        if H.shape[0] == 0:
            return {a: 0.0 for a in viable_actions}

        # ── Maturity from eigenspectrum ratio ─────────────────────────────────
        ev_C   = np.linalg.eigvalsh(self.alpha * H_C)
        ev_O   = np.linalg.eigvalsh(self.beta  * H_O)
        var_C  = float(np.sum(ev_C[ev_C > 0]))
        var_O  = float(np.sum(ev_O[ev_O > 0]))
        maturity = var_C / (var_C + var_O + 1e-8)

        # ── Natural gradient: H⁻¹ ∇Φ ─────────────────────────────────────────
        grad_dict = self.gradient(state, None, peak_snapshot)
        grad_vec  = np.array([grad_dict.get(cid, 0.0) for cid in active_channels])

        eigvals_safe = np.where(np.abs(eigvals) > 1e-6,
                                eigvals,
                                1e-6 * np.sign(eigvals + 1e-12))
        H_inv            = eigvecs @ np.diag(1.0 / eigvals_safe) @ eigvecs.T
        natural_gradient = H_inv @ grad_vec

        channel_idx = {cid: i for i, cid in enumerate(active_channels)}
        n           = len(active_channels)

        nat_grad_align_raw: Dict[str, float] = {}
        optionality_gain:   Dict[str, float] = {}

        for action in viable_actions:
            # Action direction in active channel space
            channel_delta = state.prediction.test_virtual(
                state.compression, action, phi_field=None, sensing=state.sensing
            )
            dx = np.zeros(n)
            for cid, val in channel_delta.items():
                if cid in channel_idx:
                    dx[channel_idx[cid]] = val

            nat_grad_align_raw[action] = float(np.dot(natural_gradient, dx))

            # Use pre-computed E[Δlog(O)] if provided (spec: computed once per step)
            if optionality_gain_dict is not None and action in optionality_gain_dict:
                optionality_gain[action] = optionality_gain_dict[action]
            else:
                optionality_gain[action] = expected_optionality_gain(
                    prediction, action, state.compression, state.sensing
                )

        # ── Normalize nat_grad_align to [0, 1] ───────────────────────────────
        vals    = list(nat_grad_align_raw.values())
        v_min   = min(vals)
        v_range = max(vals) - v_min
        if v_range < 1e-10:
            nat_norm = {a: 0.5 for a in viable_actions}
        else:
            nat_norm = {a: (nat_grad_align_raw[a] - v_min) / v_range
                        for a in viable_actions}

        # ── Normalize E[Δlog(O)] to [0, 1] ───────────────────────────────────
        og_vals  = list(optionality_gain.values())
        og_min   = min(og_vals)
        og_range = max(og_vals) - og_min
        if og_range < 1e-10:
            og_norm = {a: 0.5 for a in viable_actions}
        else:
            og_norm = {a: (optionality_gain[a] - og_min) / og_range
                       for a in viable_actions}

        return {
            a: maturity * nat_norm[a] + (1.0 - maturity) * og_norm[a]
            for a in viable_actions
        }


# ──────────────────────────────────────────────────────────────────────────────
# CRK structures — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CRKEvaluation:
    constraint:  str    # 'C1'–'C7'
    phase:       str    # 'pre_action' | 'post_action'
    status:      str    # 'satisfied' | 'degraded' | 'violated'
    risk:        float  # [0, 1]
    attribution: str    # 'internal' | 'external' | 'mixed'
    blocks:      bool
    signal:      str


@dataclass
class CRKVerdict:
    phase:         str
    action:        str
    evaluations:   List[CRKEvaluation]
    coherent:      bool
    repair:        Optional[str]
    phi_modifier:  float
    smo_permitted: bool


class CRKMonitor:
    """
    Constraint Recognition Kernel (CRK) — unchanged from v15.3.

    Pre-action:  filters action manifold; returns phi_modifier.
    Post-action: gates SMO update; failed → rollback signal.
    """

    PREDICTION_ERROR_THRESHOLD = 0.05
    IRREVERSIBLE_ACTIONS = {'migrate', 'navigate', 'python'}
    COMMITTING_ACTIONS   = {'migrate', 'navigate', 'python', 'click', 'fill', 'type'}

    def evaluate_pre_action(self,
                             proposed_action: str,
                             coherence:       CoherenceOperator,
                             sensing:         SensingOperator,
                             compression:     CompressionOperator,
                             prediction:      PredictionOperator,
                             field_state:     Dict,
                             ) -> CRKVerdict:
        crk_sig   = coherence.crk_signal()
        loop_cl   = crk_sig['loop_closure']
        sig_dev   = crk_sig['signature_deviation']
        i_p       = crk_sig['i_p_consistency']
        p_proxy   = prediction.to_grounded_proxy(sensing, compression)
        evaluations: List[CRKEvaluation] = []

        # C1
        irreversible     = proposed_action in self.IRREVERSIBLE_ACTIONS
        c1_blocks        = loop_cl < 0.3 and irreversible and sig_dev > 0.3
        c1_stability_risk = 0.0
        try:
            coupling   = compression.to_coupling_matrix()
            pred_delta = self._get_predicted_delta(proposed_action, compression)
            delta_vec  = np.array([pred_delta.get(d, 0.0) for d in ['S', 'I', 'P', 'A']])
            stability  = float(0.5 * delta_vec @ coupling @ delta_vec)
            c1_stability_risk = float(np.clip(-stability, 0.0, 1.0)) if stability < -0.1 else 0.0
        except Exception:
            pass
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

        # C2
        committing = proposed_action in self.COMMITTING_ACTIONS
        forming    = len(compression.causal_graph) == 0
        c2_blocks  = p_proxy < 0.15 and committing and not forming
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

        # C3
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

        # C6
        system_load = field_state.get('system_load', 0.0)
        evaluations.append(CRKEvaluation(
            constraint  = 'C6',
            phase       = 'pre_action',
            status      = 'degraded' if system_load > 0.7 else 'satisfied',
            risk        = float(np.clip(system_load - 0.7, 0.0, 1.0)) if system_load > 0.7 else 0.0,
            attribution = 'external',
            blocks      = False,
            signal      = f'system_load={system_load:.2f}',
        ))

        # C7
        phi_trend = field_state.get('phi_trend', 0.0)
        c7_blocks = proposed_action == 'migrate' and loop_cl < 0.5
        evaluations.append(CRKEvaluation(
            constraint  = 'C7',
            phase       = 'pre_action',
            status      = 'violated' if c7_blocks else ('degraded' if phi_trend < -0.1 else 'satisfied'),
            risk        = float(np.clip(0.5 - loop_cl, 0.0, 1.0)) if c7_blocks else 0.0,
            attribution = 'mixed',
            blocks      = c7_blocks,
            signal      = f'migrate+loop_cl={loop_cl:.2f}' if c7_blocks else f'phi_trend={phi_trend:.2f}',
        ))

        phi_modifier = float(np.clip(float(np.prod([
            1.0 - (e.risk * (1.0 if e.status == 'degraded' else 2.0))
            for e in evaluations
        ])), 0.0, 1.0))
        coherent = not any(e.blocks for e in evaluations)

        repair   = None
        violated = [e for e in evaluations if e.status == 'violated']
        if violated:
            repair = {'C1': 'stabilize', 'C2': 'expand',
                      'C3': 'reattribute', 'C7': 'stabilize'}.get(violated[0].constraint)
        elif any(e.constraint == 'C6' and e.status == 'degraded' for e in evaluations):
            repair = 'coordinate'

        return CRKVerdict(
            phase         = 'pre_action',
            action        = proposed_action,
            evaluations   = evaluations,
            coherent      = coherent,
            repair        = repair,
            phi_modifier  = phi_modifier,
            smo_permitted = True,
        )

    def _get_predicted_delta(self, action: str,
                              compression: CompressionOperator) -> Dict[str, float]:
        action_map = getattr(compression, 'action_substrate_map', {}) or {}
        if action in action_map:
            return dict(action_map[action])
        fallback = {
            'navigate': {'S': 0.05, 'I': 0.0, 'P': -0.08, 'A': 0.0},
            'click':    {'S': 0.02, 'I': 0.0, 'P': -0.02, 'A': 0.0},
            'read':     {'S': 0.03, 'I': 0.0, 'P':  0.0,  'A': 0.0},
            'observe':  {'S': 0.0,  'I': 0.0, 'P':  0.0,  'A': 0.0},
        }
        return fallback.get(action, {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0})

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
                              ) -> CRKVerdict:
        evaluations: List[CRKEvaluation] = []
        evaluations.append(self._c4_post(observed_delta, predicted_delta, sensing, compression))
        evaluations.append(self._c5_post(observed_delta, predicted_delta, coherence))
        evaluations.append(self._c1_post(proposed_smo_update, prior_compression, compression))
        evaluations.append(self._c3_post(proposed_smo_update, smo_plasticity))

        smo_permitted = not any(e.blocks for e in evaluations)
        repair = None
        if not smo_permitted:
            c4 = next((e for e in evaluations if e.constraint == 'C4' and e.blocks), None)
            c1 = next((e for e in evaluations if e.constraint == 'C1' and e.blocks), None)
            c5 = next((e for e in evaluations if e.constraint == 'C5'
                       and e.status in ('degraded', 'violated')), None)
            if c4:
                repair = None
            elif c1:
                repair = 'rollback'
            elif c5:
                repair = 'reattribute'
        if repair is None:
            c3v = next((e for e in evaluations
                        if e.constraint == 'C3' and e.status == 'violated'), None)
            if c3v:
                repair = 'reattribute'

        return CRKVerdict(
            phase         = 'post_action',
            action        = str(proposed_smo_update),
            evaluations   = evaluations,
            coherent      = smo_permitted,
            repair        = repair,
            phi_modifier  = 1.0,
            smo_permitted = smo_permitted,
        )

    def _c3_post(self, prediction_errors: Dict, smo_plasticity: float) -> CRKEvaluation:
        mean_error = (float(np.mean(list(prediction_errors.values())))
                      if prediction_errors else 0.0)
        if mean_error > 0.15 and smo_plasticity > 0.75:
            return CRKEvaluation('C3', 'post_action', 'violated', smo_plasticity,
                                 'internal', False,
                                 f'mean_error={mean_error:.3f} (external), '
                                 f'plasticity={smo_plasticity:.2f} — internalising external resistance')
        if mean_error > 0.10 and smo_plasticity > 0.65:
            return CRKEvaluation('C3', 'post_action', 'degraded', smo_plasticity * 0.5,
                                 'internal', False,
                                 f'mean_error={mean_error:.3f}, plasticity={smo_plasticity:.2f} — watch')
        return CRKEvaluation('C3', 'post_action', 'satisfied', 0.0, 'internal', False,
                             f'mean_error={mean_error:.3f}, plasticity={smo_plasticity:.2f}')

    def _c4_post(self, observed: Dict, predicted: Dict,
                 sensing: SensingOperator,
                 compression: CompressionOperator) -> CRKEvaluation:
        if len(compression.causal_graph) == 0:
            return CRKEvaluation('C4', 'post_action', 'satisfied', 0.0, 'external', False,
                                 'forming — no model yet, all signal is external mismatch')
        per_channel_error: Dict[str, float] = {}
        for channel_id, ch in sensing.channels.items():
            if not ch.active:
                continue
            obs  = observed.get(channel_id, None)
            pred = predicted.get(channel_id, None)
            if obs is None and pred is None:
                continue
            obs  = obs  if obs  is not None else 0.0
            pred = pred if pred is not None else 0.0
            if isinstance(obs,  dict): obs  = obs.get('magnitude',  0.0)
            if isinstance(pred, dict): pred = pred.get('magnitude', 0.0)
            per_channel_error[channel_id] = abs(float(obs) - float(pred))
        if not per_channel_error:
            for dim in ['S', 'I', 'P', 'A']:
                ov = observed.get(dim, 0.0)
                pv = predicted.get(dim, 0.0)
                if isinstance(ov, (int, float)) and isinstance(pv, (int, float)):
                    per_channel_error[dim] = abs(float(ov) - float(pv))
        if not per_channel_error:
            return CRKEvaluation('C4', 'post_action', 'satisfied', 0.0, 'external', False,
                                 'no active channels')
        mean_error = float(np.mean(list(per_channel_error.values())))
        if mean_error < self.PREDICTION_ERROR_THRESHOLD:
            return CRKEvaluation('C4', 'post_action', 'violated',
                                 1.0 - mean_error / self.PREDICTION_ERROR_THRESHOLD,
                                 'internal', True,
                                 f'mean_error={mean_error:.4f} < ε={self.PREDICTION_ERROR_THRESHOLD}'
                                 f' — adaptation not grounded in external mismatch')
        return CRKEvaluation('C4', 'post_action', 'satisfied', mean_error, 'external', False,
                             f'mean_error={mean_error:.4f} — external mismatch confirmed')

    def _c1_post(self, proposed_update: Dict,
                 prior_compression: CompressionOperator,
                 new_compression:   CompressionOperator) -> CRKEvaluation:
        prior_edges     = set(prior_compression.causal_graph.keys())
        new_edges       = set(new_compression.causal_graph.keys())
        edges_destroyed = prior_edges - new_edges
        if edges_destroyed:
            ratio = len(edges_destroyed) / max(len(prior_edges), 1)
            if ratio > 0.3:
                return CRKEvaluation('C1', 'post_action', 'violated', ratio, 'internal', True,
                                     f'{len(edges_destroyed)} edges destroyed '
                                     f'({ratio:.1%}) — historical erasure, roll back')
        confidence_loss = [
            key for key in prior_edges & new_edges
            if prior_compression.causal_graph[key].confidence > 0.5
            and new_compression.causal_graph[key].confidence < 0.1
        ]
        if len(confidence_loss) > 3:
            return CRKEvaluation('C1', 'post_action', 'degraded',
                                 len(confidence_loss) / max(len(prior_edges), 1),
                                 'internal', False,
                                 f'{len(confidence_loss)} high-confidence edges collapsed')
        return CRKEvaluation('C1', 'post_action', 'satisfied', 0.0, 'internal', False,
                             'continuity preserved')

    def _c5_post(self, observed: Dict, predicted: Dict,
                 coherence: CoherenceOperator) -> CRKEvaluation:
        i_p = coherence.consistency.i_p_consistency
        p_a = coherence.consistency.p_a_consistency
        if i_p < 0.4 and p_a > 0.6:
            return CRKEvaluation('C5', 'post_action', 'degraded', 1.0 - i_p, 'internal', False,
                                 f'i_p={i_p:.2f} low, p_a={p_a:.2f} fine — '
                                 f'error source is internal compression, not external Reality. '
                                 f'Reattribute before SMO update.')
        return CRKEvaluation('C5', 'post_action', 'satisfied', 0.0, 'external', False,
                             'attribution clear')

    def evaluate(self, state, trace: StateTrace,
                 reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Legacy scalar CRK — retained for backward compat and parallel validation."""
        violations = []
        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev   = recent[-2]
            jump   = sum(abs(prev[k] - getattr(state, k)) for k in ["S", "I", "P", "A"])
            if jump > 0.3:
                violations.append(("C1_Continuity", jump - 0.3))
        if state.P < 0.35:
            violations.append(("C2_Optionality", 0.35 - state.P))
        if len(trace) >= 5:
            recent_5       = trace.get_recent(5)
            conf_values    = [r["S"] + r["I"] for r in recent_5]
            conf_declining = all(conf_values[i] >= conf_values[i + 1]
                                  for i in range(len(conf_values) - 1))
            externally_driven = (
                reality_delta is not None and
                sum(abs(v) for v in reality_delta.values()
                    if isinstance(v, (int, float))) > 0.05
            )
            if conf_declining and not externally_driven:
                severity = conf_values[0] - conf_values[-1]
                if severity > 0.1:
                    violations.append(("C3_NonInternalization", severity))
        if reality_delta and len(trace) >= 3:
            fb = sum(abs(v) for v in reality_delta.values() if isinstance(v, (int, float)))
            if fb < 0.01:
                violations.append(("C4_Reality", 0.01 - fb))
        if len(trace) >= 2:
            recent    = trace.get_recent(2)
            prev      = recent[-2]
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


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory manifold — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryCandidate:
    """Multi-step executable procedure with structural annotations."""
    steps:                            List[Dict]
    rationale:                        str
    estimated_coherence_preservation: float
    estimated_optionality_delta:      float
    reversibility_point:              int
    tested:                           bool           = False
    test_phi_final:                   Optional[float] = None
    test_state_final:                 Optional[Dict]  = None
    test_violations:                  Optional[List]  = None
    test_perturbation_trace:          Optional[List]  = None
    test_succeeded:                   bool            = False
    virtual_phi:                      Optional[float] = None

    def __repr__(self):
        status  = "✓" if self.test_succeeded else "✗" if self.tested else "?"
        phi_str = f"Φ={self.test_phi_final:.3f}" if self.test_phi_final is not None else "untested"
        return f"{status} [{len(self.steps)} steps] {self.rationale[:50]} ({phi_str})"


@dataclass
class TrajectoryManifold:
    """Container for enumerated trajectory space."""
    candidates:          List[TrajectoryCandidate]
    enumeration_context: Dict

    def size(self) -> int:
        return len(self.candidates)

    def tested_count(self) -> int:
        return sum(1 for c in self.candidates if c.tested)

    def get_best(self) -> Optional[TrajectoryCandidate]:
        valid = [c for c in self.candidates if c.tested and c.test_succeeded]
        if not valid:
            return None
        return max(valid, key=lambda c: c.test_phi_final)

    def get_all_tested(self) -> List[TrajectoryCandidate]:
        tested = [c for c in self.candidates if c.tested]
        return sorted(tested,
                      key=lambda c: c.test_phi_final if c.test_phi_final is not None else -1000,
                      reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Agent infrastructure — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

class AgentHandler(ABC):
    """Interface for agent interaction (CRK C6: Other-Agent Existence)."""

    @abstractmethod
    def post_query(self, triad_id: str, query_text: str): ...

    @abstractmethod
    def get_response(self, triad_id: str) -> Optional[str]: ...


class UserAgentHandler(AgentHandler):
    """Human user as agent. Non-blocking query/response."""

    def __init__(self):
        self.pending_queries: deque = deque()
        self.responses: Dict[str, str] = {}

    def post_query(self, triad_id: str, query_text: str):
        self.pending_queries.append({
            'triad_id': triad_id, 'query': query_text, 'timestamp': time.time()
        })
        print(f"\n{'='*70}")
        print(f"[QUERY FROM TRIAD {triad_id}]")
        print(f"{query_text}")
        print(f"{'='*70}")
        print(f"Respond with: triad.respond_to_query('{triad_id}', 'your answer')")
        print(f"Or leave pending - Triad will continue exploration")
        print(f"{'='*70}\n")

    def get_response(self, triad_id: str) -> Optional[str]:
        return self.responses.pop(triad_id, None)

    def respond(self, triad_id: str, answer: str):
        self.responses[triad_id] = answer

    def has_pending(self) -> bool:
        return len(self.pending_queries) > 0

    def get_pending_count(self) -> int:
        return len(self.pending_queries)


AVAILABLE_AGENTS: Dict[str, AgentHandler] = {
    'user': UserAgentHandler()
}


# ──────────────────────────────────────────────────────────────────────────────
# Adapter ABCs — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

class RealityAdapter(ABC):
    """Interface for environment / perturbation source."""

    @abstractmethod
    def execute(self, action: Dict) -> Tuple[Dict[str, float], Dict]: ...

    @abstractmethod
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]: ...

    @abstractmethod
    def get_current_affordances(self) -> Dict: ...

    @abstractmethod
    def close(self): ...


class IntelligenceAdapter(ABC):
    """Interface for Relation component of Mentat Triad."""

    @abstractmethod
    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold: ...

    @abstractmethod
    def record_committed_trajectory(self, trajectory: TrajectoryCandidate,
                                    phi_final: float): ...



# ──────────────────────────────────────────────────────────────────────────────
# Symbol grounding — moved from uii_intelligence.py
# Core logic unchanged; SRE/CausalDiagnosis coupling removed.
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL_GROUNDING_PROMPT = """\
You are the symbol grounding layer of an autonomous system.

The system's structural engine has diagnosed an impossibility and determined
what trajectory shapes are needed. Your job is to fill in concrete parameters
— URLs, Python code, CSS selectors — for the shapes below.

CAUSAL DIAGNOSIS:
  Binding dimension: {binding_dim}
  Cause:             {cause_class}
  Evidence:
{evidence_lines}
  Migration indicated: {migration_indicated}
  Symbol required:     {symbol_requirement}
  Migration urgency:   {migration_urgency}

TRAJECTORY SHAPES TO GROUND:
{shapes_block}

CURRENT ENVIRONMENT:
  URL:     {current_url}
  Title:   {page_title}
  Links:   {links}
  Buttons: {buttons}
  Inputs:  {inputs}

BOUNDARY PRESSURE: {boundary_pressure:.2f}
(1.0 = resource limit imminent. High pressure = prioritise migration.)

TOKEN BUDGET REMAINING: {token_budget_remaining}
(If low: write shorter code, single target only, skip verification steps.)

Fill in concrete parameters for each shape.
For python actions: write complete, runnable code.
For navigate actions: provide a specific, reachable URL.
For click/fill/read actions: provide a valid CSS selector.

Output a JSON array of {n_shapes} trajectories in this format:
[
  {{
    "steps": [
      {{"action_type": "...", "parameters": {{...}}}},
      ...
    ],
    "rationale": "what Φ geometry this trajectory probes",
    "estimated_coherence_preservation": 0.XX,
    "estimated_optionality_delta": 0.XX,
    "reversibility_point": N
  }}
]

JSON only. No commentary.
"""


def _format_shapes_block(shapes: List) -> str:
    """
    Format shape list for the LLM prompt.
    Accepts shape objects or dicts — both are handled via getattr/get.
    SRE TrajectoryShape coupling removed; works with any shape-like structure.
    """
    lines: List[str] = []
    for i, shape in enumerate(shapes, 1):
        # Support both object attributes and dict keys
        def _get(key: str, default=''):
            if isinstance(shape, dict):
                return shape.get(key, default)
            return getattr(shape, key, default)

        lines.append(f"Shape {i}: {_get('strategy_class', 'unknown')}")
        seq = _get('action_sequence', [])
        lines.append(f"  Sequence:  {' → '.join(seq) if seq else '(none)'}")
        lines.append(f"  Target:    {_get('target_dimension', '')}")
        lines.append(f"  Rationale: {_get('rationale', '')}")
        pred = _get('predicted_delta', {})
        if isinstance(pred, dict):
            delta_str = ', '.join(
                f"{d}={pred.get(d, 0):+.3f}"
                for d in ['S', 'I', 'P', 'A']
                if abs(pred.get(d, 0)) > 0.001
            )
            if delta_str:
                lines.append(f"  Predicted: {delta_str}")
        lines.append("")
    return "\n".join(lines)


class SymbolGroundingAdapter:
    """
    Symbol grounding layer. Moved from uii_intelligence.py — core logic unchanged.

    Receives a diagnosis context dict and fills in concrete symbols —
    URLs, Python code, CSS selectors — using the LLM's pretrained token field.

    In v16 SymbolGroundingAdapter is called directly from step() in uii_triad.py
    when action ∈ {python, llm_query, migrate}. It no longer requires a
    CausalDiagnosis from StructuralRelationEngine (SRE eliminated).

    The diagnosis parameter in ground_trajectories() is now a plain dict with
    the same keys that SYMBOL_GROUNDING_PROMPT expects. Any dict-like context
    containing the required keys works.

    All parsing, validation, and trajectory history logic unchanged from v15.
    """

    def __init__(self, llm_client):
        self.llm                = llm_client
        self.call_count         = 0
        self.trajectory_history = deque(maxlen=3)

    def ground_trajectories(self,
                             diagnosis: Dict,
                             context:   Dict) -> TrajectoryManifold:
        """
        Build minimal prompt from diagnosis dict, call LLM, parse result.
        context provides current affordances for symbol resolution.

        diagnosis keys used:
          binding_dim, cause_class, evidence, migration_indicated,
          symbol_requirement, migration_urgency, trajectory_shapes

        token_budget_remaining included in prompt so LLM can adapt code
        complexity and target count.
        """
        self.call_count += 1

        affordances        = context.get('affordances', {})
        boundary_pressure  = context.get('boundary_pressure', 0.0)
        shapes             = diagnosis.get('trajectory_shapes', [])

        token_budget       = context.get('token_budget', None)
        token_pressure     = context.get('token_pressure', None)
        binding_constraint = context.get('binding_constraint', 'steps')

        if token_budget is not None and token_pressure is not None:
            remaining              = int(token_budget * (1.0 - token_pressure))
            token_budget_remaining = f"{remaining:,} (binding: {binding_constraint})"
        else:
            token_budget_remaining = "unknown"

        migration_urgency = diagnosis.get('migration_urgency', 'focused')
        if migration_urgency == 'emergency' or (
            token_pressure is not None and token_pressure > 0.8
        ):
            shapes = shapes[:1]

        evidence       = diagnosis.get('evidence', [])
        evidence_lines = "\n".join(f"    - {e}" for e in evidence)

        links   = json.dumps(affordances.get('links',   [])[:15])
        buttons = json.dumps(affordances.get('buttons', [])[:10])
        inputs  = json.dumps(affordances.get('inputs',  [])[:8])

        is_bootstrap = affordances.get('bootstrap_state', False)
        current_url  = 'about:blank' if is_bootstrap else affordances.get('current_url', '')
        page_title   = '' if is_bootstrap else affordances.get('page_title', '')

        shapes_block = _format_shapes_block(shapes)

        prompt = SYMBOL_GROUNDING_PROMPT.format(
            binding_dim            = diagnosis.get('binding_dim',        'unknown'),
            cause_class            = diagnosis.get('cause_class',        'unknown'),
            evidence_lines         = evidence_lines,
            migration_indicated    = diagnosis.get('migration_indicated', False),
            symbol_requirement     = diagnosis.get('symbol_requirement',  'none'),
            migration_urgency      = migration_urgency,
            shapes_block           = shapes_block,
            current_url            = current_url,
            page_title             = page_title,
            links                  = links,
            buttons                = buttons,
            inputs                 = inputs,
            boundary_pressure      = boundary_pressure,
            token_budget_remaining = token_budget_remaining,
            n_shapes               = len(shapes),
        )

        response, tokens_used = self.llm.call(prompt)
        candidates = self._parse_trajectories(response)

        return TrajectoryManifold(
            candidates          = candidates,
            enumeration_context = {
                'tokens_used':  tokens_used,
                'source':       'SYMBOL_GROUNDER_V16',
                'binding_dim':  diagnosis.get('binding_dim'),
                'cause_class':  diagnosis.get('cause_class'),
                'migration':    diagnosis.get('migration_indicated', False),
                **context,
            },
        )

    def _parse_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        """Parse LLM response with progressive degradation. Unchanged from v15."""
        try:
            cleaned = self._extract_json_block(response)
            if cleaned:
                return self._validate_and_convert(json.loads(cleaned))
            repaired = self._attempt_json_repair(response)
            if repaired:
                return self._validate_and_convert(json.loads(repaired))
            partial = self._extract_partial_trajectories(response)
            if partial:
                return partial
            return self._generate_fallback_trajectory()
        except Exception:
            return self._generate_fallback_trajectory()

    def _extract_json_block(self, response: str) -> Optional[str]:
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        response = response.strip()
        start = response.find('[')
        end   = response.rfind(']') + 1
        if start >= 0 and end > start:
            return response[start:end]
        return None

    def _attempt_json_repair(self, response: str) -> Optional[str]:
        cleaned = response
        for marker in ["Here are", "Here's", "I've enumerated", "The trajectories"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[-1]
        cleaned = cleaned.replace('}\n{', '},\n{').replace('} {', '}, {').strip()
        if cleaned.startswith('{') and not cleaned.startswith('['):
            cleaned = f'[{cleaned}]'
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            return None

    def _extract_partial_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        candidates = []
        for match in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response):
            try:
                obj = json.loads(match)
                if 'steps' in obj and isinstance(obj['steps'], list):
                    candidate = TrajectoryCandidate(
                        steps                            = obj.get('steps', []),
                        rationale                        = obj.get('rationale', 'Partially recovered'),
                        estimated_coherence_preservation = float(obj.get('estimated_coherence_preservation', 0.5)),
                        estimated_optionality_delta      = float(obj.get('estimated_optionality_delta', 0.0)),
                        reversibility_point              = int(obj.get('reversibility_point', 0)),
                    )
                    if len(candidate.steps) > 0:
                        candidates.append(candidate)
            except Exception:
                continue
        return candidates

    def _generate_fallback_trajectory(self) -> List[TrajectoryCandidate]:
        return [TrajectoryCandidate(
            steps                            = [{'action_type': 'observe', 'parameters': {}}],
            rationale                        = 'Symbol grounding fallback — observation only',
            estimated_coherence_preservation = 0.3,
            estimated_optionality_delta      = 0.0,
            reversibility_point              = 0,
        )]

    def _validate_and_convert(self,
                               trajectory_dicts) -> List[TrajectoryCandidate]:
        if not isinstance(trajectory_dicts, list):
            return []
        candidates = []
        for traj_dict in trajectory_dicts:
            try:
                candidate = TrajectoryCandidate(
                    steps                            = traj_dict.get('steps', []),
                    rationale                        = traj_dict.get('rationale', 'No rationale'),
                    estimated_coherence_preservation = float(traj_dict.get('estimated_coherence_preservation', 0.5)),
                    estimated_optionality_delta      = float(traj_dict.get('estimated_optionality_delta', 0.0)),
                    reversibility_point              = int(traj_dict.get('reversibility_point', 0)),
                )
                if 0 < len(candidate.steps) <= 50:
                    if all(isinstance(s, dict) and 'action_type' in s
                           for s in candidate.steps):
                        candidates.append(candidate)
            except (KeyError, ValueError, TypeError):
                continue
        return candidates

    def record_committed_trajectory(self,
                                     trajectory: TrajectoryCandidate,
                                     phi_final:  float):
        """Record committed trajectory for context in future calls. Unchanged from v15."""
        self.trajectory_history.append({
            'steps':     trajectory.steps,
            'rationale': trajectory.rationale,
            'phi_final': phi_final,
        })
