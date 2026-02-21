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
from typing import Dict, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
import numpy as np
import copy
import time
from collections import deque

BASE_AFFORDANCES = {
    'navigate', 'click', 'fill', 'type', 'read',
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',
    'python'
}

# Interface-coupled signals — artifacts of HTML/browser rendering, not reality structure.
# BLOCKED from becoming discovered genome axes.
# A signal that vanishes when you change browsers is modeling the interface, not the world.
INTERFACE_COUPLED_SIGNALS = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity'
}

# v14.1: Named constant for Layer 1 parameter list — used by velocity system throughout
LAYER1_PARAMS = ['S_bias', 'I_bias', 'P_bias', 'A_bias', 'rigidity_init', 'phi_coherence_weight']

# v14.1: Explicit mapping from Layer 1 param name to its velocity field name
VELOCITY_FIELD = {
    'S_bias': 'S_velocity',
    'I_bias': 'I_velocity',
    'P_bias': 'P_velocity',
    'A_bias': 'A_velocity',
    'rigidity_init': 'rigidity_init_velocity',
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


@dataclass
class SubstrateState:
    """Four-dimensional information processing geometry."""
    S: float
    I: float
    P: float
    A: float

    def __post_init__(self):
        self.smo = SMO(history_depth=10)

    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}

    def apply_delta(self, observed_delta: Dict[str, float], predicted_delta: Dict[str, float] = None):
        if predicted_delta is None:
            predicted_delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

        # S, I, P update from reality contact via SMO.
        # A is NOT updated here — A is a derived measurement of basin drift,
        # computed from genome geometry after each micro-perturbation batch.
        # See MentatTriad._compute_a().
        self.S = self.smo.apply(self.S, observed_delta.get('S', 0), predicted_delta.get('S', 0))
        self.I = self.smo.apply(self.I, observed_delta.get('I', 0), predicted_delta.get('I', 0))
        self.P = self.smo.apply(self.P, observed_delta.get('P', 0), predicted_delta.get('P', 0))

        # P grounding invariant: P cannot exceed what S/I capacity structurally supports.
        # High P with collapsed S/I is hallucination — internal confidence uncoupled from
        # reality contact. This writeback makes the grounding authoritative on the substrate
        # itself, not just on Phi reads. Replaces the previous scalar damping threshold
        # (P > 0.9) which couldn't prevent S=0, I=0, P=1.0 collapse.
        # CANONICAL formula: PhiField.si_capacity(S, I) — must stay identical to that.
        SI_capacity = min(self.S, self.I) * 0.5 + (self.S + self.I) / 4.0
        self.P = float(np.clip(self.P, 0.0, SI_capacity * 2.0))

    def rollback(self) -> bool:
        if not self.smo.can_reverse():
            return False

        prev_S = self.smo.reverse()
        prev_I = self.smo.reverse()
        prev_P = self.smo.reverse()
        prev_A = self.smo.reverse()

        if all(x is not None for x in [prev_S, prev_I, prev_P, prev_A]):
            self.S = prev_S
            self.I = prev_I
            self.P = prev_P
            self.A = prev_A
            return True

        return False


class StateTrace:
    """Ordered history of substrate states for field calculations."""

    def __init__(self, max_length: int = 1000):
        self.history: deque = deque(maxlen=max_length)

    def record(self, state: SubstrateState):
        self.history.append(state.as_dict())

    def get_recent(self, n: int) -> List[Dict]:
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]

    def __len__(self) -> int:
        return len(self.history)


class PhiField:
    """Information Potential Field (Φ-field)."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, A0=0.7, alpha_crk=2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.A0 = A0
        self.alpha_crk = alpha_crk

    @staticmethod
    def si_capacity(S: float, I: float) -> float:
        """
        Single source of truth for the SI_capacity grounding formula.
        Called by SubstrateState.apply_delta() for the authoritative writeback,
        and by phi() for read-only grounding during gradient probes.
        Both must use the same formula — this ensures they do.
        """
        return min(S, I) * 0.5 + (S + I) / 4.0

    def phi(self, state: SubstrateState, trace: StateTrace, crk_violations: List[Tuple[str, float]] = None) -> float:
        """
        Φ field: grounded optionality + coherence - curvature - violations

        v13.8: P grounded to S/I capacity via bottleneck formula. Prevents compensation.
        Φ measures viability in entropy manifold, not reward signal.
        """
        # Read-only grounding: gradient() probes via deepcopy+setattr, bypassing
        # apply_delta. Without this, gradient probes that push P above SI_capacity
        # would see phantom optionality — false gradient signal near the ceiling.
        # apply_delta() does the authoritative writeback; phi() does the safe read.
        cap = self.si_capacity(state.S, state.I)
        effective_P = min(state.P, cap * 2.0)
        opt = np.log(1.0 + max(effective_P, 0.0))

        strain = (state.A - self.A0) ** 2

        recent = trace.get_recent(3)
        curv = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2*h1[k] + h0[k])

        phi_raw = self.alpha * opt - self.beta * strain - self.gamma * curv

        crk_penalty = 0.0
        if crk_violations:
            crk_penalty = self.alpha_crk * sum(severity for _, severity in crk_violations)

        phi_net = phi_raw - crk_penalty

        return phi_net

    def gradient(self, state: SubstrateState, trace: StateTrace, crk_violations: List[Tuple[str, float]] = None, eps=0.01) -> Dict[str, float]:
        phi_current = self.phi(state, trace, crk_violations)
        grad = {}

        for dim in ['S', 'I', 'P', 'A']:
            state_plus = copy.deepcopy(state)
            setattr(state_plus, dim, getattr(state_plus, dim) + eps)
            phi_plus = self.phi(state_plus, trace, crk_violations)
            grad[dim] = (phi_plus - phi_current) / eps

        return grad


class CRKMonitor:
    """Constraint Recognition Kernel (CRK)."""

    def evaluate(self, state: SubstrateState, trace: StateTrace,
                 reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
        violations = []

        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            jump = sum(abs(prev[k] - getattr(state, k)) for k in ["S", "I", "P", "A"])
            if jump > 0.3:
                violations.append(("C1_Continuity", jump - 0.3))

        if state.P < 0.35:
            violations.append(("C2_Optionality", 0.35 - state.P))

        confidence = state.S + state.I
        if confidence < 0.7:
            violations.append(("C3_NonInternalization", 0.7 - confidence))

        if reality_delta and len(trace) >= 3:
            feedback_magnitude = sum(abs(v) for v in reality_delta.values())
            if feedback_magnitude < 0.01:
                violations.append(("C4_Reality", 0.01 - feedback_magnitude))

        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            prev_P = prev["P"]
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
