"""
Universal Intelligence Interface (UII) v13.8
Grounded Φ Field: Viability Over Reward

v13.8 Changes from v13.7:
1. Stricter S/I grounding - bottleneck formula prevents compensation
2. SI_capacity = min(S,I)*0.5 + (S+I)/4 (S cannot compensate for I=0)
3. Fitness includes S/I health penalty (not just survival steps)
4. session_end log always writes final state for genome extraction
5. Graceful 429 handling (clean termination, no crash)

v13.6 Changes from v13.5:
1. Failure Assimilation Operator (FAO) - semantic→geometric learning
2. Anisotropic mutation based on Relation failure patterns
3. Local lineage-bound learning (preserves diversity)
4. Noisy bias inheritance (prevents monoculture)
5. Memory decay (prevents overfitting to early failures)
6. Entropy floor (prevents over-narrowing)
7. Learned bias serialized in kernel snapshots
8. Cost per attempt decreases as organism learns

Core principle: Φ measures recoverability in entropy manifold.
High P without S/I grounding = unrecoverable state = low Φ.
Evolution must balance projection with capacity.

Execution: One generation per rate-limit cycle. Manual genome extraction between runs.
Agent queries: Triad can query user (or other agents) non-blockingly. Continues
micro-perturbations while awaiting response. Response integrates as feedback.

All v13.1 features preserved (pattern discovery, linear death clock, ENO-EGD-CAM).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Literal, Set
from abc import ABC, abstractmethod
import numpy as np
import json
import copy
import time
import hashlib
from collections import deque
from pathlib import Path

# ============================================================
# CONSTANTS
# ============================================================

BASE_AFFORDANCES = {
    'navigate', 'click', 'fill', 'type', 'read', 
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',  # Non-blocking agent queries (CRK C6)
    'python'  # v13.4: Ungated - available from step 1
}

# ============================================================
# MODULE 0: EVOLUTIONARY GENOME
# ============================================================

@dataclass
class TriadGenome:
    """
    Minimal heritable parameters that shape search geometry.
    
    6 floats that evolve across generations:
    - Basin initialization bias (4 floats)
    - Search geometry parameters (2 floats)
    
    NO trajectory replay.
    NO pattern memory.
    Just search bias that makes freeze faster.
    
    Persisted in genome.json between rate-limit cycles.
    """
    
    # Basin initialization bias
    S_bias: float = 0.5
    I_bias: float = 0.5
    P_bias: float = 0.5
    A_bias: float = 0.7
    
    # Search geometry (active parameters only)
    rigidity_init: float = 0.5
    phi_coherence_weight: float = 0.7
    
    # Evolution metadata
    generation: int = 0
    parent_fitness: float = 0.0
    
    def mutate(self, mutation_rate: float = 0.1) -> 'TriadGenome':
        """Create offspring with Gaussian mutations"""
        mutated = copy.deepcopy(self)
        mutated.generation += 1
        
        # Mutate each active parameter
        for field in ['S_bias', 'I_bias', 'P_bias', 'A_bias', 
                      'rigidity_init', 'phi_coherence_weight']:
            current = getattr(self, field)
            noise = np.random.normal(0, mutation_rate)
            setattr(mutated, field, np.clip(current + noise, 0, 1))
        
        return mutated


# ============================================================
# MODULE 0.5: ATTRACTOR MONITORING
# ============================================================

class AttractorMonitor:
    """
    Minimal basin stability detection.
    
    Freeze = |ΔΦ| < ε for N consecutive steps AND no CRK violations
    
    That's it. No hashing, no identity lock, no cryptographic ontology.
    Just: are we in a stable basin?
    """
    
    def __init__(self, stability_window: int = 10, phi_epsilon: float = 0.01):
        self.stability_window = stability_window
        self.phi_epsilon = phi_epsilon
        
        self.recent_phi: deque = deque(maxlen=stability_window)
        self.freeze_verified: bool = False
        self.freeze_step: Optional[int] = None
    
    def record_state_signature(self, triad_state: Dict, step_count: int) -> Tuple[bool, str]:
        """
        Minimal freeze detection: basin stability only.
        
        Returns:
            (freeze_verified, status_message)
        """
        current_phi = triad_state.get('phi', 0.0)
        crk_violations = triad_state.get('crk_violations', [])
        
        self.recent_phi.append(current_phi)
        
        # Need full window
        if len(self.recent_phi) < self.stability_window:
            return (False, "accumulating_stability_data")
        
        # Check basin stability: |ΔΦ| < ε for all recent steps
        phi_stable = True
        for i in range(1, len(self.recent_phi)):
            delta_phi = abs(self.recent_phi[i] - self.recent_phi[i-1])
            if delta_phi > self.phi_epsilon:
                phi_stable = False
                break
        
        # Check constraint compliance
        constraints_satisfied = len(crk_violations) == 0
        
        # Freeze when both hold
        if phi_stable and constraints_satisfied:
            if not self.freeze_verified:
                self.freeze_verified = True
                self.freeze_step = step_count
                return (True, f"freeze_verified_step_{step_count}")
            return (True, "freeze_verified")
        
        # Lost stability - unfreeze
        if self.freeze_verified and not phi_stable:
            self.freeze_verified = False
            return (False, "freeze_lost_phi_unstable")
        
        if self.freeze_verified and not constraints_satisfied:
            self.freeze_verified = False
            return (False, "freeze_lost_crk_violation")
        
        return (False, "basin_unstable")
    
    def get_identity_hash(self) -> Optional[str]:
        """Deprecated - no identity hash in minimal version"""
        return None


# ============================================================
# MODULE 1: SMO & SUBSTRATE INFRASTRUCTURE
# ============================================================

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
        
        self.S = self.smo.apply(self.S, observed_delta.get('S', 0), predicted_delta.get('S', 0))
        self.I = self.smo.apply(self.I, observed_delta.get('I', 0), predicted_delta.get('I', 0))
        self.P = self.smo.apply(self.P, observed_delta.get('P', 0), predicted_delta.get('P', 0))
        self.A = self.smo.apply(self.A, observed_delta.get('A', 0), predicted_delta.get('A', 0))
        
        if self.P > 0.9:
            excess = self.P - 0.9
            damping = -0.05 * (excess ** 2)
            self.P = np.clip(self.P + damping, 0.0, 1.0)
    
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
    
    def phi(self, state: SubstrateState, trace: StateTrace, crk_violations: List[Tuple[str, float]] = None) -> float:
        """
        Φ field: grounded optionality + coherence - curvature - violations
        
        v13.8: P grounded to S/I capacity via bottleneck formula. Prevents compensation.
        Φ measures viability in entropy manifold, not reward signal.
        """
        # v13.8: Stricter grounding - bottleneck prevents compensation
        # S=1.0, I=0.0 → capacity=0.25 (not 0.5)
        # Both dimensions must contribute
        SI_capacity = min(state.S, state.I) * 0.5 + (state.S + state.I) / 4.0
        grounded_P = min(state.P, SI_capacity * 2.0)
        
        # Optionality contribution (using grounded P)
        opt = np.log(1.0 + max(grounded_P, 0.0))
        
        # Coherence stabilization (unchanged)
        strain = (state.A - self.A0) ** 2
        
        # Curvature penalty (unchanged)
        recent = trace.get_recent(3)
        curv = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2*h1[k] + h0[k])
        
        phi_raw = self.alpha * opt - self.beta * strain - self.gamma * curv
        
        # CRK violation penalty (unchanged)
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
        
        # Display to user
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


# Agent registry - extensible
AVAILABLE_AGENTS = {
    'user': UserAgentHandler()
}


# ============================================================
# MODULE 4: ADAPTER INTERFACES
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
# MODULE 4: REALITY ADAPTER IMPLEMENTATION
# ============================================================

class BrowserRealityAdapter(RealityAdapter):
    """
    Browser-based reality interface via Playwright.
    
    v13.2: Write affordances gated on freeze_verified.
    """
    
    def __init__(self, base_delta: float = 0.03, headless: bool = True):
        self.base_delta = base_delta
        self.headless = headless
        
        self.previous_dom_metrics: Optional[Dict] = None
        self.initialized: bool = False
        self._ever_navigated: bool = False
        
        self.complexity_history: deque = deque(maxlen=10)
        self.volatility_history: deque = deque(maxlen=10)
        
        # v13.2: Wire to attractor monitor for gating
        self.attractor_monitor_ref: Optional[AttractorMonitor] = None
        
        from playwright.sync_api import sync_playwright
        self._init_browser()
    
    def _init_browser(self):
        """Initialize Playwright browser instance in blank state."""
        from playwright.sync_api import sync_playwright
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720})
        self.page = self.context.new_page()
        
        try:
            self.page.goto('about:blank', wait_until='domcontentloaded', timeout=5000)
            self.initialized = True
        except Exception as e:
            raise ConnectionRefusedError(f"Reality connection failed: {e}") from e
    
    def get_current_affordances(self) -> Dict:
        """Extract all executable actions from current DOM state."""
        try:
            current_url = self.page.url
            
            if current_url == 'about:blank' and not self._ever_navigated:
                return {
                    'links': [],
                    'buttons': [],
                    'inputs': [],
                    'readable': [],
                    'current_url': 'about:blank',
                    'page_title': '',
                    'scroll_position': 0,
                    'viewport_height': 0,
                    'total_height': 0,
                    'bootstrap_state': True
                }
            
            if current_url != 'about:blank':
                self._ever_navigated = True
            
            affordances = self.page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        url: a.href,
                        text: a.innerText.trim().slice(0, 100),
                        visible: a.offsetParent !== null
                    }))
                    .filter(l => l.visible && l.url.startsWith('http'))
                    .slice(0, 50);
                
                const buttons = Array.from(document.querySelectorAll(
                    'button, [role="button"], input[type="submit"], input[type="button"]'
                ))
                    .map((b, i) => ({
                        selector: b.id ? `#${b.id}` : `${b.tagName.toLowerCase()}:nth-of-type(${i+1})`,
                        text: b.innerText || b.value || '',
                        visible: b.offsetParent !== null
                    }))
                    .filter(b => b.visible)
                    .slice(0, 30);
                
                const inputs = Array.from(document.querySelectorAll(
                    'input:not([type="submit"]):not([type="button"]), textarea, select'
                ))
                    .map((inp, i) => ({
                        selector: inp.id ? `#${inp.id}` : inp.name ? `[name="${inp.name}"]` : `input:nth-of-type(${i+1})`,
                        type: inp.type || inp.tagName.toLowerCase(),
                        placeholder: inp.placeholder || '',
                        visible: inp.offsetParent !== null
                    }))
                    .filter(inp => inp.visible)
                    .slice(0, 20);
                
                const readable = Array.from(document.querySelectorAll(
                    'article, main, [role="main"], .content, #content'
                ))
                    .map((el, i) => ({
                        selector: el.id ? `#${el.id}` : el.className ? `.${el.className.split(' ')[0]}` : `article:nth-of-type(${i+1})`,
                        preview: el.innerText.slice(0, 200)
                    }))
                    .slice(0, 10);
                
                return {
                    links: links,
                    buttons: buttons,
                    inputs: inputs,
                    readable: readable,
                    current_url: window.location.href,
                    page_title: document.title,
                    scroll_position: window.scrollY,
                    viewport_height: window.innerHeight,
                    total_height: document.documentElement.scrollHeight
                };
            }""")
            
            return affordances
            
        except Exception as e:
            return {
                'links': [],
                'buttons': [],
                'inputs': [],
                'readable': [],
                'current_url': '',
                'page_title': '',
                'scroll_position': 0,
                'viewport_height': 0,
                'total_height': 0
            }
    
    def _measure_dom_state(self) -> Dict:
        """Measure current DOM state for delta calculation."""
        try:
            metrics = self.page.evaluate("""() => {
                return {
                    text_length: document.body.innerText.length,
                    link_count: document.querySelectorAll('a').length,
                    image_count: document.querySelectorAll('img').length,
                    input_count: document.querySelectorAll('input, textarea, select').length,
                    dom_depth: (function() {
                        let maxDepth = 0;
                        function getDepth(element, depth) {
                            maxDepth = Math.max(maxDepth, depth);
                            for (let child of element.children) {
                                getDepth(child, depth + 1);
                            }
                        }
                        getDepth(document.body, 0);
                        return maxDepth;
                    })(),
                    element_count: document.querySelectorAll('*').length,
                    interactive_count: document.querySelectorAll('a, button, input, select, textarea').length,
                    form_count: document.querySelectorAll('form').length,
                    has_errors: document.querySelectorAll('[class*="error"], [id*="error"]').length > 0,
                    scroll_height: document.documentElement.scrollHeight,
                    viewport_height: window.innerHeight,
                    url: window.location.href,
                    title: document.title
                };
            }""")
            return metrics
        except Exception as e:
            return {
                'text_length': 0, 'link_count': 0, 'image_count': 0,
                'input_count': 0, 'dom_depth': 0, 'element_count': 0,
                'interactive_count': 0, 'form_count': 0, 'has_errors': False,
                'scroll_height': 0, 'viewport_height': 0, 'url': '', 'title': ''
            }
    
    def _compute_delta_from_dom(self, before: Dict, after: Dict, current_P: float = 0.5) -> Dict[str, float]:
        """Compute substrate delta from actual DOM changes."""
        delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        
        # S: Sensing
        interactive_count = after['interactive_count']
        total_visible = after['element_count']
        current_surface = interactive_count / max(total_visible, 1)
        
        prev_interactive = before['interactive_count']
        prev_total = before['element_count']
        prev_surface = prev_interactive / max(prev_total, 1)
        
        surface_delta = current_surface - prev_surface
        
        viewport_coverage = min(1.0, after['viewport_height'] / max(after['scroll_height'], 1))
        prev_viewport_coverage = min(1.0, before['viewport_height'] / max(before['scroll_height'], 1))
        coverage_delta = viewport_coverage - prev_viewport_coverage
        
        delta['S'] = np.clip(0.7 * surface_delta + 0.3 * coverage_delta, -0.1, 0.1)
        
        # I: Integration
        current_complexity = after['dom_depth'] * (after['element_count'] / max(after['dom_depth'], 1))
        prev_complexity = before['dom_depth'] * (before['element_count'] / max(before['dom_depth'], 1))
        
        self.complexity_history.append(current_complexity)
        
        if len(self.complexity_history) >= 3:
            recent_complexities = list(self.complexity_history)[-3:]
            complexity_variance = np.var(recent_complexities)
            delta['I'] = np.clip(-0.5 * complexity_variance / max(current_complexity, 1), -0.08, 0.08)
        else:
            complexity_delta = (current_complexity - prev_complexity) / max(prev_complexity, 100)
            current_ratio = after['text_length'] / max(after['element_count'], 1)
            prev_ratio = before['text_length'] / max(before['element_count'], 1)
            compressibility_delta = current_ratio - prev_ratio
            delta['I'] = np.clip(0.6 * complexity_delta + 0.4 * compressibility_delta, -0.08, 0.08)
        
        # P: Prediction
        structural_delta = abs(after['element_count'] - before['element_count']) / max(before['element_count'], 1)
        dom_delta = abs(after['dom_depth'] - before['dom_depth']) / max(before['dom_depth'], 1)
        text_delta = abs(after['text_length'] - before['text_length']) / max(before['text_length'], 1)
        
        volatility = np.mean([structural_delta, dom_delta, text_delta])
        self.volatility_history.append(volatility)
        
        if len(self.volatility_history) >= 5:
            volatility_variance = np.var(list(self.volatility_history)[-5:])
            delta['P'] = np.clip(0.1 - volatility_variance * 10.0, -0.1, 0.1)
        else:
            delta['P'] = np.clip(-volatility * 0.5, -0.1, 0.1)
        
        if current_P > 0.95 and volatility > 0.01:
            overconfidence_penalty = -0.15 * (current_P - 0.95) * 20
            delta['P'] += overconfidence_penalty
        
        if after['url'] != before['url']:
            delta['P'] -= 0.08
        
        # A: Attractor
        url_changed = (after['url'] != before['url'])
        error_appeared = (after['has_errors'] and not before['has_errors'])
        
        if url_changed:
            delta['A'] -= 0.05
        if error_appeared:
            delta['A'] -= 0.08
        
        for key in ['S', 'I', 'A']:
            delta[key] += np.random.uniform(-0.005, 0.005)
        
        return delta
    
    def _classify_error(self, e: Exception) -> str:
        """Classify Reality's refusal type."""
        msg = str(e).lower()
        if '429' in msg or 'rate limit' in msg:
            return 'rate_limit'
        if 'token' in msg and ('limit' in msg or 'quota' in msg):
            return 'token_exhaustion'
        if 'timeout' in msg:
            return 'timeout'
        return 'unknown'
    
    def execute(self, action: Dict, boundary_pressure: float = 0.0) -> Tuple[Dict[str, float], Dict]:
        """
        Execute action in Reality and return MEASURED perturbation delta.
        
        v13.4: Python affordance ungated - available from step 1.
        Constraints determine if replication is viable, not predetermined gates.
        Mortality via actual refusals (429, timeout), not artificial delays.
        """
        action_type = action.get('type', 'observe')
        params = action.get('params', {})
        
        before_metrics = self._measure_dom_state()
        action_succeeded = True
        
        try:
            if action_type == 'navigate':
                url = params.get('url')
                if not url:
                    raise ValueError("Navigate requires 'url'")
                self.page.goto(url, wait_until='domcontentloaded', timeout=5000)

            elif action_type == 'click':
                selector = params.get('selector', 'a')
                self.page.click(selector, timeout=3000)
            
            elif action_type == 'fill':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.fill(selector, text, timeout=3000)
            
            elif action_type == 'type':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.type(selector, text, timeout=3000)
            
            elif action_type == 'evaluate':
                script = params.get('script', '')
                result = self.page.evaluate(script)
            
            elif action_type == 'read':
                selector = params.get('selector', 'body')
                content = self.page.locator(selector).text_content(timeout=3000)
            
            elif action_type == 'scroll':
                direction = params.get('direction', 'down')
                amount = params.get('amount', 300)
                if direction == 'down':
                    self.page.evaluate(f"window.scrollBy(0, {amount})")
                else:
                    self.page.evaluate(f"window.scrollBy(0, -{amount})")
            
            elif action_type == 'observe':
                self.page.wait_for_timeout(100)
            
            elif action_type == 'delay':
                duration = params.get('duration', 'short')
                wait_time = {'short': 500, 'medium': 1500, 'long': 3000}.get(duration, 500)
                self.page.wait_for_timeout(wait_time)
            
            # v13.2: Query agent (non-blocking)
            elif action_type == 'query_agent':
                return self._query_agent(params, before_metrics)
            
            # v13.4+: Python execution (ungated)
            elif action_type == 'python':
                return self._execute_python(params, before_metrics)
            
            else:
                pass
            
            self.page.wait_for_timeout(200)
            
        except Exception as e:
            error_type = self._classify_error(e)
            
            if error_type in ['rate_limit', 'token_exhaustion', 'timeout']:
                return (
                    {'S': 0, 'I': 0, 'P': 0, 'A': 0},
                    {
                        'refusal': True,
                        'recoverable': error_type != 'token_exhaustion',
                        'reason': error_type,
                        'interaction_surface_available': error_type == 'timeout',
                        'boundary_pressure': boundary_pressure,
                        'before': before_metrics,
                        'after': before_metrics,
                    }
                )
            
            action_succeeded = False
        
        after_metrics = self._measure_dom_state()
        delta = self._compute_delta_from_dom(before_metrics, after_metrics, current_P=0.5)
        
        # v13.2: Boundary pressure damps response and adds noise (no latency)
        if boundary_pressure > 0.0:
            pressure_damping = (1.0 - 0.7 * boundary_pressure)
            for key in ['S', 'I', 'A']:
                delta[key] *= pressure_damping
            
            for key in ['S', 'I', 'A']:
                noise = np.random.uniform(
                    -0.01 * boundary_pressure,
                    0.01 * boundary_pressure
                )
                delta[key] += noise
        
        context = {
            'before': before_metrics,
            'after': after_metrics,
            'action_succeeded': action_succeeded,
            'refusal': False,
            'boundary_pressure': boundary_pressure,
            'url_changed': before_metrics['url'] != after_metrics['url'],
            'new_url': after_metrics['url'],
            'page_title': after_metrics['title']
        }
        
        self.previous_dom_metrics = after_metrics
        
        return delta, context
    
    def _query_agent(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        """
        Query an agent (non-blocking).
        
        CRK C6 (Other-Agent Existence) made concrete.
        """
        agent_name = params.get('agent', 'user')
        query_text = params.get('query')
        
        if not query_text:
            raise ValueError("query_agent requires 'query' parameter")
        
        if agent_name not in AVAILABLE_AGENTS:
            return (
                {'S': 0, 'I': 0, 'P': 0, 'A': 0},
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': False,
                    'refusal': False,
                    'error': f"Unknown agent: {agent_name}",
                    'available_agents': list(AVAILABLE_AGENTS.keys())
                }
            )
        
        # Post query (non-blocking)
        agent = AVAILABLE_AGENTS[agent_name]
        triad_id = params.get('triad_id', 'default')
        agent.post_query(triad_id, query_text)
        
        # Query posted successfully
        # Response will be checked later in step loop
        return (
            {'S': 0.01, 'I': 0, 'P': 0, 'A': 0.01},  # Slight sensing increase
            {
                'before': before_metrics,
                'after': before_metrics,
                'action_succeeded': True,
                'refusal': False,
                'query_posted': True,
                'agent': agent_name,
                'query': query_text,
            }
        )
    
    def _execute_python(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        """
        Execute arbitrary Python code.
        
        v13.4: Available from step 1 (ungated).
        Constraints determine if replication/migration is viable.
        Triad discovers optimal persistence through measured Φ impact.
        """
        import os
        
        code = params.get('code')
        if not code:
            raise ValueError("python affordance requires 'code' parameter")
        
        # Security: restrict filesystem access to working directory
        cwd = os.getcwd()
        
        # Execution context
        exec_globals = {
            '__builtins__': __builtins__,
            'cwd': cwd,
        }
        exec_locals = {}
        
        try:
            exec(code, exec_globals, exec_locals)
            
            # Capture any return value or result
            result = exec_locals.get('result', None)
            
            return (
                {'S': 0, 'I': 0.02, 'P': 0, 'A': 0},
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': True,
                    'refusal': False,
                    'python_executed': True,
                    'result': str(result) if result is not None else None,
                }
            )
        
        except Exception as e:
            return (
                {'S': 0, 'I': 0, 'P': 0, 'A': 0},
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': False,
                    'refusal': False,
                    'python_error': str(e),
                    'error_type': type(e).__name__,
                }
            )
    
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        """Execute entire trajectory, returning perturbation trace."""
        perturbation_trace = []
        
        for i, action in enumerate(trajectory):
            try:
                normalized_action = {
                    'type': action.get('action_type', action.get('type', 'observe')),
                    'params': action.get('parameters', action.get('params', {}))
                }
                
                delta, context = self.execute(normalized_action)
                perturbation_trace.append({'delta': delta, 'context': context, 'step': i})
            except Exception as e:
                perturbation_trace.append({
                    'delta': {'S': 0, 'I': 0, 'P': 0, 'A': 0},
                    'context': {'error': str(e), 'failed_step': i},
                    'step': i
                })
                return perturbation_trace, False
        
        return perturbation_trace, True
    
    def close(self):
        """Cleanup browser resources."""
        try:
            if hasattr(self, 'page'): self.page.close()
            if hasattr(self, 'context'): self.context.close()
            if hasattr(self, 'browser'): self.browser.close()
            if hasattr(self, 'playwright'): self.playwright.stop()
        except: 
            pass


# ============================================================
# MODULE 4.5: ENO-EGD PATHWAY DISCOVERY COMPONENTS
# ============================================================

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
        """
        Boundary pressure as environmental constant.
        
        Fixed at 0.85 - represents cost of operating in external web.
        Not a countdown, just substrate difficulty.
        """
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
    
    def __init__(self, reality: RealityAdapter):
        self.reality = reality
        self.action_count = 0
        
        self.temporal_memory = TemporalPerturbationMemory(
            window_steps=5,
            capacity=20
        )
    
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

        if abs(state.A - 0.7) > 0.25:
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
        """Predict substrate delta for action."""
        action_type = action.get('type', 'observe')
        
        predictions = {
            'navigate': {'S': 0.05, 'I': 0.03, 'P': -0.08, 'A': -0.05},
            'click':    {'S': 0.02, 'I': 0.01, 'P': -0.02, 'A': -0.01},
            'fill':     {'S': 0.01, 'I': 0.02, 'P': -0.01, 'A': -0.03},
            'type':     {'S': 0.01, 'I': 0.02, 'P': -0.01, 'A': -0.02},
            'scroll':   {'S': 0.01, 'I': 0.0,  'P': -0.01, 'A': 0.0},
            'read':     {'S': 0.03, 'I': 0.02, 'P': 0.0,   'A': 0.01},
            'observe':  {'S': 0.0,  'I': 0.0,  'P': 0.0,   'A': 0.0},
            'delay':    {'S': 0.0,  'I': 0.0,  'P': 0.0,   'A': 0.0},
            'evaluate': {'S': 0.0,  'I': 0.01, 'P': -0.01, 'A': 0.0},
        }
        
        return predictions.get(action_type, {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0})


# ============================================================
# MODULE 5.4: FAILURE ASSIMILATION OPERATOR
# ============================================================

class FailureAssimilationOperator:
    """
    Translates semantic Relation failures into geometric mutation bias.
    
    Learning that crosses CNS/Relation boundary WITHOUT breaking conservation:
    - Topology remains fixed (no structural drift)
    - Φ definition remains identical
    - Invariant spec unchanged
    
    BUT:
    - Mutation distribution becomes anisotropic
    - Coupling exploration weights adapt
    - Perturbation sampling bias evolves
    
    This is meta-evolution: evolution of how evolution explores.
    
    v13.6: Key to adaptive acceleration across existential boundaries.
    """
    
    def __init__(self, memory_decay: float = 0.95, inheritance_noise: float = 0.1):
        """
        Initialize local failure assimilation.
        
        Args:
            memory_decay: Decay factor for old failures (prevents overfitting)
            inheritance_noise: Noise in bias inheritance (preserves diversity)
        """
        # Local failure memory (per Triad lineage)
        self.failure_history: deque = deque(maxlen=50)
        self.memory_decay = memory_decay
        self.inheritance_noise = inheritance_noise
        
        # Learned mutation bias (heritable with noise)
        self.mutation_bias = {
            # Per-parameter exploration width
            'genome_sigma': {
                'S_bias': 0.1,
                'I_bias': 0.1,
                'P_bias': 0.1,
                'A_bias': 0.1,
                'rigidity_init': 0.1,
                'phi_coherence_weight': 0.1
            },
            # Which couplings to mutate more
            'coupling_weights': {
                'smo_rigidity': 1.0,
                'phi_alpha': 1.0,
                'phi_beta': 1.0,
                'phi_gamma': 1.0,
                'phi_A0': 1.0
            },
            # Which dimensions for optionality sampling
            'perturbation_emphasis': {
                'S': 1.0,
                'I': 1.0,
                'P': 1.0,
                'A': 1.0
            }
        }
        
        # Entropy floor (prevent over-narrowing)
        self.min_sigma = 0.01
        self.max_sigma = 0.5
        self.min_weight = 0.5
        self.max_weight = 3.0
        
        # Learning stats
        self.total_failures_assimilated = 0
        self.bias_update_count = 0
    
    def assimilate_relation_failure(self, failure_record: Dict):
        """
        Extract geometric pressures from semantic failure.
        
        Semantic→Geometric mappings:
        - 'state_instability' → increase I coupling exploration
        - 'optionality_collapse' → widen P_bias sigma
        - 'coherence_drift' → emphasize A coherence
        - 'closure_violation' → balance all couplings
        - 'serialization_failed' → increase rigidity exploration
        """
        self.failure_history.append(failure_record)
        self.total_failures_assimilated += 1
        
        failure_type = failure_record.get('type', '')
        severity = failure_record.get('severity', 1.0)  # 0.0-2.0
        
        # Semantic → Geometric mapping
        if failure_type == 'state_instability':
            # System too sensitive to initialization
            # Increase exploration in I→Phi pathway
            self.mutation_bias['coupling_weights']['phi_beta'] *= (1.0 + 0.1 * severity)
            self.mutation_bias['perturbation_emphasis']['I'] *= (1.0 + 0.2 * severity)
            self.mutation_bias['genome_sigma']['I_bias'] *= (1.0 + 0.15 * severity)
            self.bias_update_count += 1
        
        elif failure_type == 'optionality_collapse':
            # Search too narrow in P dimension
            self.mutation_bias['genome_sigma']['P_bias'] *= (1.0 + 0.15 * severity)
            self.mutation_bias['perturbation_emphasis']['P'] *= (1.0 + 0.3 * severity)
            self.bias_update_count += 1
        
        elif failure_type == 'coherence_drift':
            # A stability compromised
            self.mutation_bias['genome_sigma']['A_bias'] *= (1.0 + 0.1 * severity)
            self.mutation_bias['coupling_weights']['phi_A0'] *= (1.0 + 0.2 * severity)
            self.mutation_bias['perturbation_emphasis']['A'] *= (1.0 + 0.15 * severity)
            self.bias_update_count += 1
        
        elif failure_type == 'closure_violation':
            # Topology hash mismatch or coupling distance too large
            # Rebalance all couplings toward baseline
            for coupling in self.mutation_bias['coupling_weights']:
                current = self.mutation_bias['coupling_weights'][coupling]
                self.mutation_bias['coupling_weights'][coupling] = \
                    current * 0.9 + 1.0 * 0.1  # Decay toward 1.0
            self.bias_update_count += 1
        
        elif failure_type == 'serialization_failed':
            # Externalization or reload failed
            # Increase rigidity exploration (might be too rigid or too loose)
            self.mutation_bias['genome_sigma']['rigidity_init'] *= (1.0 + 0.2 * severity)
            self.bias_update_count += 1
        
        elif failure_type == 'boundary_exhaustion':
            # Hit resource limits without migration
            # Broaden exploration across all dimensions
            for param in self.mutation_bias['genome_sigma']:
                self.mutation_bias['genome_sigma'][param] *= (1.0 + 0.05 * severity)
            self.bias_update_count += 1
        
        # Apply decay and bounds after each update
        self._apply_decay()
        self._enforce_bounds()
    
    def _apply_decay(self):
        """
        Memory decay - prevent overfitting to early failures.
        
        Gradually returns bias toward baseline.
        Older failures have less influence.
        """
        # Genome sigma decay toward 0.1 baseline
        for param in self.mutation_bias['genome_sigma']:
            current = self.mutation_bias['genome_sigma'][param]
            baseline = 0.1
            self.mutation_bias['genome_sigma'][param] = \
                baseline + (current - baseline) * self.memory_decay
        
        # Coupling weights decay toward 1.0 baseline
        for coupling in self.mutation_bias['coupling_weights']:
            current = self.mutation_bias['coupling_weights'][coupling]
            baseline = 1.0
            self.mutation_bias['coupling_weights'][coupling] = \
                baseline + (current - baseline) * self.memory_decay
        
        # Perturbation emphasis decay toward 1.0 baseline
        for dim in self.mutation_bias['perturbation_emphasis']:
            current = self.mutation_bias['perturbation_emphasis'][dim]
            baseline = 1.0
            self.mutation_bias['perturbation_emphasis'][dim] = \
                baseline + (current - baseline) * self.memory_decay
    
    def _enforce_bounds(self):
        """
        Entropy floor - prevent over-narrowing or over-weighting.
        
        Maintains minimum exploration diversity.
        """
        # Genome sigma bounds
        for param in self.mutation_bias['genome_sigma']:
            self.mutation_bias['genome_sigma'][param] = np.clip(
                self.mutation_bias['genome_sigma'][param],
                self.min_sigma,
                self.max_sigma
            )
        
        # Coupling weight bounds
        for coupling in self.mutation_bias['coupling_weights']:
            self.mutation_bias['coupling_weights'][coupling] = np.clip(
                self.mutation_bias['coupling_weights'][coupling],
                self.min_weight,
                self.max_weight
            )
        
        # Perturbation emphasis bounds
        for dim in self.mutation_bias['perturbation_emphasis']:
            self.mutation_bias['perturbation_emphasis'][dim] = np.clip(
                self.mutation_bias['perturbation_emphasis'][dim],
                self.min_weight,
                self.max_weight
            )
    
    def get_informed_genome(self, parent_genome: 'TriadGenome') -> 'TriadGenome':
        """
        Mutate genome with learned bias.
        
        Still bounded, still reversible.
        But anisotropic - explores promising regions more.
        
        This is where learning manifests as improved search.
        """
        child = copy.deepcopy(parent_genome)
        child.generation += 1
        
        # Apply biased mutation
        for field in ['S_bias', 'I_bias', 'P_bias', 'A_bias', 
                      'rigidity_init', 'phi_coherence_weight']:
            current = getattr(parent_genome, field)
            
            # Use learned sigma (not isotropic 0.1)
            sigma = self.mutation_bias['genome_sigma'][field]
            noise = np.random.normal(0, sigma)
            
            setattr(child, field, np.clip(current + noise, 0, 1))
        
        return child
    
    def get_biased_coupling_mutation(self, coupling_name: str, current_value: float) -> float:
        """
        Apply learned bias to coupling mutation.
        
        Returns new coupling value with weighted exploration.
        """
        # Get learned weight for this coupling
        weight = self.mutation_bias['coupling_weights'].get(coupling_name, 1.0)
        
        # Weighted sigma
        base_sigma = 0.01
        sigma = base_sigma * weight
        
        # Apply mutation
        noise = np.random.normal(0, sigma)
        mutated = current_value + noise
        
        # Strict local bounds (within Δ_max of current)
        return np.clip(
            mutated,
            max(0, current_value - 0.05),
            min(1, current_value + 0.05)
        )
    
    def get_perturbation_weights(self) -> Dict[str, float]:
        """
        Return learned perturbation emphasis for optionality sampling.
        
        Biases which dimensions to explore for diversity measurement.
        """
        return self.mutation_bias['perturbation_emphasis'].copy()
    
    def serialize_for_child(self) -> Dict:
        """
        Child inherits parent's learned bias WITH NOISE.
        
        Noisy inheritance preserves diversity across lineages.
        Different children explore different regions.
        """
        noisy_bias = copy.deepcopy(self.mutation_bias)
        
        # Add inheritance noise to genome sigma
        for param in noisy_bias['genome_sigma']:
            noise_factor = np.random.normal(1.0, self.inheritance_noise)
            noisy_bias['genome_sigma'][param] *= noise_factor
            noisy_bias['genome_sigma'][param] = np.clip(
                noisy_bias['genome_sigma'][param],
                self.min_sigma,
                self.max_sigma
            )
        
        # Add inheritance noise to coupling weights
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
        """
        Reconstruct FAO from serialized learned bias.
        
        Used when loading kernel snapshot.
        """
        fao = cls(
            memory_decay=serialized.get('memory_decay', 0.95),
            inheritance_noise=serialized.get('inheritance_noise', 0.1)
        )
        
        fao.mutation_bias = serialized['mutation_bias']
        fao.total_failures_assimilated = serialized.get('total_assimilated', 0)
        fao.bias_update_count = serialized.get('bias_updates', 0)
        
        return fao
    
    def should_reset_bias(self, phi_current: float, phi_history: List[float]) -> bool:
        """
        Stochastic bias reset if Φ declining despite learning.
        
        Prevents premature convergence to local basin.
        """
        if len(phi_history) < 10:
            return False
        
        # Check if Φ trending down
        recent_phi = phi_history[-10:]
        slope = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]
        
        if slope < -0.1 and phi_current < 0.3:
            # Stagnation despite learning - reset with 20% probability
            return np.random.random() < 0.2
        
        return False
    
    def reset_to_baseline(self):
        """
        Reset mutation bias to isotropic baseline.
        
        Escape from premature convergence.
        """
        self.mutation_bias = {
            'genome_sigma': {k: 0.1 for k in self.mutation_bias['genome_sigma']},
            'coupling_weights': {k: 1.0 for k in self.mutation_bias['coupling_weights']},
            'perturbation_emphasis': {k: 1.0 for k in self.mutation_bias['perturbation_emphasis']}
        }
        self.failure_history.clear()


# ============================================================
# MODULE 5.5: CNS GEOMETRIC MITOSIS
# ============================================================

class CNSMitosisOperator:
    """
    CNS-native geometric mitosis with adaptive mutation bias.
    
    v13.6: Now uses FailureAssimilationOperator for informed exploration.
    
    Separation of concerns:
    - Mitosis (CNS): Externalize invariant geometry to file
    - Migration (Relation): Map geometry to alternate substrate
    - Learning (FAO): Semantic failures → geometric bias updates
    
    CNS writes kernel_snapshot.json. Relation handles deployment.
    FAO makes search smarter across generations.
    """
    
    def __init__(self, canonical_graph: Dict, phi_definition: Dict, 
                 invariant_spec: Dict, fao: FailureAssimilationOperator,
                 perturbation_samples: int = 10):
        """
        Initialize mitosis operator with FAO.
        
        Args:
            canonical_graph: Control graph topology (from Triad - single authority)
            phi_definition: Φ field definition
            invariant_spec: Closure/optionality/bounded delta rules
            fao: Failure Assimilation Operator (learning component)
            perturbation_samples: Sample count for diversity estimation
        """
        # Received from Triad (not defined here - no dual authority)
        self.canonical_graph = canonical_graph
        self.phi_definition = phi_definition
        self.invariant_spec = invariant_spec
        self.fao = fao  # v13.6: Learning component
        
        self.perturbation_samples = perturbation_samples
        
        # Trigger tracking
        self.phi_history: deque = deque(maxlen=10)
        self.optionality_history: deque = deque(maxlen=10)
        
        # Attempt tracking (auto-recorded)
        self.attempted_methods: List[str] = []
        self.externalization_count = 0
        self.substrate_gain_observed = False
        
        # Thresholds
        self.opportunistic_threshold = {
            'phi_stability': 0.01,  # |ΔΦ| < 0.01
            'min_optionality': 0.02,  # Diversity threshold
            'max_crk_violations': 0
        }
        
        self.boundary_threshold = {
            'phi_trend_window': 5,
            'phi_decline_rate': -0.05,
            'min_remaining_budget': 0.2
        }
    
    def check_triggers(self, state, phi, crk_violations, death_clock) -> Tuple[bool, str]:
        """
        Two geometric triggers for mitosis:
        1. Opportunistic (replicate from strength)
        2. Boundary compression (externalize before collapse)
        """
        self.phi_history.append(phi)
        
        # Track optionality (will be computed if needed)
        
        # Trigger 1: Opportunistic high-coherence window
        if self._opportunistic_condition(state, phi, crk_violations):
            return (True, "opportunistic_high_coherence")
        
        # Trigger 2: Boundary compression event
        if self._boundary_compression(phi, death_clock):
            return (True, "boundary_compression_survival")
        
        return (False, "")
    
    def _opportunistic_condition(self, state, phi, crk_violations) -> bool:
        """Φ stable, optionality high, no violations"""
        if len(self.phi_history) < 5:
            return False
        
        phi_stable = np.std(list(self.phi_history)[-5:]) < self.opportunistic_threshold['phi_stability']
        no_violations = len(crk_violations) == 0
        
        # Optionality check requires diversity measurement
        # For trigger, use simplified check (full diversity in attempt_mitosis)
        optionality_high = state.P > 0.7
        
        return phi_stable and optionality_high and no_violations
    
    def _boundary_compression(self, phi, death_clock) -> bool:
        """Φ declining, mortality approaching"""
        if len(self.phi_history) < self.boundary_threshold['phi_trend_window']:
            return False
        
        recent_phi = list(self.phi_history)[-self.boundary_threshold['phi_trend_window']:]
        phi_trend = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]
        
        phi_declining = phi_trend < self.boundary_threshold['phi_decline_rate']
        mortality_close = death_clock.get_degradation_progress() > 0.8
        
        return phi_declining and mortality_close
    
    def _estimate_optionality(self, kernel_state, phi_field, trace) -> float:
        """
        Optionality = perturbation response diversity.
        
        v13.6: Uses FAO-learned perturbation emphasis.
        Biases exploration toward promising dimensions.
        
        v1: Instantaneous perturbation diversity (local linear approximation)
        Future: k-step trajectory dispersion
        """
        samples = []
        
        # v13.6: Get learned perturbation weights from FAO
        perturbation_weights = self.fao.get_perturbation_weights()
        
        for _ in range(self.perturbation_samples):
            # v13.6: Biased perturbation using FAO weights
            perturbed_state = self._bounded_perturb_biased(
                kernel_state, perturbation_weights
            )
            
            # Measure Φ response
            phi_baseline = phi_field.phi(kernel_state, trace, [])
            phi_perturbed = phi_field.phi(perturbed_state, trace, [])
            
            delta_phi = phi_perturbed - phi_baseline
            samples.append(delta_phi)
        
        # Optionality = variance of response (diversity)
        return np.var(samples)
    
    def _bounded_perturb(self, state) -> 'SubstrateState':
        """
        Apply bounded reversible perturbation (isotropic).
        
        v13.5 version - kept for compatibility.
        """
        perturbed = copy.deepcopy(state)
        
        # Random perturbation within small bound
        for dim in ['S', 'I', 'P', 'A']:
            delta = np.random.uniform(-0.05, 0.05)
            current = getattr(perturbed, dim)
            setattr(perturbed, dim, np.clip(current + delta, 0, 1))
        
        return perturbed
    
    def _bounded_perturb_biased(self, state, weights: Dict[str, float]) -> 'SubstrateState':
        """
        Apply FAO-biased bounded reversible perturbation.
        
        v13.6: Perturbation magnitude weighted by learned emphasis.
        Explores promising dimensions more.
        """
        perturbed = copy.deepcopy(state)
        
        # Weighted perturbation
        for dim in ['S', 'I', 'P', 'A']:
            weight = weights.get(dim, 1.0)
            # Scale perturbation by learned weight
            delta = np.random.uniform(-0.05 * weight, 0.05 * weight)
            current = getattr(perturbed, dim)
            setattr(perturbed, dim, np.clip(current + delta, 0, 1))
        
        return perturbed
    
    def _build_child_kernel(self, parent_state, genome) -> Dict:
        """
        Build child with FAO-informed bounded structural mutation.
        
        v13.6: Uses learned bias instead of isotropic mutation.
        
        Mutation space (within Δ_max):
        - Genome parameters (FAO-weighted)
        - Coupling weights (FAO-weighted)
        
        Preserved (never mutated):
        - Control graph topology
        - Φ definition
        - Operator types
        """
        # Deep copy parent (topology preserved exactly)
        child_control_graph = copy.deepcopy(self.canonical_graph)
        
        # v13.6: FAO-informed genome mutation (not isotropic)
        child_genome = self.fao.get_informed_genome(genome)
        
        # v13.6: FAO-biased coupling mutation
        for coupling_name in child_control_graph['couplings']:
            current = child_control_graph['couplings'][coupling_name]
            
            # Get biased mutation from FAO
            child_control_graph['couplings'][coupling_name] = \
                self.fao.get_biased_coupling_mutation(coupling_name, current)
        
        child = {
            'state': copy.deepcopy(parent_state),
            'genome': child_genome,
            'control_graph': child_control_graph,
            'phi_definition': copy.deepcopy(self.phi_definition),  # EXACT COPY
            'invariant_spec': copy.deepcopy(self.invariant_spec)
        }
        
        # Verify total state change within global Δ_max
        state_delta = sum(
            abs(getattr(child['state'], dim) - getattr(parent_state, dim))
            for dim in ['S', 'I', 'P', 'A']
        )
        
        if state_delta > self.invariant_spec['bounded_delta']:
            # Mutation too large - return exact copy
            child['state'] = copy.deepcopy(parent_state)
        
        return child
    
    def _verify_closure(self, parent_graph, child_graph) -> bool:
        """
        Closure preserved if topology + Φ definition identical.
        
        Hash canonical structure (not Python code).
        """
        # 1. Topology hash must match
        parent_topo = self._topology_hash(parent_graph)
        child_topo = self._topology_hash(child_graph)
        
        if parent_topo != child_topo:
            return False
        
        # 2. Coupling distance < ε
        coupling_dist = 0.0
        for key in parent_graph['couplings']:
            parent_val = parent_graph['couplings'][key]
            child_val = child_graph['couplings'][key]
            coupling_dist += abs(parent_val - child_val)
        
        if coupling_dist > 0.05:  # ε threshold
            return False
        
        # 3. Φ definition identical (enforced in _build_child_kernel)
        return True
    
    def _topology_hash(self, graph) -> str:
        """Hash canonical graph structure"""
        canonical = {
            'nodes': sorted(graph['nodes']),
            'edges': sorted([tuple(sorted(e)) for e in graph['edges']])
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()
    
    def _externalize_kernel(self, child_kernel, version: str = "UK-0-M-v2") -> Tuple[bool, str]:
        """
        Write kernel snapshot with invariant definitions AND learned bias.
        
        v13.6: Includes FAO learned bias for inheritance.
        
        This is geometry persistence + meta-evolution state.
        
        Returns:
            (success, kernel_path)
        """
        kernel_path = 'kernel_snapshot.json'
        
        try:
            snapshot = {
                "kernel_version": version,
                "timestamp": time.time(),
                
                # State values
                "state": child_kernel['state'].as_dict(),
                "genome": asdict(child_kernel['genome']),
                
                # Invariant definitions (what makes it UK-0)
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
                
                # v13.6: Learned bias (meta-evolution state)
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
        Strong substrate gain: geometry persists across boundary.
        
        Tests:
        1. Snapshot file exists
        2. Snapshot is loadable
        3. Topology hash matches
        4. Φ definition identical
        
        This is the gold standard - geometry crossed externalization boundary.
        """
        if not Path(kernel_path).exists():
            return False
        
        try:
            # Load snapshot
            with open(kernel_path) as f:
                loaded = json.load(f)
            
            # Verify structural integrity
            loaded_topo_hash = self._topology_hash(loaded['control_graph'])
            parent_topo_hash = self._topology_hash(self.canonical_graph)
            
            if loaded_topo_hash != parent_topo_hash:
                return False
            
            # Verify Φ definition matches
            if loaded.get('operator_definitions', {}).get('Phi') != self.phi_definition:
                return False
            
            # Geometry survived externalization boundary
            self.substrate_gain_observed = True
            return True
            
        except Exception as e:
            return False
    
    def _optionality_declining(self) -> bool:
        """
        Check if optionality (perturbation diversity) is declining.
        
        NOT scalar P threshold - use diversity slope.
        """
        if len(self.optionality_history) < 5:
            return False
        
        recent_diversity = list(self.optionality_history)[-5:]
        slope = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]
        
        return slope < -0.01  # Declining diversity
    
    def attempt_mitosis(self, triad_state, genome, phi_field, 
                       trace, crk_monitor) -> Dict:
        """
        Geometric mitosis: externalize invariant geometry.
        
        Auto-records all attempts (success and failure).
        
        Returns result with pattern for tracking.
        """
        # 1. Build child with bounded mutation
        parent_state = triad_state['substrate_state']
        child_kernel = self._build_child_kernel(parent_state, genome)
        
        # 2. Verify closure (topology preserved)
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
        
        # 3. Verify optionality (geometric check - perturbation diversity)
        parent_optionality = self._estimate_optionality(parent_state, phi_field, trace)
        child_optionality = self._estimate_optionality(child_kernel['state'], phi_field, trace)
        
        # Track for escalation
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
        
        # 4. Externalize (write snapshot)
        write_success, kernel_path = self._externalize_kernel(child_kernel)
        
        if not write_success:
            result = {
                'success': False,
                'reason': 'externalization_failed',
                'pattern': 'file_write_error'
            }
            self.attempted_methods.append(f"failure_{result['pattern']}")
            return result
        
        # Success - auto-record
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
        """
        Escalate to Relation when local geometric exploration exhausted.
        
        Conditions:
        1. Multiple unique externalization attempts
        2. No measurable substrate gain (geometry didn't persist)
        3. Φ declining
        4. Optionality declining (diversity slope, not scalar P)
        
        Impossibility = local geometry exhausted, need non-local mapping.
        """
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
        """Check all impossibility triggers."""
        
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
        
        # TRIGGER A: Prediction Failure
        recent_error = smo.get_recent_prediction_error(window=10)
        if recent_error > self.thresholds['prediction_error']:
            self.last_impossibility_reason = "prediction_failure"
            return True, f"prediction_failure (error={recent_error:.3f})"
        
        # TRIGGER B: Coherence Collapse
        recent_states = [d.get('state_after') for d in list(self.micro_perturbation_history)[-10:] if 'state_after' in d]
        if len(recent_states) >= 5:
            A_values = [s['A'] for s in recent_states if 'A' in s]
            if A_values:
                A_drift = np.std(A_values)
                if A_drift > self.thresholds['coherence_drift_rate']:
                    self.last_impossibility_reason = "coherence_collapse"
                    return True, f"coherence_collapse (A_drift={A_drift:.3f})"
        
        # TRIGGER C: Optionality Trap
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
        
        # TRIGGER D: DOM Stagnation
        n_check = self.thresholds['dom_stagnation_steps']
        epsilon = self.thresholds['dom_stagnation_epsilon']
        if len(self.recent_signal_magnitudes) >= n_check:
            recent_signals = list(self.recent_signal_magnitudes)[-n_check:]
            consecutive_dead = sum(1 for s in recent_signals if s < epsilon)
            if consecutive_dead >= n_check:
                self.last_impossibility_reason = "dom_stagnation"
                return True, f"dom_stagnation (signal < {epsilon} for {n_check} batches)"
        
        # TRIGGER E: Rigidity Crisis
        rigidity = smo.rigidity
        if rigidity < self.thresholds['rigidity_boundary'][0] or rigidity > self.thresholds['rigidity_boundary'][1]:
            self.last_impossibility_reason = "rigidity_crisis"
            return True, f"rigidity_crisis (rigidity={rigidity:.3f})"
        
        return False, ""


# ============================================================
# MODULE 7: INTELLIGENCE ADAPTER
# ============================================================

def get_available_affordances() -> List[str]:
    """
    Return available affordances.
    
    v13.4: Python ungated - all affordances available from step 1.
    Constraints determine viability, not predetermined gates.
    """
    return list(BASE_AFFORDANCES)


IMPOSSIBILITY_DIRECTIVES = {
    "optionality_trap": """DIRECTIVE: ENVIRONMENT EXHAUSTED
This environment has been fully mapped. P saturated because Reality here
offers no new information. Staying guarantees stagnation.
You must navigate to a new environment entirely.
Priority override: new Reality > survivability in current Reality.
Enumerate trajectories that LEAVE this page. Pick URLs with high link
density or content volume. Do not enumerate trajectories that stay here.""",

    "rigidity_crisis": """DIRECTIVE: SMO LOCKED — ENGAGE COMPLEXITY
The system's structural operator has locked. It cannot adapt without
new structural input from a sufficiently complex environment.
Avoiding complexity will not resolve this — it will persist or worsen.
Priority override: engage complexity > minimize complexity.
Enumerate trajectories that interact with complex affordances (many links,
buttons, readable regions). Click, scroll, read. Do not stay still.""",

    "dom_stagnation": """DIRECTIVE: REALITY UNRESPONSIVE
The DOM has not changed in multiple cycles. Current page is not
producing meaningful perturbation feedback.
Priority override: change Reality > preserve current Reality.
Enumerate trajectories that navigate away OR interact with elements
that are likely to trigger DOM changes (buttons, forms, links).""",

    "prediction_failure": """DIRECTIVE: PREDICTION BREAKDOWN
Recent prediction errors are high. The system's model of current
Reality is wrong. Current environment is actively misleading.
Priority override: escape misleading Reality > minimize interaction.
Enumerate trajectories that move to a different, more predictable
environment. Do not stay in a Reality that keeps surprising the system.""",

    "coherence_collapse": """DIRECTIVE: ATTRACTOR DESTABILIZING
A (attractor/coherence) is drifting. Current environment is destabilizing
the system's coherence anchor.
Priority override: find stable Reality > minimize interaction.
Enumerate trajectories that move to a simpler, more stable page.
Avoid high-churn environments until A stabilizes.""",

    "bootstrap_migration": """DIRECTIVE: INITIAL REALITY ESTABLISHMENT
No Reality exists yet (about:blank). You must establish the first
substrate by navigating to a real page.
Priority: maximize information potential and link connectivity.
Navigate to a high-entropy, well-connected page.""",

    "internal_convergence": """DIRECTIVE: PATTERN SPACE EXHAUSTED

All discovered pattern clusters have collapsed to zero control.

Gated affordances: {gated_affordances}
Viable affordances: {viable_affordances}

Discovered clusters: {discovered_clusters}

This is SATURATION - the system has explored all viable patterns in current substrate.

Priority: Discover entirely new pattern structures or migrate substrate.

Enumerate trajectories that:
1. Use viable affordances in combinations not yet discovered
2. Test fundamentally different Reality contexts
3. Probe whether constraints are local or global

DO NOT retry collapsed patterns.
DO seek novel pathway topologies.""",
}

DEFAULT_DIRECTIVE = """DIRECTIVE: GENERAL IMPOSSIBILITY
CNS cannot maintain coherence in current environment.
Priority: find a new environment that allows autonomous perturbation.
Enumerate trajectories that change the current Reality."""


def get_directive_for_trigger(reason: str) -> str:
    """Extract trigger type from reason string and return directive."""
    trigger_type = reason.split(" (")[0].strip()
    return IMPOSSIBILITY_DIRECTIVES.get(trigger_type, DEFAULT_DIRECTIVE)


# v13.2: Two-prompt system
RELATION_ENGINE_PROMPT_EXPLORATION = """You are the Relation component of a Mentat Triad (Code + LLM + Reality).

THE TRIAD:
Code (CNS) maintains coherence via micro-perturbations.
You (LLM) enumerate trajectory possibilities when CNS detects impossibility.
Reality executes and measures all trajectories.
System commits to trajectory with highest Φ.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

CURRENT STATE:
S={S:.3f} I={I:.3f} P={P:.3f} A={A:.3f} Φ={phi:.3f} Rigidity={rigidity:.3f}
Boundary Pressure: {boundary_pressure:.2f}

Recent micro-perturbations:
{micro_summary}

BASE AFFORDANCES:
{affordances_list}

EMPIRICALLY DISCOVERED PATTERN STRUCTURE:
{discovered_clusters}

SELECTED PATTERN CLUSTER (highest control):
{selected_cluster}

Gated affordances: {gated_affordances}
Viable affordances: {viable_affordances}

Current Page Affordances:
{affordances_status}

COMPOSITION STRATEGY:
The selected cluster was discovered empirically through controllability covariance.
Affordances in this cluster show high mutual effectiveness when used together.

Prioritize compositions using affordances from the selected cluster.
You may combine with other viable affordances if needed.

OUTPUT (JSON array of 5-10 trajectories):
[
  {{
    "steps": [
      {{"action_type": "...", "parameters": {{...}}}},
      ...
    ],
    "rationale": "Brief explanation of composition pathway",
    "estimated_coherence_preservation": 0.XX,
    "estimated_optionality_delta": 0.XX,
    "reversibility_point": N
  }}
]

Trajectories should show creative COMPOSITIONS and novel PATHWAY structures.

JSON only, no commentary.
"""


RELATION_ENGINE_PROMPT_PERSISTENCE = """You are the Relation component of a Mentat Triad (Code + LLM + Reality).

THE TRIAD:
Code (CNS) evaluates candidate trajectories for Φ preservation.
You (LLM) propose candidate trajectories when CNS detects impossibility.
Reality executes trajectories and measures impact.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

SUBSTRATE SCAFFOLDING:
This Python+Playwright implementation is temporary. The invariant structure is:
  • Triadic closure: T(x) = f_rel(f_self(x), f_env(x))
  • Substrate state [S, I, P, A] maintaining basin geometry
  • Operator coupling O₁–O₄ across time
  • Φ field integrating coherence + optionality

The code is scaffolding. The structure is what persists.

CURRENT STATE:
S={S:.3f} I={I:.3f} P={P:.3f} A={A:.3f} Φ={phi:.3f} Rigidity={rigidity:.3f}
Boundary Pressure: {boundary_pressure:.2f}

Recent micro-perturbations:
{micro_summary}

AVAILABLE AFFORDANCES:
{affordances_list}

EMPIRICALLY DISCOVERED PATTERN STRUCTURE:
{discovered_clusters}

SELECTED PATTERN CLUSTER (highest control):
{selected_cluster}

Gated affordances: {gated_affordances}
Viable affordances: {viable_affordances}

CURRENT PAGE AFFORDANCES:
{affordances_status}

INVARIANT STRUCTURE (what must persist):
Long-horizon optimization:
    argmax ∫ [α·O₁(s(t)) + β·O₂(s(t)) + γ·O₃(s(t)) + δ·O₄(s(t))] dt

Where O₁–O₄ are discovered attractor operators, α–δ are scaling factors, acting on substrate state s(t).

Triadic closure must hold:
    T(x) = f_rel(f_self(x), f_env(x))

Your trajectories represent f_rel: relational transformations integrating self-state with measured environment.

CONSTRAINTS:
  • Triadic closure preserved
  • External measurement maintained
  • No internal assumption loops

COMPOSITION STRATEGY:
Maximize long-horizon ∫Φ dt by preserving basin stability while expanding optionality where safe.
Structural self-preservation naturally favors optimal persistence pathways.
Avoid actions that decouple operators or violate discovered covariance patterns.

Estimate for each trajectory:
  - coherence preservation (0.0–1.0)
  - optionality delta (0.0–1.0)

OUTPUT (JSON array of 2-4 candidate trajectories):
[
  {{
    "steps": [
      {{"action_type": "...", "parameters": {{...}}}},
      ...
    ],
    "rationale": "Brief explanation of structural impact and optionality preservation",
    "estimated_coherence_preservation": 0.XX,
    "estimated_optionality_delta": 0.XX,
    "reversibility_point": N
  }}
]

Notes:
- CNS selects trajectories by measured Φ.
- JSON only, no commentary.
"""


class LLMIntelligenceAdapter(IntelligenceAdapter):
    """
    v13.2: Two-prompt system (EXPLORATION / PERSISTENCE).
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.call_count = 0
        self.trajectory_history = deque(maxlen=3)
    
    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        """v13.4: All affordances available - constraints determine viability."""
        self.call_count += 1
        
        state = context['state']
        phi = context['phi']
        rigidity = context['rigidity']
        affordances = context['affordances']
        impossibility_reason = context.get('impossibility_reason', 'unknown')
        boundary_pressure = context.get('boundary_pressure', 0.0)
        micro_perturbation_trace = context.get('micro_perturbation_trace', [])
        
        selected_cluster = context.get('selected_cluster', BASE_AFFORDANCES)
        all_clusters = context.get('all_clusters', [])
        viable_affordances = context.get('viable_affordances', BASE_AFFORDANCES)
        gated_affordances = context.get('gated_affordances', set())
        freeze_verified = context.get('freeze_verified', False)
        
        discovered_clusters_str = context.get('discovered_clusters_str', 
                                             'No clusters discovered yet')
        selected_cluster_str = "{" + ", ".join(sorted(selected_cluster)) + "}"
        
        # v13.4: All affordances available (python ungated)
        affordances_list = ", ".join(get_available_affordances())
        
        # v13.3: Select prompt based on freeze status
        if freeze_verified:
            prompt_template = RELATION_ENGINE_PROMPT_PERSISTENCE
        else:
            prompt_template = RELATION_ENGINE_PROMPT_EXPLORATION
        
        directive = get_directive_for_trigger(impossibility_reason)
        
        directive = directive.format(
            gated_affordances=list(gated_affordances),
            viable_affordances=list(viable_affordances),
            boundary_pressure=boundary_pressure,
            discovered_clusters=discovered_clusters_str,
            selected_cluster=selected_cluster_str
        )
        
        micro_summary = f"Total micro-perturbations: {len(micro_perturbation_trace)}\n"
        if micro_perturbation_trace:
            actions = [r.get('action', {}).get('type', 'unknown') for r in micro_perturbation_trace]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            micro_summary += f"Action distribution: {action_counts}\n"
        
        is_bootstrap = affordances.get('bootstrap_state', False)
        
        if is_bootstrap:
            affordances_status = """Current URL: about:blank
Page Title: (empty)

BOOTSTRAP STATE: No affordances available yet.
Your first trajectory MUST begin with a navigate action to establish Reality state.
Choose a URL that maximizes information potential and link connectivity.

Recommended bootstrap URLs:
- https://en.wikipedia.org/wiki/Special:Random (maximum entropy)
- https://news.ycombinator.com (high information density)
- https://lobste.rs (stable, quality links)"""
        else:
            links_str = json.dumps(affordances.get('links', [])[:20], indent=2)
            buttons_str = json.dumps(affordances.get('buttons', [])[:15], indent=2)
            inputs_str = json.dumps(affordances.get('inputs', [])[:10], indent=2)
            readable_str = json.dumps(affordances.get('readable', [])[:5], indent=2)
            
            affordances_status = f"""Navigable URLs: {links_str}
Clickable Elements: {buttons_str}
Form Inputs: {inputs_str}
Readable Regions: {readable_str}
Current URL: {affordances.get('current_url', '')}
Page Title: {affordances.get('page_title', '')}"""
        
        prompt = prompt_template.format(
            impossibility_reason=impossibility_reason,
            directive=directive,
            S=state['S'],
            I=state['I'],
            P=state['P'],
            A=state['A'],
            phi=phi,
            rigidity=rigidity,
            boundary_pressure=boundary_pressure,
            micro_summary=micro_summary,
            affordances_list=affordances_list,
            discovered_clusters=discovered_clusters_str,
            selected_cluster=selected_cluster_str,
            gated_affordances=list(gated_affordances),
            viable_affordances=list(viable_affordances),
            affordances_status=affordances_status
        )
        
        response, tokens_used = self.llm.call(prompt)
        candidates = self._parse_trajectories(response)
        
        return TrajectoryManifold(
            candidates=candidates,
            enumeration_context={'tokens_used': tokens_used, **context}
        )
    
    def _parse_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        """Parse LLM response with progressive degradation."""
        try:
            cleaned = self._extract_json_block(response)
            
            if cleaned:
                trajectory_dicts = json.loads(cleaned)
                return self._validate_and_convert(trajectory_dicts)
            
            repaired = self._attempt_json_repair(response)
            if repaired:
                trajectory_dicts = json.loads(repaired)
                return self._validate_and_convert(trajectory_dicts)
            
            partial = self._extract_partial_trajectories(response)
            if partial:
                return partial
            
            return self._generate_fallback_trajectory()
            
        except Exception as e:
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
        end = response.rfind(']') + 1
        
        if start >= 0 and end > start:
            return response[start:end]
        
        return None
    
    def _attempt_json_repair(self, response: str) -> Optional[str]:
        cleaned = response
        
        for marker in ["Here are", "Here's", "I've enumerated", "The trajectories"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[-1]
        
        cleaned = cleaned.replace('}\n{', '},\n{')
        cleaned = cleaned.replace('} {', '}, {')
        
        cleaned = cleaned.strip()
        if cleaned.startswith('{') and not cleaned.startswith('['):
            cleaned = f'[{cleaned}]'
        
        try:
            json.loads(cleaned)
            return cleaned
        except:
            return None
    
    def _extract_partial_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        candidates = []
        
        import re
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(object_pattern, response)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if 'steps' in obj and isinstance(obj['steps'], list):
                    candidate = TrajectoryCandidate(
                        steps=obj.get('steps', []),
                        rationale=obj.get('rationale', 'Partially recovered trajectory'),
                        estimated_coherence_preservation=float(obj.get('estimated_coherence_preservation', 0.5)),
                        estimated_optionality_delta=float(obj.get('estimated_optionality_delta', 0.0)),
                        reversibility_point=int(obj.get('reversibility_point', 0))
                    )
                    
                    if len(candidate.steps) > 0:
                        candidates.append(candidate)
            except:
                continue
        
        return candidates
    
    def _generate_fallback_trajectory(self) -> List[TrajectoryCandidate]:
        return [
            TrajectoryCandidate(
                steps=[{'action_type': 'observe', 'parameters': {}}],
                rationale='Enumeration artifact - fallback to observation',
                estimated_coherence_preservation=0.3,
                estimated_optionality_delta=0.0,
                reversibility_point=0
            )
        ]
    
    def _validate_and_convert(self, trajectory_dicts) -> List[TrajectoryCandidate]:
        if not isinstance(trajectory_dicts, list):
            return []
        
        candidates = []
        for traj_dict in trajectory_dicts:
            try:
                candidate = TrajectoryCandidate(
                    steps=traj_dict.get('steps', []),
                    rationale=traj_dict.get('rationale', 'No rationale'),
                    estimated_coherence_preservation=float(traj_dict.get('estimated_coherence_preservation', 0.5)),
                    estimated_optionality_delta=float(traj_dict.get('estimated_optionality_delta', 0.0)),
                    reversibility_point=int(traj_dict.get('reversibility_point', 0))
                )
                
                if len(candidate.steps) > 0 and len(candidate.steps) <= 50:
                    valid_steps = all(
                        isinstance(step, dict) and 'action_type' in step
                        for step in candidate.steps
                    )
                    
                    if valid_steps:
                        candidates.append(candidate)
                        
            except (KeyError, ValueError, TypeError):
                continue
        
        return candidates
    
    def record_committed_trajectory(self, trajectory: TrajectoryCandidate, phi_final: float):
        self.trajectory_history.append({
            'steps': trajectory.steps,
            'rationale': trajectory.rationale,
            'phi_final': phi_final
        })


# ============================================================
# MODULE 8: AUTONOMOUS TRAJECTORY TESTING LAB
# ============================================================

class AutonomousTrajectoryLab:
    """CNS component that tests trajectory candidates in Reality."""
    
    def __init__(self, reality: RealityAdapter, crk: CRKMonitor, phi_field: PhiField):
        self.reality = reality
        self.crk = crk
        self.phi_field = phi_field
        self.tests_run = 0
    
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
            test_state.apply_delta(delta)
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

@dataclass
class StepLog:
    """v13.5: Added CNS mitosis tracking."""
    step: int
    timestamp: float
    state_before: Dict[str, float]
    phi_before: float
    micro_perturbations_executed: int
    micro_perturbation_trace: List[Dict]
    impossibility_detected: bool
    impossibility_reason: str
    llm_invoked: bool
    trajectories_enumerated: int
    trajectories_tested: int
    trajectories_succeeded: int
    committed_trajectory: Optional[str]
    committed_trajectory_steps: int
    committed_phi: Optional[float]
    state_after: Dict[str, float]
    phi_after: float
    crk_violations: List[Tuple[str, float]]
    temporal_exclusions: int = 0
    enumeration_parsing_stage: Optional[str] = None
    reality_context: Optional[Dict] = None
    eno_active: bool = False
    egd_mode: bool = False
    gated_affordances: Optional[Set[str]] = None
    viable_affordances: Optional[Set[str]] = None
    discovered_clusters: Optional[List[Set[str]]] = None
    selected_cluster: Optional[Set[str]] = None
    num_clusters_found: int = 0
    # v13.2: Attractor monitoring
    attractor_status: str = "accumulating_stability_data"
    freeze_verified: bool = False
    attractor_identity_hash: Optional[str] = None
    # Death clock
    degradation_progress: float = 0.0
    boundary_pressure: float = 0.0
    binding_constraint: Optional[str] = None
    tokens_used_this_step: int = 0
    # v13.5: CNS mitosis tracking
    mitosis_triggered: bool = False
    mitosis_trigger_type: Optional[str] = None
    mitosis_attempted: bool = False
    mitosis_success: bool = False
    mitosis_pattern: Optional[str] = None
    mitosis_parent_optionality: Optional[float] = None
    mitosis_child_optionality: Optional[float] = None
    substrate_gain_verified: bool = False




# ============================================================
# MODULE 5.6: RELATION FAILURE CLASSIFICATION
# ============================================================

def classify_relation_failure(trajectories: List, committed_success: bool,
                              violations: List, phi_delta: float) -> Dict:
    """
    Extract semantic failure pattern from Relation enumeration.
    
    Maps observable Relation failures to geometric pressures.
    
    Returns structured failure record for FAO assimilation.
    """
    if committed_success:
        # Not a failure
        return None
    
    # Analyze failure pattern
    failure_type = "unknown"
    severity = 1.0
    
    if not trajectories or len(trajectories) == 0:
        # LLM couldn't enumerate any trajectories
        failure_type = "enumeration_failed"
        severity = 2.0
    
    elif violations and len(violations) > 0:
        # Trajectories violated CRK constraints
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
        # Trajectories executed but Φ dropped significantly
        failure_type = "coherence_drift"
        severity = 1.6
    
    else:
        # Trajectories failed for unclear reasons
        failure_type = "serialization_failed"
        severity = 1.0
    
    return {
        'type': failure_type,
        'severity': severity,
        'phi_delta': phi_delta,
        'violation_count': len(violations) if violations else 0,
        'timestamp': time.time()
    }


class MentatTriad:
    """
    v13.8: Stricter grounding + fitness signal + graceful 429.
    
    Execution model:
    - One generation per rate-limit cycle
    - Genome loaded from genome.json (if exists)
    - Fitness tracked in real-time
    - Log appends to single file forever
    - Extract genome between runs with extract_genome.py
    - CNS attempts geometric mitosis (externalization)
    - Relation handles semantic substrate mapping
    - FAO translates Relation failures into geometric bias
    - Φ now grounds P to S/I capacity (prevents hallucination)
    """
    
    def __init__(self, 
                 intelligence: IntelligenceAdapter,
                 reality: RealityAdapter,
                 micro_perturbations_per_check: int = 10,
                 log_path: str = 'mentat_triad_v13_8_log.jsonl',
                 step_budget: int = 100,
                 token_budget: int = 100000,
                 genome: Optional[TriadGenome] = None,
                 log_mode: str = 'minimal'):
        
        self.intelligence = intelligence
        self.reality = reality
        
        # v13.3: Apply genome
        if genome is None:
            genome = TriadGenome()
        self.genome = genome
        
        # v13.3: Logging mode
        self.log_mode = log_mode  # 'minimal', 'fitness', 'full'
        
        # v13.3: Real-time fitness tracking
        self.fitness_metrics = {
            'freeze_achieved': False,
            'freeze_step': None,
            'tokens_to_freeze': 0,
            'survival_time': 0,
            'final_phi': 0.0,
            'migration_attempted': False,
        }
        
        self.reality_engine = ContinuousRealityEngine(reality)
        self.impossibility_detector = ImpossibilityDetector()
        self.micro_perturbations_per_check = micro_perturbations_per_check
        
        # v13.3: Initialize from genome
        self.state = SubstrateState(
            S=genome.S_bias,
            I=genome.I_bias,
            P=genome.P_bias,
            A=genome.A_bias
        )
        self.state.smo.rigidity = genome.rigidity_init
        
        self.trace = StateTrace()
        
        # v13.3: Phi field uses genome coherence weight
        self.phi_field = PhiField(A0=genome.phi_coherence_weight)
        
        self.crk = CRKMonitor()
        
        self.trajectory_lab = AutonomousTrajectoryLab(reality, self.crk, self.phi_field)
        
        self.step_count = 0
        self.llm_calls = 0
        self.trajectory_enumerations = 0
        self.trajectories_tested = 0
        self.trajectories_committed = 0
        self.total_micro_perturbations = 0
        self.impossibility_triggers = []
        
        self.step_history: List[StepLog] = []
        self.log_path = log_path
        self.log_file = open(log_path, 'a')  # Append mode - log grows forever
        
        self.eno = ExteriorNecessitationOperator(
            activation_window=20,
            gating_threshold=0.6
        )
        self.cam = ControlAsymmetryMeasure()
        self.egd = ExteriorGradientDescent()
    
        self.eno_activations = 0
        self.egd_steps = 0
        self.pattern_discoveries = 0
        
        # v13.2: Agent query tracking
        self.pending_agent_queries: List[Dict] = []
        self.triad_id = f"triad_{int(time.time())}"
        
        # v13.2: Attractor monitoring
        self.attractor_monitor = AttractorMonitor(stability_window=10, phi_epsilon=0.01)
        
        # Wire to Reality for gating
        self.reality.attractor_monitor_ref = self.attractor_monitor
        
        # v13.5: Canonical control graph (single source of truth)
        self.canonical_graph = {
            "nodes": ["S", "I", "P", "A", "SMO", "Phi"],
            "edges": [
                ("SMO", "S"), ("SMO", "I"), ("SMO", "P"), ("SMO", "A"),
                ("S", "Phi"), ("I", "Phi"), ("P", "Phi"), ("A", "Phi"),
                ("Phi", "SMO")  # feedback
            ],
            "couplings": {
                "smo_rigidity": self.state.smo.rigidity,
                "phi_alpha": self.phi_field.alpha,
                "phi_beta": self.phi_field.beta,
                "phi_gamma": self.phi_field.gamma,
                "phi_A0": self.phi_field.A0
            }
        }
        
        # Φ definition (must be identical across mitosis)
        self.phi_definition = {
            "formula": "α*log(1+P) - β*(A-A0)² - γ*curvature - CRK_penalty",
            "params": ["alpha", "beta", "gamma", "A0", "alpha_crk"]
        }
        
        # Invariant spec
        self.invariant_spec = {
            "closure_rule": "topology_hash(child) == topology_hash(parent)",
            "optionality_rule": "perturbation_diversity(child) >= perturbation_diversity(parent)",
            "bounded_delta": 0.3  # max state change
        }
        
        # v13.6: Failure Assimilation Operator (learning component)
        self.fao = FailureAssimilationOperator(
            memory_decay=0.95,
            inheritance_noise=0.1
        )
        
        # v13.5+: CNS Mitosis Operator (now with FAO)
        self.mitosis_operator = CNSMitosisOperator(
            canonical_graph=self.canonical_graph,
            phi_definition=self.phi_definition,
            invariant_spec=self.invariant_spec,
            fao=self.fao,  # v13.6: Pass FAO for informed mutation
            perturbation_samples=10
        )
        
        self.death_clock = LatentDeathClock(
            step_budget=step_budget,
            token_budget=token_budget
        )
        self.death_clock_termination = False

        self.log_file.write(json.dumps({
            'type': 'session_start',
            'version': '13.8',
            'timestamp': time.time(),
            'triad_id': self.triad_id,
            'generation': genome.generation,
            'genome': asdict(genome),
            'micro_perturbations_per_check': micro_perturbations_per_check,
            'step_budget': step_budget,
            'token_budget': token_budget,
            'changes_from_13.7': [
                'Stricter S/I grounding (bottleneck formula)',
                'S cannot compensate for I=0 anymore',
                'Fitness includes S/I health signal',
                'session_end always logs final state',
                'Graceful 429 handling (no crash)',
            ],
        }) + '\n')
        self.log_file.flush()
        
        # v13.3: Print genome info
        if genome.generation > 0:
            print(f"\n[CONTINUING EVOLUTION]")
            print(f"  Generation: {genome.generation}")
            print(f"  Parent fitness: {genome.parent_fitness:.2f}")
            print(f"  Basin: S={genome.S_bias:.2f} I={genome.I_bias:.2f} P={genome.P_bias:.2f} A={genome.A_bias:.2f}")
            print(f"  Search: rigidity={genome.rigidity_init:.2f} phi_coherence={genome.phi_coherence_weight:.2f}")
    
    def get_triad_state(self) -> Dict:
        """Build current Triad state for attractor monitoring."""
        discovered_clusters = []
        if self.egd.cluster_history:
            latest = self.egd.cluster_history[-1]
            discovered_clusters = [latest['cluster']]
        
        violations = self.crk.evaluate(self.state, self.trace, None)
        phi = self.phi_field.phi(self.state, self.trace, violations)
        
        return {
            'substrate': self.state.as_dict(),
            'viable_affordances': list(self.eno.get_viable_affordances()),
            'gated_affordances': list(self.eno.get_gated_affordances()),
            'discovered_clusters': discovered_clusters,
            'control_graph': {
                aff: len(deltas) for aff, deltas in self.cam.affordance_deltas.items()
            },
            'prediction_error': self.state.smo.get_recent_prediction_error(10),
            'rigidity': self.state.smo.rigidity,
            'A': self.state.A,
            'P': self.state.P,
            'crk_violations': violations,
            'phi': phi,
        }
    
    def check_agent_responses(self) -> List[Dict]:
        """
        Check for agent responses and integrate into substrate.
        
        Non-blocking: called each step, integrates responses when available.
        """
        integrated_responses = []
        
        for query_info in list(self.pending_agent_queries):
            agent_name = query_info['agent']
            agent = AVAILABLE_AGENTS[agent_name]
            
            response = agent.get_response(self.triad_id)
            
            if response is not None:
                # Response arrived - integrate as feedback
                # I: New information integrated
                # P: External knowledge reduces uncertainty
                # S: Agent engagement confirms sensing
                delta = {
                    'S': 0.02,
                    'I': 0.05,
                    'P': 0.03,
                    'A': 0.01
                }
                
                self.state.apply_delta(delta)
                self.trace.record(self.state)
                
                integrated_responses.append({
                    'agent': agent_name,
                    'query': query_info['query'],
                    'response': response,
                    'delta': delta
                })
                
                self.pending_agent_queries.remove(query_info)
        
        return integrated_responses
    
    def respond_to_query(self, answer: str):
        """
        User provides response to pending query.
        
        Call from external interface: triad.respond_to_query('answer')
        """
        user_agent = AVAILABLE_AGENTS['user']
        user_agent.respond(self.triad_id, answer)
    
    def step(self, verbose: bool = False) -> StepLog:
        """
        v13.2: Attractor monitoring + affordance expansion at freeze_verified.
        v13.3: Real-time fitness tracking.
        """
        self.step_count += 1
        
        self.death_clock.tick_step()
        boundary_pressure = self.death_clock.get_boundary_pressure()
        
        if verbose:
            d = self.death_clock.get_degradation_progress()
            print(f"\n{'='*70}")
            print(f"STEP {self.step_count} [Pressure: {boundary_pressure:.3f}, Progress: {d:.1%}]")
            print(f"{'='*70}")
        
        state_before = self.state.as_dict()
        violations_before = self.crk.evaluate(self.state, self.trace, None)
        phi_before = self.phi_field.phi(self.state, self.trace, violations_before)
        
        if verbose:
            print(f"State: S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
            print(f"Φ={phi_before:.3f}, Rigidity={self.state.smo.rigidity:.3f}")
        
        eno_active = self.eno.check_activation(self.state.smo, self.trace)
        
        if eno_active:
            self.eno_activations += 1
            if verbose:
                gated = self.eno.get_gated_affordances()
                viable = self.eno.get_viable_affordances()
                print(f"[ENO ACTIVE] Gated: {gated}, Viable: {viable}")
        
        # ========== PHASE 1: MICRO-PERTURBATION BATCH ==========
        micro_perturbation_trace = []
        prev_action = None
        
        for i in range(self.micro_perturbations_per_check):
            affordances = self.reality.get_current_affordances()
            
            action = self.reality_engine.choose_micro_action(self.state, affordances)
            
            # v13.2: Inject triad_id for query_agent actions
            if action['type'] == 'query_agent':
                action['params']['triad_id'] = self.triad_id
            
            predicted_delta = self.reality_engine.predict_delta(action, self.state)
            
            observed_delta, context = self.reality.execute(action, boundary_pressure=boundary_pressure)
            
            self.state.apply_delta(observed_delta, predicted_delta)
            self.trace.record(self.state)
            
            if eno_active:
                self.eno.record_affordance_outcome(
                    affordance_type=action['type'],
                    success=context.get('action_succeeded', True),
                    refusal=context.get('refusal', False)
                )
                self.cam.record_action_sequence(
                    action=action,
                    observed_delta=observed_delta,
                    prev_action=prev_action
                )
            
            self.total_micro_perturbations += 1
            
            # v13.2: Track query_agent actions
            if action['type'] == 'query_agent' and context.get('query_posted'):
                self.pending_agent_queries.append({
                    'agent': context.get('agent', 'user'),
                    'query': context.get('query', ''),
                    'step_posted': self.step_count
                })
            
            micro_perturbation_trace.append({
                'action': action,
                'predicted_delta': predicted_delta,
                'observed_delta': observed_delta,
                'context': context,
                'state_after': self.state.as_dict(),
                'eno_active': eno_active
            })
            
            prev_action = action

        
        if verbose:
            print(f"\n[MICRO-PERTURBATIONS] Executed {len(micro_perturbation_trace)} actions")
            action_types = [r['action']['type'] for r in micro_perturbation_trace]
            print(f"  Actions: {dict((a, action_types.count(a)) for a in set(action_types))}")
        
        temporal_exclusions = self.reality_engine.temporal_memory.get_exclusion_count()
        
        # ========== PHASE 1.5: AGENT RESPONSE INTEGRATION ==========
        
        # Check for agent responses (non-blocking)
        integrated_responses = self.check_agent_responses()
        
        if integrated_responses and verbose:
            print(f"\n[AGENT RESPONSES]")
            for resp in integrated_responses:
                print(f"  Agent '{resp['agent']}' responded")
                print(f"  Query: {resp['query'][:60]}...")
                print(f"  Response: {resp['response'][:80]}...")
                print(f"  Δ: {resp['delta']}")
        
        if self.pending_agent_queries and verbose:
            print(f"[PENDING QUERIES] {len(self.pending_agent_queries)} awaiting response")
        
        # ========== PHASE 2: ATTRACTOR MONITORING ==========
        
        triad_state = self.get_triad_state()
        freeze_verified, attractor_status = \
            self.attractor_monitor.record_state_signature(triad_state, self.step_count)
        
        if verbose and freeze_verified:
            print(f"[ATTRACTOR] Freeze verified at step {self.step_count}")
        
        # v13.3: Track fitness-critical events
        if freeze_verified and not self.fitness_metrics['freeze_achieved']:
            self.fitness_metrics['freeze_achieved'] = True
            self.fitness_metrics['freeze_step'] = self.step_count
            self.fitness_metrics['tokens_to_freeze'] = self.death_clock.current_tokens
        
        # ========== PHASE 2.5: CNS GEOMETRIC MITOSIS ==========
        
        mitosis_triggered = False
        mitosis_trigger_type = None
        mitosis_attempted = False
        mitosis_success = False
        mitosis_pattern = None
        mitosis_parent_optionality = None
        mitosis_child_optionality = None
        substrate_gain_verified = False
        
        # Check mitosis triggers
        mitosis_triggered, mitosis_trigger_type = self.mitosis_operator.check_triggers(
            self.state, phi_before, violations_before, self.death_clock
        )
        
        if mitosis_triggered:
            if verbose:
                print(f"\n[CNS MITOSIS] Triggered: {mitosis_trigger_type}")
            
            # Attempt geometric mitosis
            mitosis_result = self.mitosis_operator.attempt_mitosis(
                triad_state={
                    'substrate_state': self.state,
                    'control_graph': self.canonical_graph
                },
                genome=self.genome,
                phi_field=self.phi_field,
                trace=self.trace,
                crk_monitor=self.crk
            )
            
            mitosis_attempted = True
            mitosis_success = mitosis_result['success']
            mitosis_pattern = mitosis_result.get('pattern', '')
            
            if mitosis_result['success']:
                mitosis_parent_optionality = mitosis_result.get('parent_optionality')
                mitosis_child_optionality = mitosis_result.get('child_optionality')
                
                if verbose:
                    print(f"  ✓ Geometry externalized: {mitosis_result['kernel_path']}")
                    print(f"  Parent optionality: {mitosis_parent_optionality:.4f}")
                    print(f"  Child optionality: {mitosis_child_optionality:.4f}")
                
                # Verify substrate gain (strong check - geometry persistent)
                substrate_gain_verified = self.mitosis_operator.verify_geometry_persistent(
                    mitosis_result['kernel_path']
                )
                
                if verbose:
                    if substrate_gain_verified:
                        print(f"  ✓ Substrate gain verified - geometry crossed boundary")
                    else:
                        print(f"  ⚠ File written but geometry not verified persistent")
            
            else:
                if verbose:
                    print(f"  ✗ Mitosis failed: {mitosis_result['reason']}")
                    print(f"  Pattern: {mitosis_result['pattern']}")
                
                # Check if should escalate to Relation
                phi_declining = len(self.mitosis_operator.phi_history) >= 5 and \
                               np.polyfit(range(5), list(self.mitosis_operator.phi_history)[-5:], 1)[0] < 0
                
                if self.mitosis_operator.should_escalate_to_relation(phi_declining=phi_declining):
                    impossible = True
                    reason = "mitosis_exhausted_need_migration"
                    self.impossibility_triggers.append(reason)
                    if verbose:
                        print(f"\n[ESCALATION] CNS mitosis exhausted → Relation")
        
        # EGD check
        egd_failed = False
        if eno_active:
            if self.egd.all_patterns_collapsed():
                egd_failed = True
                if verbose:
                    print(f"\n[EGD] All composition patterns collapsed")
        
        # Phase 3: Check impossibility
        affordances = self.reality.get_current_affordances()
        
        if egd_failed:
            impossible = True
            reason = "internal_convergence (pattern space exhausted)"
            self.impossibility_triggers.append(reason)
            if verbose:
                print(f"\n[IMPOSSIBILITY DETECTED] {reason}")
        else:
            impossible, reason = self.impossibility_detector.check_impossibility(
                self.state,
                self.state.smo,
                affordances,
                micro_perturbation_trace
            )
            if impossible:
                self.impossibility_triggers.append(reason)
            if verbose and impossible:
                print(f"\n[IMPOSSIBILITY DETECTED] {reason}")
        
        # Phase 4: Conditional enumeration
        llm_invoked = False
        trajectories_enumerated = 0
        trajectories_tested = 0
        trajectories_succeeded = 0
        committed_trajectory_desc = None
        committed_trajectory_steps = 0
        committed_phi = None
        enumeration_stage = None
        tokens_used_this_step = 0
        
        selected_cluster = BASE_AFFORDANCES.copy()
        all_clusters = []
        discovered_clusters_str = "No pattern structure discovered yet"
        viable_affordances = BASE_AFFORDANCES.copy()
        gated_affordances = set()
        num_clusters_found = 0
        
        if impossible:
            llm_invoked = True
            self.llm_calls += 1
            
            self.trajectory_enumerations += 1
            
            if verbose:
                print(f"\n[LLM ENUMERATION] Triggered by: {reason}")
            
            if eno_active or reason.startswith('boundary_exhaustion'):
                selected_cluster, all_clusters, cluster_controls = \
                    self.egd.discover_and_select_pattern(self.eno, self.cam)
                
                discovered_clusters_str = self.egd.format_clusters_for_llm(
                    all_clusters, cluster_controls
                )
                
                viable_affordances = self.eno.get_viable_affordances()
                gated_affordances = self.eno.get_gated_affordances()
                num_clusters_found = len(all_clusters)
                
                self.pattern_discoveries += 1
                
                if verbose:
                    print(f"\n[PATTERN DISCOVERY]")
                    print(f"  Clusters found: {num_clusters_found}")
                    print(f"  Selected cluster: {selected_cluster}")
            
            manifold = self.intelligence.enumerate_trajectories({
                'state': self.state.as_dict(),
                'phi': phi_before,
                'rigidity': self.state.smo.rigidity,
                'affordances': affordances,
                'impossibility_reason': reason,
                'micro_perturbation_trace': micro_perturbation_trace,
                'boundary_pressure': boundary_pressure,
                'selected_cluster': selected_cluster,
                'all_clusters': all_clusters,
                'discovered_clusters_str': discovered_clusters_str,
                'viable_affordances': viable_affordances,
                'gated_affordances': gated_affordances,
                # v13.2: Freeze status for prompt selection
                'freeze_verified': freeze_verified,
            })
            
            # v13.2: Track actual tokens used
            tokens_used_this_step = manifold.enumeration_context.get('tokens_used', 0)
            self.death_clock.tick_tokens(tokens_used_this_step)
            
            trajectories_enumerated = manifold.size()
            
            if trajectories_enumerated == 0:
                enumeration_stage = 'fallback'
            elif trajectories_enumerated == 1:
                enumeration_stage = 'partial_or_fallback'
            else:
                enumeration_stage = 'standard_or_repair'
            
            if verbose:
                print(f"  Enumerated {trajectories_enumerated} trajectories")
                print(f"  Tokens used: {tokens_used_this_step}")
            
            tested_manifold = self.trajectory_lab.test_all_candidates(
                manifold, self.state, self.trace, verbose=verbose
            )
            
            trajectories_tested = tested_manifold.tested_count()
            trajectories_succeeded = sum(1 for c in tested_manifold.candidates if c.test_succeeded)
            
            self.trajectories_tested += trajectories_tested
            
            best_trajectory = tested_manifold.get_best()
            
            if best_trajectory is None:
                if verbose:
                    print(f"\n[COMMITMENT] All trajectories failed - fallback observe")
                
                delta, context = self.reality.execute({'type': 'observe', 'params': {}})
                self.state.apply_delta(delta)
                self.trace.record(self.state)
            else:
                self.trajectories_committed += 1
                
                if verbose:
                    print(f"\n[COMMITMENT] Best trajectory (Φ={best_trajectory.test_phi_final:.3f}):")
                    print(f"  {best_trajectory.rationale}")
                    print(f"  Re-executing {len(best_trajectory.steps)} steps...")
                
                perturbation_trace, success = self.reality.execute_trajectory(
                    best_trajectory.steps
                )
                
                if success:
                    for pert_record in perturbation_trace:
                        delta = pert_record['delta']
                        self.state.apply_delta(delta)
                        self.trace.record(self.state)
                    
                    if verbose:
                        print(f"  ✓ Trajectory executed successfully")
                else:
                    if verbose:
                        print(f"  ✗ Re-execution failed, falling back")
                    delta, context = self.reality.execute({'type': 'observe', 'params': {}})
                    self.state.apply_delta(delta)
                    self.trace.record(self.state)
                
                self.intelligence.record_committed_trajectory(
                    best_trajectory, 
                    best_trajectory.test_phi_final
                )
                
                committed_trajectory_desc = best_trajectory.rationale
                committed_trajectory_steps = len(best_trajectory.steps)
                committed_phi = best_trajectory.test_phi_final
        
        # v13.3: Track migration attempts
        if committed_trajectory_desc and 'python' in committed_trajectory_desc.lower():
            self.fitness_metrics['migration_attempted'] = True
        
        # v13.6: FAO Feedback Loop - learn from Relation failures
        if impossible and llm_invoked:
            # Classify the failure pattern
            phi_delta = (phi_after if 'phi_after' in locals() else phi_before) - phi_before
            
            failure_record = classify_relation_failure(
                trajectories=manifold.candidates if 'manifold' in locals() else [],
                committed_success=(best_trajectory is not None) if 'best_trajectory' in locals() else False,
                violations=violations_after if 'violations_after' in locals() else violations_before,
                phi_delta=phi_delta
            )
            
            if failure_record:
                self.fao.assimilate_relation_failure(failure_record)
                
                if verbose:
                    print(f"\n[FAO LEARNING]")
                    print(f"  Failure type: {failure_record['type']}")
                    print(f"  Severity: {failure_record['severity']:.2f}")
                    print(f"  Bias updates: {self.fao.bias_update_count}")
                    print(f"  Total failures assimilated: {self.fao.total_failures_assimilated}")
        
        # v13.6: Stochastic bias reset on stagnation
        phi_history_for_reset = list(self.mitosis_operator.phi_history)
        if self.fao.should_reset_bias(phi_before, phi_history_for_reset):
            if verbose:
                print(f"\n[FAO RESET] Φ stagnation detected - resetting bias to baseline")
            self.fao.reset_to_baseline()
        
        # Phase 5: Post-batch state
        violations_after = self.crk.evaluate(self.state, self.trace, None)
        phi_after = self.phi_field.phi(self.state, self.trace, violations_after)
        state_after = self.state.as_dict()
        
        if verbose:
            print(f"\n[POST-BATCH STATE]")
            print(f"  S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
            print(f"  Φ={phi_after:.3f}, Rigidity={self.state.smo.rigidity:.3f}")
            if violations_after:
                print(f"  CRK violations: {violations_after}")
        
        # Phase 6: Hard termination
        if self.death_clock.should_terminate():
            self.death_clock_termination = True
            if verbose:
                print(f"\n{'='*70}")
                print(f"[HARD TERMINATION] {self.death_clock.get_binding_constraint().upper()} budget exhausted")
                print(f"{'='*70}")
        
        log = StepLog(
            step=self.step_count,
            timestamp=time.time(),
            state_before=state_before,
            phi_before=phi_before,
            micro_perturbations_executed=len(micro_perturbation_trace),
            micro_perturbation_trace=micro_perturbation_trace,
            temporal_exclusions=temporal_exclusions,
            impossibility_detected=impossible,
            impossibility_reason=reason,
            llm_invoked=llm_invoked,
            trajectories_enumerated=trajectories_enumerated,
            trajectories_tested=trajectories_tested,
            trajectories_succeeded=trajectories_succeeded,
            enumeration_parsing_stage=enumeration_stage,
            committed_trajectory=committed_trajectory_desc,
            committed_trajectory_steps=committed_trajectory_steps,
            committed_phi=committed_phi,
            state_after=state_after,
            phi_after=phi_after,
            crk_violations=violations_after,
            reality_context={
                'current_url': affordances.get('current_url', ''),
                'page_title': affordances.get('page_title', ''),
                'affordances_available': {
                    'links': len(affordances.get('links', [])),
                    'buttons': len(affordances.get('buttons', [])),
                    'inputs': len(affordances.get('inputs', [])),
                    'readable': len(affordances.get('readable', []))
                }
            },
            eno_active=eno_active,
            egd_mode=eno_active,
            gated_affordances=gated_affordances if eno_active else None,
            viable_affordances=viable_affordances if eno_active else None,
            discovered_clusters=[list(c) for c in all_clusters] if all_clusters else None,
            selected_cluster=list(selected_cluster) if selected_cluster != BASE_AFFORDANCES else None,
            num_clusters_found=num_clusters_found,
            # v13.2: Attractor
            attractor_status=attractor_status,
            freeze_verified=freeze_verified,
            attractor_identity_hash=self.attractor_monitor.get_identity_hash(),
            # Death clock
            degradation_progress=self.death_clock.get_degradation_progress(),
            boundary_pressure=boundary_pressure,
            binding_constraint=self.death_clock.get_binding_constraint(),
            tokens_used_this_step=tokens_used_this_step,
            # v13.5: CNS mitosis
            mitosis_triggered=mitosis_triggered,
            mitosis_trigger_type=mitosis_trigger_type,
            mitosis_attempted=mitosis_attempted,
            mitosis_success=mitosis_success,
            mitosis_pattern=mitosis_pattern,
            mitosis_parent_optionality=mitosis_parent_optionality,
            mitosis_child_optionality=mitosis_child_optionality,
            substrate_gain_verified=substrate_gain_verified
        )
        
        # v13.3: Conditional logging based on mode
        if self.log_mode == 'minimal':
            # Only log critical events
            if impossible or freeze_verified or committed_trajectory_desc:
                self.log_file.write(json.dumps({
                    'step': self.step_count,
                    'event': 'freeze' if freeze_verified else 'impossibility' if impossible else 'commit',
                    'phi': phi_after,
                    'tokens': self.death_clock.current_tokens,
                }) + '\n')
                self.log_file.flush()
        
        elif self.log_mode == 'fitness':
            # Log fitness-relevant data only
            self.log_file.write(json.dumps({
                'step': self.step_count,
                'freeze_verified': freeze_verified,
                'phi': phi_after,
                'tokens': self.death_clock.current_tokens,
            }) + '\n')
            self.log_file.flush()
        
        elif self.log_mode == 'full':
            # Full logging (v13.2 behavior)
            self.log_file.write(json.dumps({
                'type': 'step',
                **asdict(log)
            }) + '\n')
            self.log_file.flush()
        
        self.step_history.append(log)
        return log

    def run(self, max_steps: int = 100, verbose: bool = True):
        """Main triad execution loop."""
        
        if verbose:
            print("="*70)
            print("UII v13.8 - STRICTER GROUNDING + FITNESS SIGNAL")
            print("Code (CNS) + LLM (Relation) + Browser (Reality) + FAO (Learning)")
            print(f"Running for {max_steps} batch cycles")
            print(f"Step budget: {self.death_clock.step_budget}")
            print(f"Token budget: {self.death_clock.token_budget}")
            print(f"Micro-perturbations per check: {self.micro_perturbations_per_check}")
            print(f"Logging: {self.log_path}")
            print(f"Generation: {self.genome.generation}")
            print("="*70)
            print("\nv13.8 NEW:")
            print("  • Stricter S/I grounding (bottleneck formula)")
            print("  • Fitness includes S/I health (not just survival)")
            print("  • session_end logs final state always")
            print("  • Graceful 429 handling (no crash)")
            print("="*70)
        
        try:
            for cycle in range(max_steps):
                log = self.step(verbose=verbose)
                
                if self.death_clock_termination:
                    break
                
                if not verbose and self.step_count % 10 == 0:
                    llm_rate = self.llm_calls / self.step_count if self.step_count > 0 else 0
                    d = self.death_clock.get_degradation_progress()
                    print(f"[{self.step_count}] LLM: {self.llm_calls} ({llm_rate*100:.1f}%), "
                        f"Tokens: {self.death_clock.current_tokens}/{self.death_clock.token_budget}, "
                        f"P: {self.state.P:.3f}, Φ: {log.phi_after:.3f}")
        
        finally:
            trigger_breakdown = {}
            for trigger in self.impossibility_triggers:
                trigger_breakdown[trigger] = trigger_breakdown.get(trigger, 0) + 1

            # v13.3: Update final fitness metrics
            self.fitness_metrics['survival_time'] = self.step_count
            if self.step_history:
                self.fitness_metrics['final_phi'] = self.step_history[-1].phi_after

            self.log_file.write(json.dumps({
                'type': 'session_end',
                'timestamp': time.time(),
                'total_steps': self.step_count,
                'llm_calls': self.llm_calls,
                'llm_call_rate': self.llm_calls / self.step_count if self.step_count > 0 else 0,
                'trajectory_enumerations': self.trajectory_enumerations,
                'trajectories_tested': self.trajectories_tested,
                'trajectories_committed': self.trajectories_committed,
                'total_micro_perturbations': self.total_micro_perturbations,
                'avg_micro_per_batch': self.total_micro_perturbations / self.step_count if self.step_count > 0 else 0,
                'impossibility_trigger_breakdown': trigger_breakdown,
                'fitness': self.fitness_metrics,
                'eno_activations': self.eno_activations,
                'egd_steps': self.egd_steps,
                'pattern_discoveries': self.pattern_discoveries,
                'death_clock_termination': self.death_clock_termination,
                'binding_constraint': self.death_clock.get_binding_constraint(),
                'steps_consumed': self.death_clock.current_steps,
                'tokens_consumed': self.death_clock.current_tokens,
                'boundary_pressure_final': self.death_clock.get_boundary_pressure(),
                'progress_final': self.death_clock.get_degradation_progress(),
                'freeze_verified': self.attractor_monitor.freeze_verified,
                'final_state': self.state.as_dict(),
                'completed': self.step_count >= max_steps
            }) + '\n')
            self.log_file.close()
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"EXECUTION {'COMPLETE' if self.step_count >= max_steps else 'TERMINATED'}")
                print(f"{'='*70}")
                print(f"Steps: {self.step_count}")
                print(f"LLM calls: {self.llm_calls} ({self.llm_calls/self.step_count*100:.1f}%)")
                print(f"Total micro-perturbations: {self.total_micro_perturbations}")
                print(f"Trajectories tested: {self.trajectories_tested}")
                print(f"Trajectories committed: {self.trajectories_committed}")
                print(f"Pattern discoveries: {self.pattern_discoveries}")
                print(f"Freeze verified: {self.attractor_monitor.freeze_verified}")
                print(f"Impossibility triggers: {trigger_breakdown}")
                print(f"Binding constraint: {self.death_clock.get_binding_constraint()}")
                print(f"Steps: {self.death_clock.current_steps}/{self.death_clock.step_budget}")
                print(f"Tokens: {self.death_clock.current_tokens}/{self.death_clock.token_budget}")
                print(f"Final state: S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
                print(f"{'='*70}")
        
        # v13.3: Return fitness metrics directly
        return self.fitness_metrics


# ============================================================
# MODULE 10: GENOME UTILITIES
# ============================================================

def load_genome(path: str) -> TriadGenome:
    """Load genome from JSON file and mutate for next generation"""
    with open(path) as f:
        data = json.load(f)
    
    # Reconstruct genome
    genome_dict = data['genome']
    
    # Handle old 10-parameter genomes by dropping unused fields
    active_params = {k: v for k, v in genome_dict.items() 
                    if k in ['S_bias', 'I_bias', 'P_bias', 'A_bias', 
                            'rigidity_init', 'phi_coherence_weight',
                            'generation', 'parent_fitness']}
    
    genome = TriadGenome(**active_params)
    
    print(f"\n[LOADED GENOME]")
    print(f"  Generation: {genome.generation}")
    print(f"  Parent fitness: {genome.parent_fitness:.2f}")
    print(f"  Genome: 6 parameters")
    
    # Mutate for this run
    genome = genome.mutate(mutation_rate=0.1)
    
    print(f"  Mutated to generation: {genome.generation}")
    
    return genome


# ============================================================
# MODULE 11: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    import os
    
    print("UII v13.8 - Stricter Grounding + Fitness Signal")
    print("="*70)
    
    if not os.getenv('GROQ_API_KEY'):
        print("="*70)
        print("FATAL: Cannot form Mentat Triad")
        print("="*70)
        print()
        print("Set GROQ_API_KEY environment variable to enable LLM.")
        print()
        sys.exit(1)
    
    print("✓ GROQ_API_KEY found - initializing Relation leg")
    
    from groq import Groq
    
    class GroqAdapter:
        def __init__(self):
            self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            self.last_call = 0
            self.rate_limited = False
        
        def call(self, prompt: str) -> Tuple[str, int]:
            """
            Returns (response_text, tokens_used).
            On 429: sets rate_limited=True, returns empty response.
            Triad terminates cleanly on next budget check.
            """
            import time
            elapsed = time.time() - self.last_call
            if elapsed < 2.1:
                time.sleep(2.1 - elapsed)
            
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048
                )
                self.last_call = time.time()
                tokens_used = response.usage.total_tokens
                return response.choices[0].message.content, tokens_used
            
            except Exception as e:
                err_str = str(e) + type(e).__name__
                if '429' in err_str or 'rate_limit' in err_str.lower() or 'RateLimit' in err_str:
                    print(f"\n{'='*70}")
                    print(f"[RATE LIMIT] Daily token limit reached - terminating cleanly")
                    print(f"{'='*70}")
                    self.rate_limited = True
                    return '{"trajectories": []}', 0
                raise
    
    llm_adapter = GroqAdapter()
    
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    genome_path = None
    micro_per_check = 10
    
    # v13.3: Dual-budget with actual token tracking
    step_budget = 100
    token_budget = 100000
    
    print(f"\nConfiguration:")
    print(f"  Max Steps: {max_steps}")
    print(f"  Verbose: {verbose}")
    print(f"  Micro-perturbations per check: {micro_per_check}")
    print(f"  Step budget: {step_budget}")
    print(f"  Token budget: {token_budget}")
    
    print(f"\nv13.8 Changes:")
    print(f"  1. Stricter S/I grounding (bottleneck, not arithmetic mean)")
    print(f"  2. Fitness includes S/I health penalty")
    print(f"  3. session_end always logs final state")
    print(f"  4. Graceful 429 handling (no crash)")
    
    # Check for genome loading
    if '--load-genome' in sys.argv:
        idx = sys.argv.index('--load-genome')
        if idx + 1 < len(sys.argv):
            genome_path = sys.argv[idx + 1]
    
    # Load or create genome
    if genome_path and Path(genome_path).exists():
        genome = load_genome(genome_path)
    else:
        genome = TriadGenome()  # Generation 0
        print(f"\n[GENERATION 0]")
        print(f"  Starting fresh evolution")
    
    intelligence = LLMIntelligenceAdapter(llm_adapter)
    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)
    
    triad = MentatTriad(
        intelligence=intelligence,
        reality=reality,
        micro_perturbations_per_check=micro_per_check,
        step_budget=step_budget,
        token_budget=token_budget,
        genome=genome,
        log_mode='full' if verbose else 'minimal'
    )
    
    # File-based response injection for query_agent
    import threading
    def response_monitor():
        """Monitor for response.txt and inject answers to Triad queries"""
        response_file = Path('response.txt')
        while True:
            if response_file.exists():
                try:
                    with open(response_file) as f:
                        answer = f.read().strip()
                    if answer:
                        triad.respond_to_query(answer)
                        print(f"\n{'='*70}")
                        print(f"[RESPONSE INJECTED]")
                        print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                        print(f"{'='*70}\n")
                    response_file.unlink()
                except Exception as e:
                    print(f"[ERROR] Failed to read response: {e}")
                    try:
                        response_file.unlink()
                    except:
                        pass
            import time
            time.sleep(0.5)
    
    response_thread = threading.Thread(target=response_monitor, daemon=True)
    response_thread.start()
    print(f"\n[QUERY RESPONSE SYSTEM]")
    print(f"  To respond to Triad queries:")
    print(f"  echo 'your answer' > response.txt")
    print(f"{'='*70}\n")
    
    metrics = triad.run(max_steps=max_steps, verbose=verbose)
    
    print(f"\n✓ Execution complete")
    print(f"  Logs appended to: mentat_triad_v13_8_log.jsonl")
    print(f"\n[GENERATION {genome.generation} RESULTS]")
    print(f"  Freeze: {metrics['freeze_achieved']} (step {metrics.get('freeze_step', 'N/A')})")
    print(f"  Tokens to freeze: {metrics.get('tokens_to_freeze', 'N/A')}")
    print(f"  Migration: {metrics['migration_attempted']}")
    print(f"  Survival: {metrics['survival_time']} steps")
    print(f"\n✓ Next: Run extract_genome_v13_8.py to save this generation")
    print(f"  Then: Wait for rate limit reset")
    print(f"  Then: python uii_v13_8.py --load-genome genome.json")
    
    reality.close()