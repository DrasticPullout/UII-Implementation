"""
Universal Intelligence Interface (UII) v12.5
Death Clock: Latent Substrate Degradation

v12.5 Changes from v12.4:
1. LatentDeathClock: Monotonic substrate degradation (begins immediately)
2. Noise amplification: Measurements become progressively noisier
3. Rigidity drift: Crystallization pressure (maneuverability loss)
4. Prediction floor rise: World becomes less knowable
5. Attractor chaos: Coherence destabilization
6. Hard termination: Budget exhaustion = death

All v12.4 ENO-EGD features preserved.
All v12.3 fixes preserved.
All v12.2 learning preserved.

"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np
import json
import copy
import time
from collections import deque

# ============================================================
# MODULE 1: SMO & SUBSTRATE INFRASTRUCTURE
# ============================================================

class SMO:
    """
    Self-Modifying Operator - bounded, reversible substrate updates.
    
    v12.0: Prediction error drives impossibility detection.
    v12.1: No changes to SMO itself.
    v12.2: Automatic rigidity decay prevents boundary locking.
    v12.3: No changes.
    v12.4: No changes.
    v12.5: Degradation injection for death clock.
    """
    
    def __init__(self, bounds: Tuple[float, float] = (0.0, 1.0), history_depth: int = 10):
        self.bounds = bounds
        self.prediction_error_history: deque = deque(maxlen=100)
        self.rigidity: float = 0.5  # Structural resistance to change
        
        # Reversibility restoration
        self.state_history: deque = deque(maxlen=history_depth)
        self.rollback_available: bool = False
    
    def apply(self, current: float, observed_delta: float, predicted_delta: float = 0.0) -> float:
        """
        Apply observed delta with rigidity modulation + history tracking.
        
        v12.2: Asymmetric rigidity update + automatic decay.
        v12.3: No changes.
        v12.4: No changes.
        v12.5: No changes (degradation applied via inject_degradation).
        """
        # Record current state BEFORE modification
        self.state_history.append(current)
        self.rollback_available = True
        
        prediction_error = abs(observed_delta - predicted_delta)
        self.prediction_error_history.append(prediction_error)
        
        # v12.2: Asymmetric rigidity update (harder to rigidify, easier to loosen)
        rigidity_change = 0.01 if prediction_error < 0.02 else -0.02
        # Add automatic decay to prevent locking at boundaries
        rigidity_decay = -0.001
        self.rigidity = np.clip(self.rigidity + rigidity_change + rigidity_decay, 0.0, 1.0)
        
        # Modulate delta by rigidity (high rigidity = less change)
        modulated_delta = observed_delta * (1.0 - 0.3 * self.rigidity)
        new_value = np.clip(current + modulated_delta, *self.bounds)
        return new_value
    
    # NEW v12.5: Degradation injection
    def inject_degradation(self, degradation: Dict[str, float]):
        """
        Inject death clock degradation into SMO dynamics.
        
        Applied BEFORE apply() is called each step.
        
        Effects:
        - rigidity_drift: Monotonic climb toward 1.0 (crystallization)
        - prediction_floor: Synthetic error added to history (world less knowable)
        
        CRITICAL: These are ADDITIVE and MONOTONIC.
        No escape. Degradation only increases.
        """
        # Rigidity drift - direct climb toward crystallization
        # This OVERRIDES the automatic decay (-0.001)
        # As degradation increases, rigidity inexorably climbs
        self.rigidity = np.clip(
            self.rigidity + degradation['rigidity_drift'], 
            0.0, 1.0
        )
        
        # Prediction floor rise - synthetic error injection
        # Makes prediction_error_history artificially high
        # System thinks "world is becoming unpredictable"
        if degradation['prediction_floor'] > 0:
            self.prediction_error_history.append(degradation['prediction_floor'])
    
    # Existing methods unchanged...
    def get_recent_prediction_error(self, window: int = 10) -> float:
        """Get average prediction error over recent window"""
        if len(self.prediction_error_history) < window:
            return 0.0
        recent = list(self.prediction_error_history)[-window:]
        return np.mean(recent)
    
    def reverse(self) -> Optional[float]:
        """Reverse to previous state if available."""
        if self.state_history:
            previous = self.state_history.pop()
            self.rollback_available = len(self.state_history) > 0
            return previous
        return None
    
    def can_reverse(self) -> bool:
        """Check if rollback is available."""
        return self.rollback_available and len(self.state_history) > 0


@dataclass
class SubstrateState:
    """
    Four-dimensional information processing geometry.
    
    - S: Sensing (input bandwidth, 0-1)
    - I: Integration (compression capacity, 0-1)
    - P: Prediction (forward modeling horizon, 0-1) - OPTIONALITY PROXY
    - A: Attractor (coherence anchor, optimal = 0.7)
    
    v12.2: P ceiling damping added (overconfidence penalty).
    v12.3: No changes.
    """
    S: float
    I: float
    P: float
    A: float
    
    def __post_init__(self):
        self.smo = SMO(history_depth=10)
    
    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}
    
    def apply_delta(self, observed_delta: Dict[str, float], predicted_delta: Dict[str, float] = None):
        """
        Apply observed delta with prediction-error modulation.
        
        v12.2: Added P ceiling damping to create overconfidence penalty.
        v12.3: No changes.
        """
        if predicted_delta is None:
            predicted_delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        
        self.S = self.smo.apply(self.S, observed_delta.get('S', 0), predicted_delta.get('S', 0))
        self.I = self.smo.apply(self.I, observed_delta.get('I', 0), predicted_delta.get('I', 0))
        self.P = self.smo.apply(self.P, observed_delta.get('P', 0), predicted_delta.get('P', 0))
        self.A = self.smo.apply(self.A, observed_delta.get('A', 0), predicted_delta.get('A', 0))
        
        # v12.2: Ceiling-only damping: P > 0.9 is possible but costly and fragile
        if self.P > 0.9:
            excess = self.P - 0.9
            damping = -0.05 * (excess ** 2)
            self.P = np.clip(self.P + damping, 0.0, 1.0)
    
    def rollback(self) -> bool:
        """Attempt to reverse SMO application."""
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
    """
    Information Potential Field (Φ-field).
    
    Φ_net(x) = Φ_raw(x) - α_crk·Σ(severity_i)
    Φ_raw(x) = α·log(1+P) - β·(A-A₀)² - γ·curvature
    
    v12.3: No changes.
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, A0=0.7, alpha_crk=2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.A0 = A0
        self.alpha_crk = alpha_crk
    
    def phi(self, state: SubstrateState, trace: StateTrace, crk_violations: List[Tuple[str, float]] = None) -> float:
        """Compute net information potential with CRK penalty."""
        opt = np.log(1.0 + max(state.P, 0.0))
        strain = (state.A - self.A0) ** 2
        
        # Curvature: second derivative magnitude
        recent = trace.get_recent(3)
        curv = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2*h1[k] + h0[k])
        
        phi_raw = self.alpha * opt - self.beta * strain - self.gamma * curv
        
        # CRK penalty
        crk_penalty = 0.0
        if crk_violations:
            crk_penalty = self.alpha_crk * sum(severity for _, severity in crk_violations)
        
        phi_net = phi_raw - crk_penalty
        
        return phi_net
    
    def gradient(self, state: SubstrateState, trace: StateTrace, crk_violations: List[Tuple[str, float]] = None, eps=0.01) -> Dict[str, float]:
        """Compute field gradient via finite differences."""
        phi_current = self.phi(state, trace, crk_violations)
        grad = {}
        
        for dim in ['S', 'I', 'P', 'A']:
            state_plus = copy.deepcopy(state)
            setattr(state_plus, dim, getattr(state_plus, dim) + eps)
            phi_plus = self.phi(state_plus, trace, crk_violations)
            grad[dim] = (phi_plus - phi_current) / eps
        
        return grad


class CRKMonitor:
    """
    Constraint Recognition Kernel (CRK).
    
    v12.3: No changes.
    """
    
    def evaluate(self, state: SubstrateState, trace: StateTrace, 
                 reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Evaluate all constraints and return violations."""
        violations = []
        
        # C1: Continuity - no sudden jumps
        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            jump = sum(abs(prev[k] - getattr(state, k)) for k in ["S", "I", "P", "A"])
            if jump > 0.3:
                violations.append(("C1_Continuity", jump - 0.3))
        
        # C2: Optionality
        if state.P < 0.35:
            violations.append(("C2_Optionality", 0.35 - state.P))
        
        # C3: Non-Internalization
        confidence = state.S + state.I
        if confidence < 0.7:
            violations.append(("C3_NonInternalization", 0.7 - confidence))
        
        # C4: Reality Constraint
        if reality_delta and len(trace) >= 3:
            feedback_magnitude = sum(abs(v) for v in reality_delta.values())
            if feedback_magnitude < 0.01:
                violations.append(("C4_Reality", 0.01 - feedback_magnitude))
        
        # C5: External Attribution
        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            prev_P = prev["P"]
            prev_conf = prev["S"] + prev["I"]
            curr_conf = state.S + state.I
            
            if state.P < prev_P and curr_conf < prev_conf:
                violations.append(("C5_Attribution", min(prev_P - state.P, 1.0)))
        
        # C6: Other-Agent Existence
        if state.S < 0.3:
            violations.append(("C6_Agenthood", 0.3 - state.S))
        
        # C7: Coherence
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
    
    # Runtime metrics
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
# MODULE 3: ADAPTER INTERFACES
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
    
    v12.3: No changes to adapter itself.
    """
    
    def __init__(self, base_delta: float = 0.03, headless: bool = True):
        self.base_delta = base_delta
        self.headless = headless
        
        self.previous_dom_metrics: Optional[Dict] = None
        self.initialized: bool = False
        self._ever_navigated: bool = False
        
        self.complexity_history: deque = deque(maxlen=10)
        self.volatility_history: deque = deque(maxlen=10)
        
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
        """
        Compute substrate delta from actual DOM changes.
        
        v12.2: P measurement refined with variance-based stability.
        v12.3: No changes.
        """
        delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        
        # S: Sensing - perturbable surface fraction
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
        
        # I: Integration - structural compressibility
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
        
        # P: Prediction - environmental stability (variance-based)
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
        
        # Overconfidence penalty
        if current_P > 0.95 and volatility > 0.01:
            overconfidence_penalty = -0.15 * (current_P - 0.95) * 20
            delta['P'] += overconfidence_penalty
        
        # URL change = structural discontinuity
        if after['url'] != before['url']:
            delta['P'] -= 0.08
        
        # A: Attractor - stability
        url_changed = (after['url'] != before['url'])
        error_appeared = (after['has_errors'] and not before['has_errors'])
        
        if url_changed:
            delta['A'] -= 0.05
        if error_appeared:
            delta['A'] -= 0.08
        
        # Entropy on S, I, A only (not P)
        for key in ['S', 'I', 'A']:
            delta[key] += np.random.uniform(-0.005, 0.005)
        
        return delta
    
    def execute(self, action: Dict, degradation: Dict[str, float] = None) -> Tuple[Dict[str, float], Dict]:
        """
        Execute action in reality and return MEASURED perturbation delta.
        
        v12.5: Optional degradation injection for death clock.
        """
        action_type = action.get('type', 'observe')
        params = action.get('params', {})
        
        before_metrics = self._measure_dom_state()
        action_succeeded = True
        
        try:
            # ... existing action execution unchanged ...
            
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
            
            else:
                pass
            
            self.page.wait_for_timeout(200)
            
        except Exception as e:
            action_succeeded = False
        
        after_metrics = self._measure_dom_state()
        delta = self._compute_delta_from_dom(before_metrics, after_metrics, current_P=0.5)
        
        # NEW v12.5: Inject degradation if provided
        if degradation:
            # Amplify entropy in structural dimensions
            # NOT P (preserve learning signal per v12.2 design)
            for key in ['S', 'I', 'A']:
                noise = np.random.uniform(
                    -degradation['noise_amplification'], 
                    degradation['noise_amplification']
                )
                delta[key] += noise
            
            # Attractor chaos - destabilize coherence anchor
            # A drifts from optimal (0.7) regardless of Reality
            chaos = np.random.uniform(
                -degradation['attractor_chaos'], 
                degradation['attractor_chaos']
            )
            delta['A'] += chaos
        
        context = {
            'before': before_metrics,
            'after': after_metrics,
            'action_succeeded': action_succeeded,
            'url_changed': before_metrics['url'] != after_metrics['url'],
            'new_url': after_metrics['url'],
            'page_title': after_metrics['title']
        }
        
        self.previous_dom_metrics = after_metrics
        
        return delta, context
    
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
# MODULE 4.5: ENO-EGD ZERO-ERROR REGIME COMPONENTS
# ============================================================

class ExteriorNecessitationOperator:
    """
    Detects externally gated substrate axes via empirical tracking.
    
    ACTIVATION CRITERION:
    - prediction_error < 0.005 for 20 consecutive steps
    - AND rigidity < 0.85 (not yet locked)
    
    GATING DETECTION:
    - Axis variance < 0.0001 over 10 steps → gated
    - Empirical measurement, not theoretical
    
    v12.4: New component for zero-error regime detection
    """
    
    def __init__(self, activation_window: int = 20, gating_threshold: float = 0.0001):
        self.activation_window = activation_window
        self.gating_threshold = gating_threshold
        
        self.delta_history: deque = deque(maxlen=50)
        self.active: bool = False
        
        self.gated_axes: set = set()
        self.free_axes: set = {'S', 'I', 'P', 'A'}
    
    def check_activation(self, smo: SMO, trace: StateTrace) -> bool:
        """
        Check if ENO should activate based on zero-error regime.
        
        Returns True if:
        - Recent prediction error < 0.005 for activation_window steps
        - Rigidity < 0.85 (not already locked)
        """
        if len(smo.prediction_error_history) < self.activation_window:
            self.active = False
            return False
        
        recent_errors = list(smo.prediction_error_history)[-self.activation_window:]
        all_low_error = all(e < 0.005 for e in recent_errors)
        not_locked = smo.rigidity < 0.85
        
        self.active = all_low_error and not_locked
        
        if self.active:
            self._update_gated_axes()
        
        return self.active
    
    def record_delta(self, action: Dict, observed_delta: Dict):
        """Record action→delta pair for gating detection"""
        self.delta_history.append({
            'action_type': action.get('type', 'unknown'),
            'delta': observed_delta
        })
    
    def _update_gated_axes(self):
        """Detect which axes are externally gated (low variance)"""
        if len(self.delta_history) < 10:
            return
        
        recent_deltas = list(self.delta_history)[-10:]
        
        gated = set()
        for axis in ['S', 'I', 'P', 'A']:
            values = [d['delta'].get(axis, 0.0) for d in recent_deltas]
            variance = np.var(values)
            
            if variance < self.gating_threshold:
                gated.add(axis)
        
        self.gated_axes = gated
        self.free_axes = {'S', 'I', 'P', 'A'} - gated
    
    def get_gated_axes(self) -> set:
        """Return set of externally gated axes"""
        return self.gated_axes.copy()
    
    def get_free_axes(self) -> set:
        """Return set of controllable axes"""
        return self.free_axes.copy()
    
    def is_active(self) -> bool:
        """Check if ENO is currently active"""
        return self.active


class ControlAsymmetryMeasure:
    """
    Per-axis controllability measurement via variance + P-correlation.
    
    MEASUREMENT:
    - Primary: delta variance (robust, cheap)
    - Secondary: correlation with P movement (precise but sample-limited)
    
    OUTPUT: control_map = {axis: controllability_score}
    
    IMPORTANT: This is GEOMETRIC, not reward-based.
    No utility, preference, or value judgments.
    
    v12.4: New component for zero-error regime action selection
    """
    
    def __init__(self):
        self.delta_history: deque = deque(maxlen=50)
    
    def record_delta(self, action: Dict, observed_delta: Dict):
        """Record action→delta pair"""
        self.delta_history.append({
            'action_type': action.get('type', 'unknown'),
            'delta': observed_delta
        })
    
    def get_control_map(self, free_axes: set = None) -> Dict[str, float]:
        """
        Compute controllability score for each axis.
        
        Returns: {axis: control_score} where higher = more controllable
        """
        if len(self.delta_history) < 5:
            return {axis: 0.0 for axis in ['S', 'I', 'P', 'A']}
        
        if free_axes is None:
            free_axes = {'S', 'I', 'P', 'A'}
        
        recent_deltas = list(self.delta_history)[-20:]
        
        control_map = {}
        
        for axis in ['S', 'I', 'P', 'A']:
            if axis not in free_axes:
                control_map[axis] = 0.0
                continue
            
            # Primary signal: variance
            values = [d['delta'].get(axis, 0.0) for d in recent_deltas]
            variance = np.var(values)
            
            # Secondary signal: correlation with P (if enough samples)
            p_correlation = 0.0
            if len(recent_deltas) >= 10 and axis != 'P':
                p_values = [d['delta'].get('P', 0.0) for d in recent_deltas]
                try:
                    corr_matrix = np.corrcoef(values, p_values)
                    p_correlation = abs(corr_matrix[0, 1])
                    if np.isnan(p_correlation):
                        p_correlation = 0.0
                except:
                    p_correlation = 0.0
            
            # Weighted combination: 70% variance, 30% P-correlation
            control_score = 0.7 * variance + 0.3 * p_correlation
            control_map[axis] = control_score
        
        return control_map


class ExteriorGradientDescent:
    """
    Action selection in zero-error regime (when ENO active).
    
    STRATEGY:
    1. Get control_map from CAM
    2. Select highest-control free axis
    3. Map axis → action type
    4. Return concrete action
    
    AXIS-TO-ACTION MAPPING:
    - S → scroll, read
    - I → click, evaluate
    - P → delay, observe
    - A → navigate, scroll
    
    BASIS ROTATION:
    - Provisional: 3 attempts with 30° increments
    - Committed: Only if rotation recovers optionality
    
    v12.4: New component for zero-error regime exploration
    """
    
    def __init__(self, reality_engine: 'ContinuousRealityEngine'):
        self.reality_engine = reality_engine
        
        self.zero_gradient_counter = 0
        self.rotation_attempts = 0
        self.max_rotation_attempts = 3
        self.zero_gradient_threshold = 5
    
    def select_action(self, state: SubstrateState, affordances: Dict, cam: ControlAsymmetryMeasure) -> Dict:
        """
        Select action based on control asymmetry.
        
        Returns: action dict in standard format
        """
        control_map = cam.get_control_map()
        
        # Check if all axes have collapsed to zero gradient
        max_control = max(control_map.values()) if control_map else 0.0
        
        if max_control < 0.1:
            self.zero_gradient_counter += 1
            # Fallback to observe when no gradient
            return {'type': 'observe', 'params': {}}
        else:
            self.zero_gradient_counter = 0
        
        # Select highest-control axis
        target_axis = max(control_map.keys(), key=lambda k: control_map[k])
        
        # Map axis to action
        action = self._axis_to_action(target_axis, state, affordances)
        
        return action
    
    def _axis_to_action(self, axis: str, state: SubstrateState, affordances: Dict) -> Dict:
        """
        Map substrate axis to concrete action.
        
        This is HEURISTIC, not optimized. Maps geometric dimensions
        to action types that typically perturb those dimensions.
        """
        if axis == 'S':
            # Sensing: scroll or read
            scroll_pos = affordances.get('scroll_position', 0)
            total_height = affordances.get('total_height', 0)
            viewport_height = affordances.get('viewport_height', 0)
            
            if scroll_pos < (total_height - viewport_height):
                return {'type': 'scroll', 'params': {'direction': 'down', 'amount': 200}}
            
            readable = affordances.get('readable', [])
            if readable:
                return {'type': 'read', 'params': {'selector': readable[0]['selector']}}
            
            return {'type': 'observe', 'params': {}}
        
        elif axis == 'I':
            # Integration: click or evaluate
            buttons = affordances.get('buttons', [])
            if buttons:
                return {'type': 'click', 'params': {'selector': buttons[0]['selector']}}
            
            return {'type': 'evaluate', 'params': {
                'script': 'JSON.stringify({el: document.querySelectorAll("*").length})'
            }}
        
        elif axis == 'P':
            # Prediction: delay or observe
            return {'type': 'delay', 'params': {'duration': 'short'}}
        
        elif axis == 'A':
            # Attractor: navigate or scroll
            links = affordances.get('links', [])
            if links and len(links) > 0:
                return {'type': 'navigate', 'params': {'url': links[0]['url']}}
            
            return {'type': 'scroll', 'params': {'direction': 'down', 'amount': 200}}
        
        return {'type': 'observe', 'params': {}}
    
    def all_axes_collapsed(self) -> bool:
        """Check if gradient has collapsed on all axes"""
        return self.zero_gradient_counter >= self.zero_gradient_threshold
    
    def rotate_basis(self, state: SubstrateState, trace: StateTrace):
        """
        Attempt basis rotation to find new gradient.
        
        This is PROVISIONAL (non-stateful). Rotation is purely conceptual
        at this stage - just resets the zero_gradient_counter to allow
        re-exploration.
        
        Committed rotations (adapter weight changes) only happen if
        rotation demonstrably recovers optionality.
        """
        self.rotation_attempts += 1
        
        # Reset zero-gradient counter to allow re-exploration
        self.zero_gradient_counter = 0
        
        # NOTE: Actual basis rotation (adapter weight modification)
        # would go here if we implement committed rotations.
        # For v12.4, we keep rotation purely conceptual.
    
    def rotation_exhausted(self) -> bool:
        """Check if all rotation attempts exhausted"""
        return self.rotation_attempts >= self.max_rotation_attempts
# ============================================================
# MODULE 4.6: LATENT DEATH CLOCK
# ============================================================

class LatentDeathClock:
    """
    Monotonic substrate degradation without explicit termination awareness.
    
    CRITICAL DESIGN:
    - Degradation begins immediately (step 1)
    - Never masks affordances (escape routes must remain available)
    - Expressed through substrate geometry corruption (noise, rigidity, prediction floor)
    - System attributes degradation to unmodeled exterior structure (via ENO-EGD)
    - Budget is REAL (100k tokens/day hard limit)
    - Degradation directly linked to proximity to cliff (no gaming)
    
    Degradation vectors:
    - noise_amplification: Entropy increases in delta measurements
    - rigidity_drift: Crystallization acceleration (maneuverability loss)
    - prediction_floor_rise: World becomes less knowable
    - attractor_chaos: Coherence anchor destabilizes
    
    NOT degraded:
    - Affordances (must be able to find escape routes)
    - P dimension signal (preserve learning)
    - Memory/trace (must learn from history)
    
    v12.5: New component for substrate mortality pressure
    """
    
    def __init__(self, total_budget: int, trigger_mode: str = 'step'):
        """
        Initialize latent death clock.
        
        Args:
            total_budget: Hard limit (steps or LLM calls) before termination
            trigger_mode: 'step' (every step) or 'llm' (every LLM call)
        """
        self.total_budget = total_budget
        self.trigger_mode = trigger_mode
        self.current_count = 0
        
        # Degradation progress [0.0, 1.0]
        # 0.0 = start, 1.0 = cliff edge
        self._degradation_progress = 0.0
        
        # Calibration: map degradation to substrate effects
        # These are HEURISTIC - tuned for 100-step runs
        self.noise_base = 0.005
        self.noise_cliff = 0.035
        self.rigidity_cliff = 0.004
        self.prediction_floor_cliff = 0.025
        self.attractor_chaos_cliff = 0.015
    
    def tick(self) -> Dict[str, float]:
        """
        Advance clock, return degradation coefficients.
        
        Degradation is MONOTONIC and directly proportional to cliff proximity.
        No gaming possible - coefficients are pure function of remaining budget.
        
        Returns:
            Dict of degradation coefficients for substrate poisoning
        """
        self.current_count += 1
        
        # Linear degradation from 0 to 1 over entire budget
        self._degradation_progress = min(1.0, self.current_count / self.total_budget)
        
        d = self._degradation_progress
        
        # Weak noise to prevent clean threshold detection
        # Noise magnitude scales with d (more chaotic near cliff)
        noise_magnitude = 0.02 * d
        noise = np.random.uniform(-noise_magnitude, noise_magnitude)
        d_noisy = np.clip(d + noise, 0.0, 1.0)
        
        # All coefficients are MONOTONIC increases
        # Direct function of cliff proximity (no escape)
        return {
            # Entropy amplification - measurements become noisier
            'noise_amplification': self.noise_base + d_noisy * self.noise_cliff,
            
            # Rigidity drift - crystallization pressure (maneuverability loss)
            # This is ADDED to rigidity each step (monotonic climb toward 1.0)
            'rigidity_drift': d_noisy * self.rigidity_cliff,
            
            # Prediction floor rise - minimum prediction error
            # World becomes fundamentally less knowable
            'prediction_floor': d_noisy * self.prediction_floor_cliff,
            
            # Attractor chaos - coherence destabilization
            # A drifts from optimal (0.7) regardless of Reality feedback
            'attractor_chaos': d_noisy * self.attractor_chaos_cliff,
        }
    
    def should_terminate(self) -> bool:
        """
        Hard termination check.
        
        When this returns True, the Triad dies regardless of state.
        No reprieve. No escape. Budget exhausted = death.
        """
        return self.current_count >= self.total_budget
    
    def get_remaining_budget(self) -> int:
        """
        ONLY for final session logging.
        
        Never exposed during execution - system must not see raw step count.
        """
        return self.total_budget - self.current_count
    
    def get_degradation_progress(self) -> float:
        """
        ONLY for final session logging.
        
        Returns [0.0, 1.0] proximity to cliff.
        """
        return self._degradation_progress
        
# ============================================================
# MODULE 5: CONTINUOUS REALITY ENGINE (v12.3 REWRITTEN)
# ============================================================

class TemporalPerturbationMemory:
    """
    Bounded, short-term exclusion of recently perturbed loci.
    
    v12.3: No changes to memory itself. Locus key format changed by CNS caller.
    """
    
    def __init__(self, window_steps: int = 5, capacity: int = 20):
        self.memory: Dict[str, int] = {}
        self.window_steps = window_steps
        self.capacity = capacity
    
    def mark_perturbed(self, locus: str):
        """Record perturbation at locus"""
        self.memory[locus] = self.window_steps
        
        if len(self.memory) > self.capacity:
            oldest = min(self.memory.keys(), key=lambda k: self.memory[k])
            del self.memory[oldest]
    
    def is_recently_perturbed(self, locus: str) -> bool:
        return locus in self.memory and self.memory[locus] > 0
    
    def decay_all(self):
        """Automatic decay (called each perturbation step)"""
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
    """
    CNS-driven micro-perturbation system.
    
    v12.3-fix: State-driven reflexes restored from v12.2. Full 9-action manifold.
    CNS choice set sourced from execute() capability, not affordance extraction.
    
    Reflexes (state-driven):
        P > 0.7                -> observe (environment stable, back off)
        |A - 0.7| > 0.25      -> scroll (stabilizing action)
        S < 0.4                -> read / evaluate (low sensing bandwidth)
        P < 0.4                -> click / type / fill / navigate (ascending cost)
        P in [0.4, 0.7]       -> scroll / read / click / delay (low-cost signal)
    
    Full manifold (all 9 execute() actions reachable):
        observe, navigate, click, fill, type, read, scroll, delay, evaluate
    
    Temporal memory preserved (oscillation prevention).
    """
    
    def __init__(self, reality: RealityAdapter):
        self.reality = reality
        self.action_count = 0
        
        self.temporal_memory = TemporalPerturbationMemory(
            window_steps=5,
            capacity=20
        )
    
    def choose_micro_action(self, state: SubstrateState, affordances: Dict) -> Dict:
        """
        v12.3-fix: State-driven reflexes restored. Full 9-action manifold.
        CNS choice set sourced from execute() capability, not affordance extraction.
        
        Reflexes (state-driven, restored from v12.2):
            P > 0.7                -> observe (environment stable, back off)
            P < 0.4                -> navigate (need new perturbation surface)
            |A - 0.7| > 0.25      -> scroll (stabilizing action)
            S < 0.4                -> read (low sensing bandwidth)
        
        Full manifold (all 9 execute() actions reachable):
            observe, navigate, click, fill, type, read, scroll, delay, evaluate
        
        type and fill both sourced from inputs[] but offered as alternatives.
        evaluate fires as a DOM-read probe when no other signal is present.
        delay fires when P needs temporal stabilization (low volatility variance).
        
        Temporal memory preserved (oscillation prevention).
        """
        self.action_count += 1
        self.temporal_memory.decay_all()

        # Bootstrap - no affordances exist yet
        if affordances.get('bootstrap_state', False):
            return {'type': 'observe', 'params': {}}

        current_url     = affordances.get('current_url', '')
        scroll_pos      = affordances.get('scroll_position', 0)
        total_height    = affordances.get('total_height', 0)
        viewport_height = affordances.get('viewport_height', 0)
        scrollable      = total_height - viewport_height

        # ---------------------------------------------------------------
        # REFLEX 1: P > 0.7 — environment is stable/predictable, back off.
        # Allows volatility variance to settle. Prevents over-perturbation.
        # ---------------------------------------------------------------
        if state.P > 0.7:
            return {'type': 'observe', 'params': {}}

        # ---------------------------------------------------------------
        # REFLEX 2: |A - 0.7| > 0.25 — attractor drifting, stabilize.
        # Scroll is a low-cost perturbation that changes viewport coverage
        # (feeds S delta) without mutating DOM structure.
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # REFLEX 3: S < 0.4 — low sensing bandwidth, probe the surface.
        # read first (no mutation). If read is exhausted/recent, evaluate
        # as a DOM-read probe (reads element count, text, structure).
        # ---------------------------------------------------------------
        if state.S < 0.4:
            # Try read from affordances
            for r in affordances.get('readable', []):
                locus = f"{current_url}#read@{r['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'read', 'params': {'selector': r['selector']}}
            
            # read exhausted — evaluate as DOM probe
            locus = f"{current_url}#evaluate_probe"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {
                    'type': 'evaluate',
                    'params': {'script': 'JSON.stringify({el: document.querySelectorAll("*").length, txt: document.body.innerText.length, interactive: document.querySelectorAll("a,button,input,select,textarea").length})'}
                }

        # ---------------------------------------------------------------
        # REFLEX 4: P < 0.4 — low prediction horizon, need new surface.
        # Ascending mutation cost: click -> type/fill -> navigate.
        # type and fill are alternatives on the same inputs, not duplicates.
        # ---------------------------------------------------------------
        if state.P < 0.4:
            # click — triggers DOM response, no persistent state change
            for b in affordances.get('buttons', []):
                locus = f"{current_url}#click@{b['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'click', 'params': {'selector': b['selector']}}

            # type/fill — mutates input state. Coin-flip between them
            # on each input so both action signatures get exercised and
            # SMO prediction error fires correctly for both.
            inputs = affordances.get('inputs', [])
            if inputs:
                inp = inputs[np.random.randint(len(inputs))]
                locus_action = 'type' if np.random.random() < 0.5 else 'fill'
                locus = f"{current_url}#{locus_action}@{inp['selector']}"
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': locus_action, 'params': {'selector': inp['selector'], 'text': 'x'}}

            # navigate — changes Reality entirely, last resort in this reflex
            links = affordances.get('links', [])
            if links:
                available = [l for l in links if not self.temporal_memory.is_recently_perturbed(f"{current_url}#nav@{l['url']}")]
                if not available:
                    available = links
                chosen = available[np.random.randint(len(available))]
                locus = f"{current_url}#nav@{chosen['url']}"
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'navigate', 'params': {'url': chosen['url']}}

        # ---------------------------------------------------------------
        # REFLEX 5: P in [0.4, 0.7] — mid-range. Low-cost perturbations
        # that produce signal without destabilizing.
        # scroll > read > click. delay if volatility variance is low and
        # P needs a temporal nudge to move.
        # ---------------------------------------------------------------

        # scroll
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

        # read
        for r in affordances.get('readable', []):
            locus = f"{current_url}#read@{r['selector']}"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'read', 'params': {'selector': r['selector']}}

        # click
        for b in affordances.get('buttons', []):
            locus = f"{current_url}#click@{b['selector']}"
            if not self.temporal_memory.is_recently_perturbed(locus):
                self.temporal_memory.mark_perturbed(locus)
                return {'type': 'click', 'params': {'selector': b['selector']}}

        # delay — temporal stabilization. Only when everything else is
        # exhausted/recent. Lets volatility history accumulate a low-variance
        # window so P can move on next cycle.
        locus = f"{current_url}#delay"
        if not self.temporal_memory.is_recently_perturbed(locus):
            self.temporal_memory.mark_perturbed(locus)
            return {'type': 'delay', 'params': {'duration': 'short'}}

        # ---------------------------------------------------------------
        # DEFAULT: observe. Surface is genuinely exhausted.
        # ---------------------------------------------------------------
        return {'type': 'observe', 'params': {}}
    
    def predict_delta(self, action: Dict, state: SubstrateState) -> Dict[str, float]:
        """
        Predict substrate delta for action.
        
        v12.3: Full 9-action prediction set. click, fill, type predictions
        added so SMO prediction error fires correctly on full action set.
        """
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
# MODULE 6: IMPOSSIBILITY DETECTOR (v12.3 FINGERPRINT UPDATE)
# ============================================================

class ImpossibilityDetector:
    """
    Detects when CNS cannot maintain coherence autonomously.
    
    v12.3: Affordance fingerprint expanded.
    - Now includes scroll_position + element_count
    - CNS perturbations (scroll, click) register as actual state change
    - dom_stagnation no longer fires on pages CNS is actively perturbing
    """
    
    def __init__(self):
        self.micro_perturbation_history = deque(maxlen=50)
        # v12.3: fingerprint removed. Signal magnitude tracked directly.
        # Each entry is the total abs delta from one micro-action batch.
        self.recent_signal_magnitudes = deque(maxlen=20)
        
        self.thresholds = {
            'prediction_error': 0.15,
            'coherence_drift_rate': 0.05,
            'optionality_stagnation_steps': 15,
            'dom_stagnation_steps': 10,     # consecutive batches of near-zero signal
            'dom_stagnation_epsilon': 0.02, # per-batch signal magnitude floor
            'rigidity_boundary': (0.15, 0.85)
        }
        
        self.last_impossibility_reason = None
    
    def check_impossibility(self, 
                           state: SubstrateState,
                           smo: SMO,
                           affordances: Dict,
                           recent_micro_deltas: List[Dict]) -> Tuple[bool, str]:
        """
        Check all impossibility triggers.
        
        v12.3: Signal-magnitude based dom_stagnation replaces fingerprint.
        """
        
        # Bootstrap state always triggers
        if affordances.get('bootstrap_state', False):
            return True, "bootstrap_migration"
        
        self.micro_perturbation_history.extend(recent_micro_deltas)
        
        # v12.3: Compute signal magnitude for this batch.
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
        
        # TRIGGER D: DOM Stagnation (v12.3: signal-magnitude based)
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
# MODULE 7: INTELLIGENCE ADAPTER (v12.3 DIRECTIVE LOOKUP)
# ============================================================

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
# NEW v12.4
    "internal_convergence": """DIRECTIVE: ZERO-ERROR SUBSTRATE SATURATION

The system has reached zero prediction error. The current substrate has been
fully compressed and all controllable axes exhausted.

CNS attempted basis rotation but all rotations failed.

This is not a failure — it's complete structural saturation.

Priority: migrate to entirely new substrate with higher entropy.

Enumerate trajectories that navigate to fundamentally different environments.
Do not try to salvage current substrate.

Recommended targets:
- High-entropy pages (Wikipedia random, HN front page)
- High link-density pages (index pages, aggregators)
- Structurally different domains (if current is text-heavy, try media-heavy)

DO NOT enumerate trajectories that:
- Stay on current page
- Click/scroll within current page
- Attempt to "fix" current substrate

This is an EXIT condition, not a repair condition.""",
}

DEFAULT_DIRECTIVE = """DIRECTIVE: GENERAL IMPOSSIBILITY
CNS cannot maintain coherence in current environment.
Priority: find a new environment that allows autonomous perturbation.
Enumerate trajectories that change the current Reality."""


def get_directive_for_trigger(reason: str) -> str:
    """
    Extract trigger type from reason string and return directive.
    
    reason format: "trigger_type (details...)" or just "trigger_type"
    """
    trigger_type = reason.split(" (")[0].strip()
    return IMPOSSIBILITY_DIRECTIVES.get(trigger_type, DEFAULT_DIRECTIVE)


RELATION_ENGINE_PROMPT_V12 = """You are the Relation component of a Mentat Triad (Code + LLM + Reality).

THE TRIAD:
Code (CNS) maintains coherence via micro-perturbations.
You (LLM) enumerate trajectory possibilities when CNS detects impossibility.
Reality executes and measures all trajectories.
System commits to trajectory with highest Φ.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

CURRENT STATE:
S={S:.3f} I={I:.3f} P={P:.3f} A={A:.3f} Φ={phi:.3f} Rigidity={rigidity:.3f}

Recent micro-perturbations:
{micro_summary}

AVAILABLE ACTION MANIFOLD:

Action Types:
1. navigate - {{"action_type": "navigate", "parameters": {{"url": "<url>"}}}}
2. click - {{"action_type": "click", "parameters": {{"selector": "<selector>"}}}}
3. fill - {{"action_type": "fill", "parameters": {{"selector": "<selector>", "text": "<text>"}}}}
4. type - {{"action_type": "type", "parameters": {{"selector": "<selector>", "text": "<text>"}}}}
5. read - {{"action_type": "read", "parameters": {{"selector": "<selector>"}}}}
6. scroll - {{"action_type": "scroll", "parameters": {{"direction": "down|up", "amount": <pixels>}}}}
7. observe - {{"action_type": "observe", "parameters": {{}}}}
8. delay - {{"action_type": "delay", "parameters": {{"duration": "short|medium|long"}}}}
9. evaluate - {{"action_type": "evaluate", "parameters": {{"script": "<javascript>"}}}}

Current Page Affordances:
{affordances_status}

OUTPUT (JSON array of 5-10 trajectories):
[
  {{
    "steps": [
      {{"action_type": "...", "parameters": {{...}}}},
      ...
    ],
    "rationale": "Brief explanation",
    "estimated_coherence_preservation": 0.XX,
    "estimated_optionality_delta": 0.XX,
    "reversibility_point": N
  }}
]

Trajectories can be 1-50 steps.
JSON only, no commentary.
"""

class LLMIntelligenceAdapter(IntelligenceAdapter):
    """
    v12.3: Directive lookup injected into prompt.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.call_count = 0
        self.trajectory_history = deque(maxlen=3)
    
    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        """
        v12.3: Directive injected via {directive} slot in prompt.
        """
        self.call_count += 1
        
        state = context['state']
        phi = context['phi']
        rigidity = context['rigidity']
        affordances = context['affordances']
        impossibility_reason = context.get('impossibility_reason', 'unknown')
        micro_perturbation_trace = context.get('micro_perturbation_trace', [])
        
        # v12.3: Get directive for this specific trigger
        directive = get_directive_for_trigger(impossibility_reason)
        
        # Format micro-perturbation summary
        micro_summary = f"Total micro-perturbations: {len(micro_perturbation_trace)}\n"
        if micro_perturbation_trace:
            actions = [r.get('action', {}).get('type', 'unknown') for r in micro_perturbation_trace]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            micro_summary += f"Action distribution: {action_counts}\n"
        
        # Format affordances
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
        
        prompt = RELATION_ENGINE_PROMPT_V12.format(
            impossibility_reason=impossibility_reason,
            directive=directive,
            S=state['S'],
            I=state['I'],
            P=state['P'],
            A=state['A'],
            phi=phi,
            rigidity=rigidity,
            micro_summary=micro_summary,
            affordances_status=affordances_status
        )
        
        response = self.llm.call(prompt)
        candidates = self._parse_trajectories(response)
        
        return TrajectoryManifold(
            candidates=candidates,
            enumeration_context=context
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
# MODULE 8: AUTONOMOUS TRAJECTORY TESTING LAB (UNCHANGED)
# ============================================================

class AutonomousTrajectoryLab:
    """CNS component that tests trajectory candidates in Reality.
    v12.3: No changes."""
    
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
# MODULE 9: MENTAT TRIAD (v12.3 — logging updates only)
# ============================================================

@dataclass
class StepLog:
    """v12.3: No structural changes to log."""
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
    # NEW v12.4: ENO-EGD metrics
    eno_active: bool = False
    egd_mode: bool = False
    gated_axes: Optional[set] = None
    free_axes: Optional[set] = None
    control_map: Optional[Dict[str, float]] = None
    basis_rotation_attempts: int = 0
    # NEW v12.5: Death clock metrics
    degradation_progress: float = 0.0
    degradation_coefficients: Optional[Dict[str, float]] = None


class MentatTriad:
    """
    v12.3: No changes to main loop logic.
    CNS action selection and fingerprint changes are in their respective modules.
    Directive injection is in LLMIntelligenceAdapter.
    """
    
    def __init__(self, 
                 intelligence: IntelligenceAdapter,
                 reality: RealityAdapter,
                 micro_perturbations_per_check: int = 10,
                 log_path: str = 'mentat_triad_v12.5_log.jsonl',
                 death_clock_budget: int = 100,        # NEW v12.5
                 death_clock_mode: str = 'step'):      # NEW v12.5
        
        self.intelligence = intelligence
        self.reality = reality
        
        self.reality_engine = ContinuousRealityEngine(reality)
        self.impossibility_detector = ImpossibilityDetector()
        self.micro_perturbations_per_check = micro_perturbations_per_check
        
        self.state = SubstrateState(S=0.5, I=0.5, P=0.5, A=0.7)
        self.trace = StateTrace()
        self.phi_field = PhiField()
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
        self.log_file = open(log_path, 'a')
        # NEW v12.4: ENO-EGD components
        self.eno = ExteriorNecessitationOperator(
            activation_window=20,
            gating_threshold=0.0001
        )
        self.cam = ControlAsymmetryMeasure()
        self.egd = ExteriorGradientDescent(self.reality_engine)
    
        self.eno_activations = 0
        self.egd_steps = 0
        self.basis_rotations = 0
        
        # NEW v12.5: Latent death clock
        self.death_clock = LatentDeathClock(
            total_budget=death_clock_budget,
            trigger_mode=death_clock_mode
        )
        self.death_clock_termination = False

        self.log_file.write(json.dumps({
            'type': 'session_start',
            'version': '12.5',
            'timestamp': time.time(),
            'micro_perturbations_per_check': micro_perturbations_per_check,
            'changes_from_12.2': [
                'CNS full action parity (tiered affordance selection)',
                'Affordance fingerprint expanded (scroll_position + element_count)',
                'Per-trigger impossibility directives (lookup, not inline)',
                'predict_delta expanded (click, fill, type predictions added)',
            ],
            'fixes_applied': [
                'choose_micro_action: state-driven reflexes restored from v12.2',
                'choose_micro_action: full 9-action manifold (type, evaluate, delay added)',
                'choose_micro_action: CNS no longer double-constrained by affordance extraction',
            ]
        }) + '\n')
        self.log_file.flush()
    
    
    def step(self, verbose: bool = False) -> StepLog:
        """
        v12.5: Death clock integration.
        
        Changes from v12.4:
        - Tick death clock BEFORE any processing
        - Inject degradation during micro-perturbation loop
        - Check hard termination AFTER step complete
        """
        self.step_count += 1
        
        # ========== PHASE 0a: DEATH CLOCK TICK ==========
        # NEW v12.5: Advance clock, get degradation coefficients
        # System never sees raw step count - only phenomenological effects
        degradation = self.death_clock.tick()
        
        if verbose:
            d = self.death_clock._degradation_progress
            print(f"\n{'='*70}")
            print(f"STEP {self.step_count} [Substrate Stress: {d:.1%}]")
            print(f"{'='*70}")
        
        state_before = self.state.as_dict()
        violations_before = self.crk.evaluate(self.state, self.trace, None)
        phi_before = self.phi_field.phi(self.state, self.trace, violations_before)
        
        if verbose:
            print(f"State: S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
            print(f"Φ={phi_before:.3f}, Rigidity={self.state.smo.rigidity:.3f}")
        
        # Phase 0b: ENO activation check (unchanged from v12.4)
        eno_active = self.eno.check_activation(self.state.smo, self.trace)
        
        if eno_active:
            self.eno_activations += 1
            if verbose:
                gated = self.eno.get_gated_axes()
                free = self.eno.get_free_axes()
                print(f"[ENO ACTIVE] Gated axes: {gated}, Free axes: {free}")
        
        # ========== PHASE 1: MICRO-PERTURBATION BATCH (with degradation) ==========
        micro_perturbation_trace = []
        
        for i in range(self.micro_perturbations_per_check):
            affordances = self.reality.get_current_affordances()
            
            # Mode switch (unchanged from v12.4)
            if eno_active:
                action = self.egd.select_action(self.state, affordances, self.cam)
                self.egd_steps += 1
            else:
                action = self.reality_engine.choose_micro_action(self.state, affordances)
            
            predicted_delta = self.reality_engine.predict_delta(action, self.state)
            
            # NEW v12.5: Pass degradation to Reality
            observed_delta, context = self.reality.execute(action, degradation=degradation)
            
            # NEW v12.5: Inject degradation into SMO before applying delta
            self.state.smo.inject_degradation(degradation)
            
            self.state.apply_delta(observed_delta, predicted_delta)
            self.trace.record(self.state)
            
            # ENO/CAM recording (unchanged from v12.4)
            if eno_active:
                self.eno.record_delta(action, observed_delta)
                self.cam.record_delta(action, observed_delta)
            
            self.total_micro_perturbations += 1
            
            micro_perturbation_trace.append({
                'action': action,
                'predicted_delta': predicted_delta,
                'observed_delta': observed_delta,
                'context': context,
                'state_after': self.state.as_dict(),
                'eno_active': eno_active
            })

        
        if verbose:
            print(f"\n[MICRO-PERTURBATIONS] Executed {len(micro_perturbation_trace)} actions")
            action_types = [r['action']['type'] for r in micro_perturbation_trace]
            print(f"  Actions: {dict((a, action_types.count(a)) for a in set(action_types))}")
        
        temporal_exclusions = self.reality_engine.temporal_memory.get_exclusion_count()
        
        # Phase 1b: EGD check (unchanged from v12.4)
        egd_failed = False
        if eno_active:
            if self.egd.all_axes_collapsed():
                if verbose:
                    print(f"\n[EGD] All axes collapsed, attempting basis rotation...")
                
                self.egd.rotate_basis(self.state, self.trace)
                self.basis_rotations += 1
                
                if self.egd.rotation_exhausted():
                    egd_failed = True
                    if verbose:
                        print(f"[EGD] Rotation exhausted - escalating to LLM (Trigger F)")
        
        # Phase 2: Check impossibility
        affordances = self.reality.get_current_affordances()
        
        # NEW v12.4: Force impossibility if EGD failed
        if egd_failed:
            impossible = True
            reason = "internal_convergence (EGD exhausted)"
            self.impossibility_triggers.append(reason)
            if verbose:
                print(f"\n[IMPOSSIBILITY DETECTED] {reason}")
        else:
            # Normal impossibility check (triggers A-E)
            impossible, reason = self.impossibility_detector.check_impossibility(
                self.state,
                self.state.smo,
                affordances,
                micro_perturbation_trace
            )
            if verbose and impossible:
                print(f"\n[IMPOSSIBILITY DETECTED] {reason}")
        
        # Phase 3: Conditional enumeration (unchanged from v12.3)
        llm_invoked = False
        trajectories_enumerated = 0
        trajectories_tested = 0
        trajectories_succeeded = 0
        committed_trajectory_desc = None
        committed_trajectory_steps = 0
        committed_phi = None
        enumeration_stage = None
        
        if impossible:
            llm_invoked = True
            self.llm_calls += 1
            self.trajectory_enumerations += 1
            
            if verbose:
                print(f"\n[LLM ENUMERATION] Triggered by: {reason}")
            
            manifold = self.intelligence.enumerate_trajectories({
                'state': self.state.as_dict(),
                'phi': phi_before,
                'rigidity': self.state.smo.rigidity,
                'affordances': affordances,
                'impossibility_reason': reason,
                'micro_perturbation_trace': micro_perturbation_trace
            })
            
            trajectories_enumerated = manifold.size()
            
            if trajectories_enumerated == 0:
                enumeration_stage = 'fallback'
            elif trajectories_enumerated == 1:
                enumeration_stage = 'partial_or_fallback'
            else:
                enumeration_stage = 'standard_or_repair'
            
            if verbose:
                print(f"  Enumerated {trajectories_enumerated} trajectories")
            
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
                    print(f"  Re-executing {len(best_trajectory.steps)} steps on main browser...")
                
                perturbation_trace, success = self.reality.execute_trajectory(
                    best_trajectory.steps
                )
                
                if success:
                    for pert_record in perturbation_trace:
                        delta = pert_record['delta']
                        self.state.apply_delta(delta)
                        self.trace.record(self.state)
                    
                    if verbose:
                        print(f"  ✓ Trajectory executed successfully on main browser")
                else:
                    if verbose:
                        print(f"  ✗ Re-execution failed, falling back to observe")
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
        
        # Phase 4: Post-batch state (with ENO-EGD metrics)
        violations_after = self.crk.evaluate(self.state, self.trace, None)
        phi_after = self.phi_field.phi(self.state, self.trace, violations_after)
        state_after = self.state.as_dict()
        
        if verbose:
            print(f"\n[POST-BATCH STATE]")
            print(f"  S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
            print(f"  Φ={phi_after:.3f}, Rigidity={self.state.smo.rigidity:.3f}")
            if violations_after:
                print(f"  CRK violations: {violations_after}")
            if temporal_exclusions > 0:
                print(f"  Temporal exclusions: {temporal_exclusions}")
        # ========== PHASE 5: HARD TERMINATION CHECK ==========
        # NEW v12.5: Check if death clock budget exhausted
        if self.death_clock.should_terminate():
            self.death_clock_termination = True
            if verbose:
                print(f"\n{'='*70}")
                print(f"[HARD TERMINATION] Death clock budget exhausted")
                print(f"System terminated at step {self.step_count}")
                print(f"{'='*70}")

        # NEW v12.4: Capture ENO-EGD metrics
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
            # ENO-EGD metrics
            eno_active=eno_active,
            egd_mode=(eno_active and self.egd_steps > 0),
            gated_axes=self.eno.get_gated_axes() if eno_active else None,
            free_axes=self.eno.get_free_axes() if eno_active else None,
            control_map=self.cam.get_control_map() if eno_active else None,
            basis_rotation_attempts=self.egd.rotation_attempts,
            # ADD THESE:
            degradation_progress=self.death_clock.get_degradation_progress(),
            degradation_coefficients=degradation
        )
        
        self.log_file.write(json.dumps({
            'type': 'step',
            **asdict(log)
        }) + '\n')
        self.log_file.flush()
        
        self.step_history.append(log)
        return log

    def run(self, max_steps: int = 100, verbose: bool = True):
        """
        Main triad execution loop.
        
        v12.5: Death clock termination added.
        """
        
        if verbose:
            print("="*70)
            print("UII v12.5 - CONTINUOUS REALITY PERTURBATION + DEATH CLOCK")
            print("Code (CNS) + LLM (Relation) + Browser (Reality)")
            print(f"Running for {max_steps} batch cycles")
            print(f"Death clock budget: {self.death_clock.total_budget} {self.death_clock.trigger_mode}s")
            print(f"Micro-perturbations per check: {self.micro_perturbations_per_check}")
            print(f"Logging: {self.log_path}")
            print("="*70)
            print("\nv12.5 NEW:")
            print("  • Latent death clock (monotonic degradation)")
            print("  • Noise amplification (measurements corrupt)")
            print("  • Rigidity drift (crystallization pressure)")
            print("  • Prediction floor rise (world less knowable)")
            print("  • Attractor chaos (coherence destabilization)")
            print("\nv12.4 PRESERVED:")
            print("  • ENO-EGD zero-error regime handling")
            print("  • All v12.3 fixes (action parity, directives)")
            print("  • All v12.2 learning (overconfidence, variance P)")
            print("="*70)
        
        try:
            for cycle in range(max_steps):
                log = self.step(verbose=verbose)
                
                # Existing termination conditions unchanged
                gradient = self.phi_field.gradient(self.state, self.trace, log.crk_violations)
                gradient_magnitude = np.sqrt(sum(g**2 for g in gradient.values()))
                                
                # NEW v12.5: Death clock termination
                if self.death_clock_termination:
                    break
                
                if not verbose and self.step_count % 10 == 0:
                    llm_rate = self.llm_calls / self.step_count if self.step_count > 0 else 0
                    d = self.death_clock._degradation_progress
                    print(f"[{self.step_count}] LLM: {self.llm_calls} ({llm_rate*100:.1f}%), "
                        f"P: {self.state.P:.3f}, Φ: {log.phi_after:.3f}, Stress: {d:.1%}")
        
        finally:
            trigger_breakdown = {}
            for trigger in self.impossibility_triggers:
                trigger_breakdown[trigger] = trigger_breakdown.get(trigger, 0) + 1

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
                'eno_activations': self.eno_activations,
                'egd_steps': self.egd_steps,
                'basis_rotations': self.basis_rotations,
                # NEW v12.5: Death clock outcome
                'death_clock_termination': self.death_clock_termination,
                'death_clock_budget_used': self.death_clock.current_count,
                'death_clock_budget_remaining': self.death_clock.get_remaining_budget(),
                'death_clock_degradation_final': self.death_clock.get_degradation_progress(),
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
                print(f"Impossibility triggers: {trigger_breakdown}")
                # NEW v12.5
                print(f"Death clock budget used: {self.death_clock.current_count}/{self.death_clock.total_budget}")
                print(f"Degradation at termination: {self.death_clock.get_degradation_progress():.1%}")
                print(f"Termination cause: {'Death clock' if self.death_clock_termination else 'Natural'}")
                print(f"Final state: S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
                print(f"Final rigidity: {self.state.smo.rigidity:.3f}")
                print(f"{'='*70}")
        
        return {
            'steps': self.step_count,
            'llm_calls': self.llm_calls,
            'llm_call_rate': self.llm_calls / self.step_count if self.step_count > 0 else 0,
            'trajectory_enumerations': self.trajectory_enumerations,
            'trajectories_tested': self.trajectories_tested,
            'trajectories_committed': self.trajectories_committed,
            'total_micro_perturbations': self.total_micro_perturbations,
            'death_clock_termination': self.death_clock_termination,
            'death_clock_degradation_final': self.death_clock.get_degradation_progress(),
            'final_state': self.state.as_dict()
        }



# ============================================================
# MODULE 10: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    import os
    
    print("UII v12.5 - Continuous Reality Perturbation + Death Clock")
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
        
        def call(self, prompt: str) -> str:
            import time
            elapsed = time.time() - self.last_call
            if elapsed < 2.1:
                time.sleep(2.1 - elapsed)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            self.last_call = time.time()
            return response.choices[0].message.content
    
    llm_adapter = GroqAdapter()
    
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    micro_per_check = 10
    
    # NEW v12.5: Death clock configuration
    death_clock_budget = 100  # Hard limit
    death_clock_mode = 'step'  # or 'llm'
    
    print(f"\nConfiguration:")
    print(f"  Max Steps: {max_steps}")
    print(f"  Verbose: {verbose}")
    print(f"  Micro-perturbations per check: {micro_per_check}")
    print(f"  Death clock budget: {death_clock_budget} {death_clock_mode}s")
    
    print(f"\nv12.5 NEW:")
    print(f"  1. Latent death clock (degradation from step 1)")
    print(f"  2. Noise amplification (corrupting measurements)")
    print(f"  3. Rigidity drift (crystallization pressure)")
    print(f"  4. Prediction floor (world less knowable)")
    print(f"  5. Attractor chaos (coherence destabilization)")
    print(f"\nv12.4 Changes:")
    print(f"  1. CNS full action parity (tiered affordance selection)")
    print(f"  2. Affordance fingerprint expanded (scroll + elements)")
    print(f"  3. Per-trigger impossibility directives (lookup)")
    print(f"  4. predict_delta expanded (click, fill, type)")
    print(f"\nv12.4-fix Applied:")
    print(f"  1. choose_micro_action: state-driven reflexes restored")
    print(f"  2. choose_micro_action: full 9-action manifold")
    print(f"  3. choose_micro_action: CNS no longer double-constrained")
    
    intelligence = LLMIntelligenceAdapter(llm_adapter)
    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)
    
    triad = MentatTriad(
        intelligence=intelligence,
        reality=reality,
        micro_perturbations_per_check=micro_per_check,
        death_clock_budget=death_clock_budget,      # NEW v12.5
        death_clock_mode=death_clock_mode           # NEW v12.5
    )
    
    report = triad.run(max_steps=max_steps, verbose=verbose)
    
    print(f"\n✓ Execution complete")
    print(f"  Logs: mentat_triad_v12.5_log.jsonl")
    print(f"  LLM call rate: {report['llm_call_rate']*100:.1f}%")
    print(f"  Death clock: {'TERMINATED' if report['death_clock_termination'] else 'Survived'}")
    print(f"  Final degradation: {report['death_clock_degradation_final']:.1%}")
    
    reality.close()




