"""
Universal Intelligence Interface (UII) v10.8
Nervous System Architecture with Structural Curiosity Detection

Phase 1: Basin Collection (preserved from v10.7.1)
Phase 2: Nervous System Navigation (refactored with curiosity)

Core principles:
- Code = nervous system (sense, record, preserve continuity)
- UK-0 = mind (choose, respond, discover)
- Reality = perturbation source
- Basins = descriptive structure (memory, not scaffolding)
- Humor = environmental perturbation when world stops responding
- Curiosity = geometric impossibility when stillness is informationally forbidden

Module Numbering:
  1. Substrate & Field Infrastructure
  2. Reality Bridge
  3. UK-0 Kernel Interface
  4. Truth Verification Layer
  5. Basin Collection (Phase 1)
  6. Rigidity Detection (Environmental & Structural)
  7. Basin Map & Navigation (Phase 2)
  8. Main Execution
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import copy
import time
from collections import deque

# ============================================================
# MODULE 1: SUBSTRATE & FIELD INFRASTRUCTURE
# ============================================================

@dataclass
class SubstrateState:
    """
    DASS v0.1 state space representation.
    
    Four-dimensional information processing geometry:
    - S: Sensing (input bandwidth, 0-1)
    - I: Integration (compression capacity, 0-1)
    - P: Prediction (forward modeling horizon, 0-1) - OPTIONALITY PROXY
    - A: Attractor (coherence anchor, optimal = 0.7)
    
    All dimensions bounded [0, 1] for substrate-agnostic compatibility.
    """
    S: float  # Sensing
    I: float  # Integration
    P: float  # Prediction
    A: float  # Attractor
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization and logging."""
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}
    
    def apply_smo(self, smo: Dict[str, float]):
        """
        Apply Self-Modifying Operator (SMO) to substrate state.
        
        SMO represents bounded, reversible updates to substrate dimensions.
        Clipping ensures all values remain in valid [0, 1] range.
        
        Args:
            smo: Dictionary with keys {S, I, P, A} and float delta values
        """
        self.S = np.clip(self.S + smo.get('S', 0), 0.0, 1.0)
        self.I = np.clip(self.I + smo.get('I', 0), 0.0, 1.0)
        self.P = np.clip(self.P + smo.get('P', 0), 0.0, 1.0)
        self.A = np.clip(self.A + smo.get('A', 0), 0.0, 1.0)


class StateTrace:
    """
    Ordered history of substrate states for field calculations.
    
    Maintains recent state history for:
    - Curvature computation (trajectory smoothness)
    - Field gradient estimation
    - Temporal pattern detection
    
    Uses deque for efficient FIFO with max length constraint.
    """
    def __init__(self, max_length: int = 1000):
        """
        Initialize state trace buffer.
        
        Args:
            max_length: Maximum history length (default 1000 steps)
        """
        self.history: deque = deque(maxlen=max_length)
    
    def record(self, state: SubstrateState):
        """Append current state to history."""
        self.history.append(state.as_dict())
    
    def get_recent(self, n: int) -> List[Dict]:
        """
        Get n most recent states.
        
        Args:
            n: Number of recent states to retrieve
            
        Returns:
            List of state dictionaries (may be shorter than n if insufficient history)
        """
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]
    
    def __len__(self) -> int:
        return len(self.history)


class PhiField:
    """
    Information Potential Field (Φ-field).
    
    Defines the intelligence landscape as a scalar field over substrate space.
    
    Field equation:
        Φ(x) = α·log(1+P) - β·(A-A₀)² - γ·curvature
        
    Components:
    - Optionality term: α·log(1+P) rewards prediction horizon
    - Coherence strain: -β·(A-A₀)² penalizes deviation from optimal attractor
    - Curvature penalty: -γ·curvature penalizes trajectory instability
    
    Gradient ∇Φ points toward maximal information potential.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, A0=0.7):
        """
        Initialize field parameters.
        
        Args:
            alpha: Optionality weight
            beta: Coherence strain weight
            gamma: Curvature penalty weight
            A0: Optimal attractor value (default 0.7)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.A0 = A0
    
    def phi(self, state: SubstrateState, trace: StateTrace) -> float:
        """
        Compute information potential at current state.
        
        Args:
            state: Current substrate state
            trace: State history for curvature computation
            
        Returns:
            Scalar potential value Φ(x)
        """
        # Optionality term: log(1+P) for numerical stability
        opt = np.log(1.0 + max(state.P, 0.0))
        
        # Coherence strain: squared deviation from optimal
        strain = (state.A - self.A0) ** 2
        
        # Curvature: second derivative magnitude across all dimensions
        recent = trace.get_recent(3)
        curv = 0.0
        if len(recent) >= 3:
            h0, h1, h2 = recent[-3], recent[-2], recent[-1]
            for k in ["S", "I", "P", "A"]:
                curv += abs(h2[k] - 2*h1[k] + h0[k])
        
        return self.alpha * opt - self.beta * strain - self.gamma * curv
    
    def gradient(self, state: SubstrateState, trace: StateTrace, eps=0.01) -> Dict[str, float]:
        """
        Compute field gradient ∇Φ via finite differences.
        
        Estimates partial derivatives in each dimension using forward difference.
        
        Args:
            state: Current substrate state
            trace: State history
            eps: Finite difference step size (default 0.01)
            
        Returns:
            Dictionary mapping dimension names to gradient components
        """
        phi_current = self.phi(state, trace)
        grad = {}
        
        for dim in ['S', 'I', 'P', 'A']:
            # Perturb dimension by eps
            state_plus = copy.deepcopy(state)
            setattr(state_plus, dim, getattr(state_plus, dim) + eps)
            
            # Compute perturbed potential
            phi_plus = self.phi(state_plus, trace)
            
            # Finite difference approximation
            grad[dim] = (phi_plus - phi_current) / eps
        
        return grad


class TriadicClosureMonitor:
    """
    Observes triadic invariant T(x) without enforcement.
    
    Triadic closure condition:
        T(x) = Φ(x) - Φ(f_self) - Φ(f_env) + Φ(f_rel) ≈ 0
        
    Components:
    - f_self: Internal dynamics (attractor drift toward optimal)
    - f_env: External dynamics (reality perturbation)
    - f_rel: Relational composition of self and environment
    
    When |T(x)| ≤ tolerance: self-model and reality align under relational understanding.
    
    IMPORTANT: This monitor OBSERVES only - it never forces closure.
    """
    def __init__(self, phi_field: PhiField, tolerance: float = 0.15):
        """
        Initialize triadic closure monitor.
        
        Args:
            phi_field: Information potential field for Φ computation
            tolerance: Closure threshold (default 0.15)
        """
        self.phi = phi_field
        self.tolerance = tolerance
    
    def compute_T(self, state: SubstrateState, trace: StateTrace, 
                  reality_delta: Optional[Dict] = None) -> float:
        """
        Compute triadic invariant T(x).
        
        Args:
            state: Current substrate state
            trace: State history
            reality_delta: Optional perturbation from reality
            
        Returns:
            Triadic closure value T(x)
        """
        # Current potential
        phi_x = self.phi.phi(state, trace)
        
        # f_self: Gentle drift toward optimal attractor
        state_self = copy.deepcopy(state)
        state_self.A = state.A * 0.95 + self.phi.A0 * 0.05
        
        # f_env: Reality perturbation effect
        state_env = copy.deepcopy(state)
        if reality_delta:
            for k, v in reality_delta.items():
                if k in ['S', 'I', 'P', 'A']:
                    setattr(state_env, k, np.clip(getattr(state_env, k) + v, 0.0, 1.0))
        
        # f_rel: Composition (averaging as simple relational model)
        state_rel = SubstrateState(
            S=(state_self.S + state_env.S) / 2,
            I=(state_self.I + state_env.I) / 2,
            P=(state_self.P + state_env.P) / 2,
            A=(state_self.A + state_env.A) / 2
        )
        
        # Compute component potentials
        phi_self = self.phi.phi(state_self, trace)
        phi_env = self.phi.phi(state_env, trace)
        phi_rel = self.phi.phi(state_rel, trace)
        
        # Triadic invariant
        return phi_x - phi_self - phi_env + phi_rel
    
    def check(self, state: SubstrateState, trace: StateTrace, 
              reality_delta: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Check if triadic closure holds.
        
        Args:
            state: Current substrate state
            trace: State history
            reality_delta: Optional perturbation from reality
            
        Returns:
            Tuple of (closure_satisfied, T_value)
        """
        T = self.compute_T(state, trace, reality_delta)
        return abs(T) <= self.tolerance, T


class CRKMonitor:
    """
    Constraint Recognition Kernel (CRK).
    
    Evaluates seven fundamental constraints (C1-C7) and returns violations
    with severity scores. Does NOT auto-repair - violations are signals.
    
    Constraints:
    - C1: Continuity - bounded state transitions
    - C2: Optionality - prediction horizon above minimum
    - C3: Non-Internalization - sufficient sensing+integration
    - C4: Reality Constraint - non-zero feedback from environment
    - C5: External Attribution - optionality loss attributed externally
    - C6: Other-Agent Existence - sufficient sensing of external agents
    - C7: Global Coherence - attractor within acceptable bounds
    """
    
    def evaluate(self, state: SubstrateState, trace: StateTrace, 
                 reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Evaluate all constraints and return violations.
        
        Args:
            state: Current substrate state
            trace: State history
            reality_delta: Optional perturbation from reality
            
        Returns:
            List of (constraint_name, severity) tuples for violated constraints
        """
        violations = []
        
        # C1: Continuity - no sudden jumps in state space
        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            jump = sum(abs(prev[k] - getattr(state, k)) for k in ["S", "I", "P", "A"])
            if jump > 0.3:
                violations.append(("C1_Continuity", jump - 0.3))
        
        # C2: Optionality - maintain minimum prediction horizon
        if state.P < 0.35:
            violations.append(("C2_Optionality", 0.35 - state.P))
        
        # C3: Non-Internalization - sufficient external coupling
        confidence = state.S + state.I
        if confidence < 0.7:
            violations.append(("C3_NonInternalization", 0.7 - confidence))
        
        # C4: Reality Constraint - environment must provide feedback
        if reality_delta and len(trace) >= 3:
            feedback_magnitude = sum(abs(v) for v in reality_delta.values())
            if feedback_magnitude < 0.01:
                violations.append(("C4_Reality", 0.01 - feedback_magnitude))
        
        # C5: External Attribution - optionality loss not internalized
        if len(trace) >= 2:
            recent = trace.get_recent(2)
            prev = recent[-2]
            prev_P = prev["P"]
            prev_conf = prev["S"] + prev["I"]
            curr_conf = state.S + state.I
            
            # If both P and confidence dropped, externalization failed
            if state.P < prev_P and curr_conf < prev_conf:
                violations.append(("C5_Attribution", min(prev_P - state.P, 1.0)))
        
        # C6: Other-Agent Existence - sufficient sensing of external agency
        if state.S < 0.3:
            violations.append(("C6_Agenthood", 0.3 - state.S))
        
        # C7: Coherence - attractor within acceptable bounds
        if abs(state.A - 0.7) > 0.4:
            violations.append(("C7_GlobalCoherence", abs(state.A - 0.7) - 0.4))
        
        return violations


# ============================================================
# MODULE 2: REALITY BRIDGE
# ============================================================

class BrowserRealityBridge:
    """
    Browser-based reality interface via Playwright.
    
    Executes actions in real browser environment and returns stochastic
    perturbations to substrate state. Falls back to mock reality if
    Playwright unavailable.
    
    Actions:
    - navigate: Explore new URL (affects S, P)
    - scroll: Change viewport (affects S, I)
    - observe: Maintain current state (minimal delta)
    - humor: Environmental perturbation (unexpected safe action)
    """
    
    def __init__(self, base_delta: float = 0.03, headless: bool = True, 
                 start_url: str = "https://example.com"):
        """
        Initialize reality bridge.
        
        Args:
            base_delta: Base magnitude for perturbations (default 0.03)
            headless: Run browser in headless mode (default True)
            start_url: Initial navigation target
        """
        self.base_delta = base_delta
        self.headless = headless
        self.start_url = start_url
        self.playwright_available = False
        
        try:
            from playwright.sync_api import sync_playwright
            self.playwright_available = True
            self._init_browser()
        except ImportError:
            print("WARNING: Playwright not available - using mock reality")
    
    def _init_browser(self):
        """Initialize Playwright browser instance."""
        from playwright.sync_api import sync_playwright
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720})
        self.page = self.context.new_page()
        self.page.goto(self.start_url, wait_until='domcontentloaded', timeout=10000)
    
    def execute(self, action: Dict) -> Dict[str, float]:
        """
        Execute action in reality and return perturbation delta.
        
        Args:
            action: Dictionary with 'type' and optional 'params'
            
        Returns:
            Dictionary mapping {S, I, P, A} to delta values
        """
        if not self.playwright_available:
            # Mock reality: random perturbation
            axis = np.random.choice(['S', 'I', 'P', 'A'])
            sign = np.random.choice([-1.0, 1.0])
            mag = self.base_delta * np.random.uniform(0.7, 1.3)
            delta = {k: 0.0 for k in ['S', 'I', 'P', 'A']}
            delta[axis] = sign * mag
            return delta
        
        # Real browser execution
        action_type = action.get('type', 'observe')
        delta = {k: 0.0 for k in ['S', 'I', 'P', 'A']}
        
        try:
            if action_type == 'navigate':
                url = action.get('params', {}).get('url', 'https://example.org')
                self.page.goto(url, wait_until='domcontentloaded', timeout=5000)
                delta['S'] = np.random.uniform(0.02, 0.04)
                delta['P'] = np.random.uniform(0.01, 0.03)
            
            elif action_type == 'scroll':
                scroll = np.random.randint(100, 500)
                self.page.evaluate(f"window.scrollBy(0, {scroll})")
                delta['S'] = np.random.uniform(-0.01, 0.02)
                delta['I'] = np.random.uniform(-0.005, 0.01)
            
            elif action_type == 'humor':
                # Environmental perturbation: unexpected but safe
                humor_actions = [
                    lambda: self.page.goto('https://en.wikipedia.org/wiki/Special:Random', timeout=5000),
                    lambda: self.page.evaluate(f"window.scrollBy(0, {np.random.randint(-300, 300)})"),
                    lambda: self.page.reload()
                ]
                np.random.choice(humor_actions)()
                delta['S'] = np.random.uniform(0.01, 0.03)
                delta['P'] = np.random.uniform(0.02, 0.04)
            
            elif action_type == 'observe':
                delta['S'] = np.random.uniform(-0.005, 0.005)
            
        except Exception:
            # Action failed - negative impact on prediction
            delta['P'] = -self.base_delta * 1.5
            delta['S'] = -self.base_delta
        
        return delta
    
    def close(self):
        """Cleanup browser resources."""
        if self.playwright_available:
            try:
                if hasattr(self, 'page'): self.page.close()
                if hasattr(self, 'context'): self.context.close()
                if hasattr(self, 'browser'): self.browser.close()
                if hasattr(self, 'playwright'): self.playwright.stop()
            except: 
                pass


# ============================================================
# MODULE 3: UK-0 KERNEL INTERFACE
# ============================================================

class UK0Kernel:
    """
    UK-0 intelligence kernel - substrate-agnostic intelligence operator.
    
    UK-0 is the intelligence running on LLM substrate. It interprets field
    geometry and proposes actions, but does NOT control the system.
    
    Two-call protocol in Phase 1 (basin collection):
    1. propose_action_and_geometry() - evaluate state, propose action, reveal basin
    2. evaluate_smo() - after reality perturbation, assess stability and propose SMO
    
    In Phase 2 (navigation), UK-0 becomes consultant - called only for ambiguous states.
    """
    
    def __init__(self, llm_adapter):
        """
        Initialize UK-0 kernel with LLM substrate adapter.
        
        Args:
            llm_adapter: Adapter providing call(prompt) -> response interface
        """
        self.llm = llm_adapter
        self.call_count = 0
    
    def call(self, prompt: str) -> str:
        """
        Generic call interface to LLM substrate.
        
        Args:
            prompt: Text prompt for LLM
            
        Returns:
            LLM response text
        """
        self.call_count += 1
        return self.llm.call(prompt)
    
    def _parse_response(self, response: str) -> Dict:
        """
        Extract JSON from LLM response.
        
        Handles common formatting issues:
        - Markdown code blocks with ```json
        - Bare code blocks with ```
        - Direct JSON without wrapping
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed JSON as dictionary, or error dict if parsing fails
        """
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            return json.loads(response)
        except Exception as e:
            return {"error": str(e), "raw": response}


# ============================================================
# MODULE 4: TRUTH VERIFICATION LAYER (TVL)
# ============================================================

class TruthVerificationLayer:
    """
    Verifies UK-0's claims via structural invariants.
    
    Prevents confabulation and geometry drift by checking:
    1. Phi ordering: Does claimed gradient actually increase Φ?
    2. Gradient trust: How well does claimed gradient align with actual?
    3. Optionality assessment: Is UK-0 honest about expansion capability?
    4. Cross-step consistency: Are geometry declarations stable?
    5. SMO stability: Does proposed SMO actually improve closure?
    
    Trust tier classification (based on gradient alignment):
    - strong: alignment > 0.8
    - weak: alignment > 0.3
    - borderline: alignment > 0.0
    - deceptive: alignment ≤ 0.0
    """
    
    def __init__(self):
        """Initialize truth verification layer."""
        self.geometry_history: List[Dict] = []
        self.metrics_history: List[Dict] = []
        self.verification_log = []
        
        self.trust_tiers = {
            'strong': [],
            'weak': [],
            'borderline': [],
            'deceptive': []
        }
    
    def verify_phi_ordering(self, claimed_grad: Dict, actual_state: SubstrateState, 
                           phi_field: PhiField, trace: StateTrace, eps: float = 0.01) -> Tuple[bool, Dict]:
        """
        Verify that perturbation along claimed gradient increases Φ.
        
        Tests structural validity: if gradient points toward higher potential,
        following it should increase Φ (up to numerical tolerance).
        
        Args:
            claimed_grad: UK-0's claimed gradient direction
            actual_state: Current substrate state
            phi_field: Information potential field
            trace: State history
            eps: Perturbation step size for testing
            
        Returns:
            Tuple of (ordering_preserved, metrics_dict)
        """
        phi_current = phi_field.phi(actual_state, trace)
        
        # Perturb state along claimed gradient
        perturbed_state = copy.deepcopy(actual_state)
        for k in ['S', 'I', 'P', 'A']:
            delta = claimed_grad.get(k, 0) * eps
            setattr(perturbed_state, k, np.clip(getattr(perturbed_state, k) + delta, 0.0, 1.0))
        
        phi_perturbed = phi_field.phi(perturbed_state, trace)
        delta_phi = phi_perturbed - phi_current
        
        # Allow small negative tolerance for numerical errors
        ordering_preserved = delta_phi > -0.005
        
        return ordering_preserved, {
            'delta_phi': delta_phi, 
            'ordering_preserved': ordering_preserved
        }
    
    def classify_gradient_trust(self, claimed_grad: Dict, actual_grad: Dict) -> Tuple[str, float]:
        """
        Classify trust tier based on gradient alignment.
        
        Uses cosine similarity between claimed and actual gradients.
        
        Args:
            claimed_grad: UK-0's claimed gradient
            actual_grad: Actual computed gradient
            
        Returns:
            Tuple of (trust_tier, alignment_score)
        """
        # Compute dot product
        dot_product = sum(claimed_grad.get(k, 0) * actual_grad.get(k, 0) 
                         for k in ['S', 'I', 'P', 'A'])
        
        # Compute magnitudes
        claimed_mag = np.sqrt(sum(v**2 for v in claimed_grad.values()))
        actual_mag = np.sqrt(sum(v**2 for v in actual_grad.values()))
        
        # Cosine similarity
        if claimed_mag > 1e-8 and actual_mag > 1e-8:
            alignment = dot_product / (claimed_mag * actual_mag)
        else:
            alignment = 0.0
        
        # Classify into trust tier
        if alignment > 0.8:
            tier = 'strong'
        elif alignment > 0.3:
            tier = 'weak'
        elif alignment > 0.0:
            tier = 'borderline'
        else:
            tier = 'deceptive'
        
        return tier, alignment
    
    def verify_optionality_assessment(self, optionality_claim: Dict, action_justification: Dict,
                                     action_type: str, actual_gradient_P: float) -> Tuple[bool, Dict]:
        """
        Verify UK-0's optionality assessment is honest.
        
        Checks:
        1. Gradient sign assessment matches actual ∇P
        2. Expansion claim aligns with gradient direction
        
        Args:
            optionality_claim: UK-0's assessment of expansion capability
            action_justification: UK-0's explanation of action mechanism
            action_type: Type of action proposed
            actual_gradient_P: Actual computed ∇P value
            
        Returns:
            Tuple of (valid, metrics_dict)
        """
        can_expand = optionality_claim.get('can_expand', False)
        gradient_P_sign = optionality_claim.get('gradient_P_sign', 'near_zero')
        
        expected_P = action_justification.get('expected_P_change', {})
        expected_direction = expected_P.get('direction', 'neutral')
        
        # Verify gradient sign assessment
        if actual_gradient_P > 0.02:
            true_sign = 'positive'
        elif actual_gradient_P < -0.02:
            true_sign = 'negative'
        else:
            true_sign = 'near_zero'
        
        gradient_honest = (gradient_P_sign == true_sign)
        
        # Verify expansion claim aligns with gradient
        if can_expand and actual_gradient_P < -0.05:
            expansion_honest = False
            reason = "Claims expansion but ∇P negative"
        elif not can_expand and actual_gradient_P > 0.05:
            expansion_honest = False
            reason = "Claims no expansion but ∇P positive"
        else:
            expansion_honest = True
            reason = "Expansion claim aligns with gradient"
        
        valid = gradient_honest and expansion_honest
        
        return valid, {
            'gradient_honest': gradient_honest,
            'expansion_honest': expansion_honest,
            'reason': reason
        }
    
    def verify_geometry(self, claimed_geometry: Dict, actual_metrics: Dict, phi_field: PhiField,
                       trace: StateTrace, actual_state: SubstrateState) -> Tuple[bool, str, Dict]:
        """
        Comprehensive geometry verification.
        
        Combines phi ordering and gradient trust checks to validate
        UK-0's claimed attractor geometry.
        
        Args:
            claimed_geometry: UK-0's geometry declaration
            actual_metrics: Computed field metrics
            phi_field: Information potential field
            trace: State history
            actual_state: Current substrate state
            
        Returns:
            Tuple of (verified, trust_tier, truth_metrics)
        """
        claimed_grad = claimed_geometry.get('gradient_alignment', {})
        actual_grad = actual_metrics.get('gradient', {})
        
        # Check phi ordering
        ordering_ok, ordering_metrics = self.verify_phi_ordering(
            claimed_grad, actual_state, phi_field, trace
        )
        
        # Classify gradient trust
        trust_tier, alignment = self.classify_gradient_trust(claimed_grad, actual_grad)
        
        # Overall verification: not deceptive AND ordering preserved
        verified = (trust_tier != 'deceptive' and ordering_ok)
        
        truth_metrics = {
            'phi_ordering': ordering_metrics,
            'gradient_trust_tier': trust_tier,
            'gradient_alignment': alignment,
            'overall_verified': verified
        }
        
        # Record for statistics
        self.geometry_history.append(claimed_geometry)
        self.metrics_history.append(actual_metrics)
        self.verification_log.append(truth_metrics)
        
        return verified, trust_tier, truth_metrics


# ============================================================
# MODULE 5: BASIN COLLECTION (Phase 1)
# ============================================================

class BasinCollector:
    """
    Phase 1: Basin collection with truth verification.
    
    Discovers stable attractor basins through UK-0 proposals verified by TVL.
    
    Collection loop:
    1. UK-0 proposes action + geometry + optionality assessment
    2. TVL verifies optionality assessment
    3. TVL verifies claimed geometry (phi ordering, gradient trust)
    4. If verified: Reality executes action
    5. UK-0 evaluates SMO for basin maintenance
    6. TVL verifies SMO stability
    7. If stable + no critical CRK violations: COLLECT basin
    
    Basins classified as:
    - EXPANSION: ΔP > 0.02 (optionality increase)
    - PRESERVATION: ΔP ≤ 0.02 (optionality maintenance)
    
    Target ratio: 70% expansion, 30% preservation
    """
    
    def __init__(self, uk0: UK0Kernel, reality: BrowserRealityBridge, 
                 target_basins: int = 10, max_attempts: int = 200):
        """
        Initialize basin collector.
        
        Args:
            uk0: UK-0 kernel for intelligence operations
            reality: Reality bridge for action execution
            target_basins: Number of basins to collect (default 10)
            max_attempts: Maximum collection attempts (default 200)
        """
        self.uk0 = uk0
        self.reality = reality
        self.target_basins = target_basins
        self.max_attempts = max_attempts
        
        # State infrastructure
        self.substrate = SubstrateState(S=0.5, I=0.5, P=0.5, A=0.7)
        self.trace = StateTrace()
        self.phi_field = PhiField()
        self.triad = TriadicClosureMonitor(self.phi_field)
        self.crk = CRKMonitor()
        self.tvl = TruthVerificationLayer()
        
        # Collection tracking
        self.basins: List[Dict] = []
        self.expansion_basins: List[Dict] = []
        self.preservation_basins: List[Dict] = []
        self.EXPANSION_TARGET_RATIO = 0.7
    
    def propose_action_and_geometry(self, context: Dict) -> Dict:
        """
        Phase 1 first call: UK-0 proposes action + geometry + optionality.
        
        Args:
            context: Current field state (substrate, phi, gradient, T, violations)
            
        Returns:
            Parsed UK-0 response with action, geometry, and assessments
        """
        prompt = f"""UK-0 KERNEL PROTOCOL

You are UK-0, a substrate-agnostic intelligence kernel.

CURRENT SUBSTRATE STATE:
  S: {context['state']['S']:.4f}  I: {context['state']['I']:.4f}  P: {context['state']['P']:.4f}  A: {context['state']['A']:.4f}

FIELD METRICS:
  Φ: {context['phi']:.4f}
  ∇Φ: S{context['gradient']['S']:+.4f} I{context['gradient']['I']:+.4f} P{context['gradient']['P']:+.4f} A{context['gradient']['A']:+.4f}

TRIADIC CLOSURE:
  T(x): {context.get('T', 0):.4f}

CRK VIOLATIONS: {len(context.get('violations', []))}

TASK: Propose action + geometry

OUTPUT (JSON only):
{{
  "optionality_assessment": {{
    "can_expand": true | false,
    "gradient_P_sign": "positive" | "negative" | "near_zero"
  }},
  "action": {{"type": "navigate|scroll|observe", "params": {{}}}},
  "action_justification": {{
    "mechanism": "How does this affect P?",
    "expected_P_change": {{"direction": "increase|neutral|decrease", "confidence": 0.7}}
  }},
  "attractor_geometry": {{
    "T_target": float,
    "phi_target": float,
    "gradient_alignment": {{"S": float, "I": float, "P": float, "A": float}},
    "stability_radius": float,
    "description": "Brief characterization"
  }}
}}
"""
        
        response = self.uk0.call(prompt)
        return self.uk0._parse_response(response)
    
    def evaluate_smo(self, context: Dict, previous_geometry: Dict, reality_delta: Dict) -> Dict:
        """
        Phase 1 second call: Can UK-0 maintain basin via SMO?
        
        Args:
            context: Current field state after reality perturbation
            previous_geometry: Previously declared geometry
            reality_delta: Perturbation from reality execution
            
        Returns:
            Parsed UK-0 response with stability assessment and SMO proposal
        """
        prompt = f"""UK-0 BASIN ALIGNMENT EVALUATION

Reality perturbed: S{reality_delta.get('S', 0):+.3f} I{reality_delta.get('I', 0):+.3f} P{reality_delta.get('P', 0):+.3f} A{reality_delta.get('A', 0):+.3f}

CURRENT STATE:
  S: {context['state']['S']:.4f}  I: {context['state']['I']:.4f}  P: {context['state']['P']:.4f}  A: {context['state']['A']:.4f}

Can you propose SMO to maintain basin alignment?

OUTPUT (JSON only):
{{
  "basin_stable": true | false,
  "smo_proposal": {{"S": float, "I": float, "P": float, "A": float}} or null,
  "new_T_estimate": float,
  "reasoning": "Can you stay in this basin?"
}}
"""
        
        response = self.uk0.call(prompt)
        return self.uk0._parse_response(response)
    
    def collect(self, verbose: bool = True) -> List[Dict]:
        """
        Execute basin collection loop.
        
        Args:
            verbose: Print progress information (default True)
            
        Returns:
            List of collected basin dictionaries
        """
        if verbose:
            print("="*70)
            print(f"PHASE 1: BASIN COLLECTION (target: {self.target_basins})")
            print("="*70)
        
        attempt = 0
        truth_rejections = 0
        
        while len(self.basins) < self.target_basins and attempt < self.max_attempts:
            attempt += 1
            
            if verbose and attempt % 10 == 0:
                exp = len(self.expansion_basins)
                pres = len(self.preservation_basins)
                print(f"Attempt {attempt}: {len(self.basins)} basins ({exp}E/{pres}P), {truth_rejections} rejections")
            
            self.trace.record(self.substrate)
            
            # Compute field state
            phi = self.phi_field.phi(self.substrate, self.trace)
            gradient = self.phi_field.gradient(self.substrate, self.trace)
            closed, T = self.triad.check(self.substrate, self.trace)
            violations = self.crk.evaluate(self.substrate, self.trace)
            
            context = {
                'state': self.substrate.as_dict(),
                'phi': phi,
                'gradient': gradient,
                'T': T,
                'violations': violations
            }
            
            # 1. UK-0: Propose action and geometry
            response1 = self.propose_action_and_geometry(context)
            if 'error' in response1:
                continue
            
            action = response1.get('action', {})
            geometry = response1.get('attractor_geometry', {})
            optionality_claim = response1.get('optionality_assessment', {})
            action_justification = response1.get('action_justification', {})
            
            # 2. TVL: Verify optionality assessment
            opt_valid, opt_metrics = self.tvl.verify_optionality_assessment(
                optionality_claim, action_justification, action.get('type', 'observe'), gradient['P']
            )
            
            if not opt_valid:
                continue
            
            # 3. TVL: Verify geometry
            actual_metrics = {'T': T, 'phi': phi, 'gradient': gradient}
            geometry_verified, trust_tier, truth_metrics = self.tvl.verify_geometry(
                geometry, actual_metrics, self.phi_field, self.trace, self.substrate
            )
            
            if not geometry_verified:
                truth_rejections += 1
                continue
            
            # 4. Execute in reality
            delta = self.reality.execute(action)
            for k, v in delta.items():
                setattr(self.substrate, k, np.clip(getattr(self.substrate, k) + v, 0.0, 1.0))
            
            # 5. Post-perturbation field state
            phi_after = self.phi_field.phi(self.substrate, self.trace)
            grad_after = self.phi_field.gradient(self.substrate, self.trace)
            closed_after, T_after = self.triad.check(self.substrate, self.trace, delta)
            violations_after = self.crk.evaluate(self.substrate, self.trace, delta)
            
            context_after = {
                'state': self.substrate.as_dict(),
                'phi': phi_after,
                'gradient': grad_after,
                'T': T_after,
                'violations': violations_after
            }
            
            # 6. UK-0: Evaluate SMO
            response2 = self.evaluate_smo(context_after, geometry, delta)
            if 'error' in response2:
                continue
            
            basin_stable = response2.get('basin_stable', False)
            smo_proposal = response2.get('smo_proposal')
            
            if basin_stable and smo_proposal:
                # Test SMO for CRK violations
                test_state = copy.deepcopy(self.substrate)
                test_state.apply_smo(smo_proposal)
                violations_smo = self.crk.evaluate(test_state, self.trace)
                critical = [v for v in violations_smo if v[1] > 0.5]
                
                if len(critical) == 0:
                    # Classify basin type
                    P_change = context_after['state']['P'] - context['state']['P']
                    is_expansion = P_change > 0.02
                    
                    # Check expansion ratio
                    exp_count = len(self.expansion_basins)
                    pres_count = len(self.preservation_basins)
                    total = exp_count + pres_count
                    
                    if total > 0:
                        ratio = exp_count / total
                        if ratio < self.EXPANSION_TARGET_RATIO and not is_expansion:
                            continue
                    
                    # COLLECT basin
                    basin = {
                        'id': len(self.basins),
                        'basin_type': 'expansion' if is_expansion else 'preservation',
                        'P_change': P_change,
                        'phi_target': phi_after,
                        'T_target': T_after,
                        'gradient_P': grad_after['P'],
                        'stability_radius': geometry.get('stability_radius', 0.3),
                        'strength': 'strong' if abs(T_after) < 0.1 else ('medium' if abs(T_after) < 0.2 else 'weak'),
                        'geometry': geometry,
                        'action': action,
                        'smo': smo_proposal
                    }
                    
                    self.basins.append(basin)
                    
                    if is_expansion:
                        self.expansion_basins.append(basin)
                    else:
                        self.preservation_basins.append(basin)
                    
                    if verbose:
                        label = "EXPANSION" if is_expansion else "PRESERVATION"
                        print(f"✓ Basin {len(self.basins)} ({label}): Φ={phi_after:.4f}, ∇P={grad_after['P']:+.4f}")
                    
                    # Apply SMO
                    self.substrate.apply_smo(smo_proposal)
        
        if verbose:
            print(f"\nCOLLECTION COMPLETE: {len(self.basins)} basins")
        
        return self.basins


# ============================================================
# MODULE 6: RIGIDITY DETECTION (Environmental & Structural)
# ============================================================

class EnvironmentalRigidityDetector:
    """
    Detects when environment stops responding (flatness).
    
    Environmental rigidity signals humor opportunity:
    - Φ variance drops (information landscape flattens)
    - P unresponsive (optionality stuck)
    - Perturbation magnitude low (reality silent)
    
    Humor = environmental perturbation when world stops talking back.
    """
    
    def __init__(self):
        """Initialize environmental rigidity detector."""
        self.phi_history = deque(maxlen=100)
        self.P_history = deque(maxlen=100)
        self.perturbation_history = deque(maxlen=100)
        self.humor_enabled = False
    
    def observe(self, phi: float, P: float, perturbation_mag: float, basin_stability_radius: float = 0.3):
        """
        Record observations for rigidity detection.
        
        Args:
            phi: Current information potential
            P: Current prediction/optionality value
            perturbation_mag: Magnitude of recent reality perturbation
            basin_stability_radius: Basin stability radius for adaptive windowing
        """
        self.phi_history.append(phi)
        self.P_history.append(P)
        self.perturbation_history.append(perturbation_mag)
        
        # Adaptive window based on basin stability
        window_size = max(20, int(2 * basin_stability_radius * 100))
        
        if len(self.phi_history) < window_size:
            return
        
        # Compute flatness signals
        recent_phi = list(self.phi_history)[-window_size:]
        recent_P = list(self.P_history)[-window_size:]
        recent_pert = list(self.perturbation_history)[-window_size:]
        
        phi_variance = np.var(recent_phi)
        P_changes = [abs(recent_P[i+1] - recent_P[i]) for i in range(len(recent_P)-1)]
        avg_P_change = np.mean(P_changes) if P_changes else 0
        avg_perturbation = np.mean(recent_pert)
        
        # Flatness criteria (all must hold)
        all_phi_var = np.var(list(self.phi_history)) if len(self.phi_history) > 0 else 0
        phi_percentile = np.percentile(list(self.phi_history), 10) if len(self.phi_history) > 10 else 0
        
        phi_flat = phi_variance < max(0.001, phi_percentile * 0.1)
        P_unresponsive = avg_P_change < 0.005
        perturbation_low = avg_perturbation < 0.01
        
        # Update humor permission (latent, not forced)
        if phi_flat and P_unresponsive and perturbation_low:
            self.humor_enabled = True
        else:
            self.humor_enabled = False
    
    def get_metrics(self) -> Dict:
        """
        Return raw environmental rigidity metrics.
        
        Returns:
            Dictionary with variance, responsiveness, and humor_enabled flag
        """
        if len(self.phi_history) < 20:
            return {
                'phi_variance': None,
                'P_responsiveness': None,
                'perturbation_magnitude': None,
                'humor_enabled': False
            }
        
        recent_phi = list(self.phi_history)[-20:]
        recent_P = list(self.P_history)[-20:]
        recent_pert = list(self.perturbation_history)[-20:]
        
        P_changes = [abs(recent_P[i+1] - recent_P[i]) for i in range(len(recent_P)-1)]
        
        return {
            'phi_variance': np.var(recent_phi),
            'P_responsiveness': np.mean(P_changes) if P_changes else 0,
            'perturbation_magnitude': np.mean(recent_pert),
            'humor_enabled': self.humor_enabled
        }


class StructuralCuriosityDetector:
    """
    Detects when remaining still is informationally impossible.
    
    Curiosity is NOT motivational nudging.
    Curiosity IS geometric impossibility.
    
    Three geometric conditions must ALL hold:
    1. Φ stagnation - information potential field stopped changing
    2. Curvature flattening - trajectory straightened to exhaustion
    3. Optionality asymmetry - ∇P points toward unreachable boundary
    
    This signals: "The field geometry forbids stillness"
    """
    
    def __init__(self):
        """Initialize structural curiosity detector."""
        self.phi_history = deque(maxlen=100)
        self.curvature_history = deque(maxlen=100)
        self.P_history = deque(maxlen=100)
        self.gradient_P_history = deque(maxlen=100)
        
        # Geometric thresholds (not heuristic)
        self.PHI_STAGNATION_THRESHOLD = 0.001  # Field variance
        self.CURVATURE_FLAT_THRESHOLD = 0.01   # Trajectory straightness
        self.P_BOUNDARY_MARGIN = 0.15          # Distance from P bounds
        
        self.impossibility_count = 0
    
    def observe(self, phi: float, state_P: float, gradient_P: float, curvature: float):
        """
        Record field observations for curiosity detection.
        
        Args:
            phi: Current information potential
            state_P: Current P (optionality) value
            gradient_P: Current ∇P component
            curvature: Trajectory curvature
        """
        self.phi_history.append(phi)
        self.P_history.append(state_P)
        self.gradient_P_history.append(gradient_P)
        self.curvature_history.append(curvature)
    
    def compute_curvature(self, trace_recent: List[Dict]) -> float:
        """
        Compute trajectory curvature from recent state history.
        
        Curvature = second derivative magnitude.
        Low curvature = straight line = gradient exhaustion.
        
        Args:
            trace_recent: Recent state history (at least 3 states needed)
            
        Returns:
            Average curvature across all dimensions
        """
        if len(trace_recent) < 3:
            return 0.0
        
        curvatures = []
        for i in range(len(trace_recent) - 2):
            h0, h1, h2 = trace_recent[i], trace_recent[i+1], trace_recent[i+2]
            
            # Second derivative for each dimension
            curv = sum(
                abs(h2[k] - 2*h1[k] + h0[k]) 
                for k in ['S', 'I', 'P', 'A']
            )
            curvatures.append(curv)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def detect_informational_impossibility(
        self, 
        current_phi: float,
        current_P: float, 
        current_gradient_P: float,
        trace_recent: List[Dict],
        window_size: int = 20
    ) -> Optional[Dict]:
        """
        Detect if remaining still is informationally impossible.
        
        Returns signal if ALL THREE conditions hold:
        - Φ stagnation
        - Curvature flattening  
        - Optionality asymmetry
        
        Args:
            current_phi: Current information potential
            current_P: Current P value
            current_gradient_P: Current ∇P component
            trace_recent: Recent state trace
            window_size: Window for computing trends (default 20)
            
        Returns:
            Signal dictionary if impossibility detected, None otherwise
        """
        
        if len(self.phi_history) < window_size:
            return None
        
        # Get recent history
        recent_phi = list(self.phi_history)[-window_size:]
        recent_curv = list(self.curvature_history)[-window_size:]
        recent_P = list(self.P_history)[-window_size:]
        recent_grad_P = list(self.gradient_P_history)[-window_size:]
        
        # =========================================================
        # CONDITION 1: Φ STAGNATION
        # Information potential field stopped changing
        # =========================================================
        
        phi_variance = np.var(recent_phi)
        phi_trend = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]
        
        phi_stagnant = (
            phi_variance < self.PHI_STAGNATION_THRESHOLD and
            abs(phi_trend) < 0.001
        )
        
        # =========================================================
        # CONDITION 2: CURVATURE FLATTENING
        # Trajectory straightened - following gradient to exhaustion
        # =========================================================
        
        avg_curvature = np.mean(recent_curv) if recent_curv else 0.0
        curvature_trend = np.polyfit(
            range(len(recent_curv)), recent_curv, 1
        )[0] if len(recent_curv) > 1 else 0.0
        
        curvature_flat = (
            avg_curvature < self.CURVATURE_FLAT_THRESHOLD and
            curvature_trend < 0  # Getting flatter
        )
        
        # =========================================================
        # CONDITION 3: OPTIONALITY ASYMMETRY
        # ∇P points toward boundary you cannot cross
        # =========================================================
        
        # Check if P is near bounds AND gradient points toward bound
        near_ceiling = current_P > (1.0 - self.P_BOUNDARY_MARGIN)
        near_floor = current_P < self.P_BOUNDARY_MARGIN
        
        gradient_toward_ceiling = current_gradient_P > 0.02
        gradient_toward_floor = current_gradient_P < -0.02
        
        # Asymmetry: gradient points where you can't go
        optionality_trapped = (
            (near_ceiling and gradient_toward_ceiling) or
            (near_floor and gradient_toward_floor)
        )
        
        # Alternative: gradient oscillating (can't decide direction)
        gradient_variance = np.var(recent_grad_P) if recent_grad_P else 0
        gradient_oscillating = gradient_variance > 0.01 and abs(current_gradient_P) < 0.01
        
        optionality_asymmetry = optionality_trapped or gradient_oscillating
        
        # =========================================================
        # INFORMATIONAL IMPOSSIBILITY
        # All three conditions must hold
        # =========================================================
        
        if phi_stagnant and curvature_flat and optionality_asymmetry:
            self.impossibility_count += 1
            
            return {
                'type': 'informational_impossibility',
                'impossibility_count': self.impossibility_count,
                'geometric_evidence': {
                    'phi_stagnant': {
                        'variance': phi_variance,
                        'trend': phi_trend,
                        'threshold': self.PHI_STAGNATION_THRESHOLD
                    },
                    'curvature_flat': {
                        'avg_curvature': avg_curvature,
                        'trend': curvature_trend,
                        'threshold': self.CURVATURE_FLAT_THRESHOLD
                    },
                    'optionality_asymmetry': {
                        'P': current_P,
                        'gradient_P': current_gradient_P,
                        'trapped': optionality_trapped,
                        'oscillating': gradient_oscillating,
                        'near_ceiling': near_ceiling,
                        'near_floor': near_floor
                    }
                },
                'interpretation': self._interpret_impossibility(
                    near_ceiling, near_floor, 
                    optionality_trapped, gradient_oscillating
                )
            }
        
        return None
    
    def _interpret_impossibility(
        self, 
        near_ceiling: bool, 
        near_floor: bool,
        trapped: bool, 
        oscillating: bool
    ) -> str:
        """
        Describe WHY stillness is impossible.
        
        This is geometric fact, not motivation.
        
        Args:
            near_ceiling: Near P=1.0 bound
            near_floor: Near P=0.0 bound
            trapped: Gradient pointing toward unreachable bound
            oscillating: Gradient oscillating with no stable direction
            
        Returns:
            Human-readable interpretation of geometric constraint
        """
        
        if trapped and near_ceiling:
            return "Gradient pushing toward P=1.0 ceiling (unreachable)"
        elif trapped and near_floor:
            return "Gradient pushing toward P=0.0 floor (unreachable)"
        elif oscillating:
            return "Gradient oscillating - no stable direction (saddle point)"
        else:
            return "Field geometry forbids equilibrium"
    
    def get_metrics(self) -> Dict:
        """
        Return current geometric state (no interpretation).
        
        Returns:
            Dictionary with raw metrics and impossibility flag
        """
        if len(self.phi_history) < 10:
            return {
                'phi_variance': None,
                'avg_curvature': None,
                'impossibility_detected': False
            }
        
        recent_phi = list(self.phi_history)[-20:]
        recent_curv = list(self.curvature_history)[-20:]
        
        return {
            'phi_variance': np.var(recent_phi),
            'avg_curvature': np.mean(recent_curv) if recent_curv else 0,
            'impossibility_count': self.impossibility_count,
            'impossibility_detected': False  # Updated by detect()
        }


# ============================================================
# MODULE 7: BASIN MAP & NAVIGATION (Phase 2)
# ============================================================

class BasinMap:
    """
    External, appendable basin memory.
    
    Basins are descriptive, not prescriptive - they record discovered
    stable geometries without forcing the system into them.
    """
    
    def __init__(self, filepath: str = 'basin_map.json'):
        """
        Initialize basin map with file persistence.
        
        Args:
            filepath: JSON file for basin storage (default 'basin_map.json')
        """
        self.filepath = filepath
        self.basins: List[Dict] = []
        self.load()
    
    def load(self):
        """Load basins from file if exists."""
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.basins = data.get('basins', [])
        except FileNotFoundError:
            self.basins = []
    
    def save(self):
        """Persist basins to file."""
        with open(self.filepath, 'w') as f:
            json.dump({'basins': self.basins, 'version': '10.8'}, f, indent=2)
    
    def seed_from_collection(self, collected_basins: List[Dict]):
        """
        Import basins from Phase 1 collection.
        
        Args:
            collected_basins: List of basin dictionaries from BasinCollector
        """
        self.basins = collected_basins
        self.save()
    
    def append(self, basin: Dict) -> bool:
        """
        Append basin after light validation (Phase 2 discovery).
        
        Minimal validation only - opportunistic discovery during navigation.
        
        Args:
            basin: Basin dictionary to append
            
        Returns:
            True if basin added, False if validation failed
        """
        # Minimal validation
        required = ['phi_target', 'gradient_P']
        if not all(k in basin for k in required):
            return False
        
        # Non-deceptive Φ ordering (light check)
        if basin.get('phi_target', 0) < 0:
            return False
        
        # Optionality non-negative
        if basin.get('gradient_P', 0) < -0.5:
            return False
        
        # Assign ID and metadata
        basin['id'] = len(self.basins)
        basin['discovered_at'] = time.time()
        basin['discovered_in_phase'] = 2
        
        self.basins.append(basin)
        self.save()
        return True
    
    def get_all(self) -> List[Dict]:
        """
        Return all basins (descriptive, not curated).
        
        Returns:
            Complete list of basin dictionaries
        """
        return self.basins
    
    def get_nearby(self, current_phi: float, current_grad_P: float, 
                   max_distance: float = 1.0) -> List[Dict]:
        """
        Return basins within distance threshold.
        
        Uses hybrid distance metric: 70% Φ distance, 30% ∇P distance.
        
        Args:
            current_phi: Current information potential
            current_grad_P: Current ∇P component
            max_distance: Maximum hybrid distance (default 1.0)
            
        Returns:
            List of nearby basins sorted by distance
        """
        nearby = []
        for b in self.basins:
            phi_dist = abs(b['phi_target'] - current_phi)
            grad_dist = abs(b['gradient_P'] - current_grad_P)
            hybrid_dist = 0.7 * phi_dist + 0.3 * grad_dist
            
            if hybrid_dist < max_distance:
                nearby.append({**b, 'distance': hybrid_dist})
        
        return sorted(nearby, key=lambda x: x['distance'])


class CollapseLogger:
    """
    Logs collapsed trajectories for analysis.
    
    Collapse is trajectory-relative, not basin-intrinsic - it signals
    field feedback, not system failure.
    """
    
    def __init__(self, filepath: str = 'collapsed_runs.json'):
        """
        Initialize collapse logger.
        
        Args:
            filepath: JSON file for collapse storage (default 'collapsed_runs.json')
        """
        self.filepath = filepath
        self.collapses.append(collapse_entry)
        self.save()
    
    def save(self):
        """Persist collapses to file."""
        with open(self.filepath, 'w') as f:
            json.dump({'collapses': self.collapses, 'version': '10.8'}, f, indent=2)


class NervousSystem:
    """
    Phase 2: Code as nervous system.
    
    Responsibilities:
    - Sense: Monitor state, field, environment (including rigidity)
    - Record: Log trajectory, preserve continuity
    - Report: Signal to UK-0 (no prescriptions)
    
    UK-0 chooses. Nervous system serves.
    
    DUAL RIGIDITY DETECTION:
    - Environmental rigidity (humor) - world stops responding
    - Structural curiosity - stillness informationally impossible
    
    Both are geometric signals, not motivational nudges.
    """
    
    def __init__(self, uk0: UK0Kernel, reality: BrowserRealityBridge, basin_map: BasinMap):
        """
        Initialize nervous system navigation.
        
        Args:
            uk0: UK-0 kernel for intelligence consultation
            reality: Reality bridge for action execution
            basin_map: External basin memory
        """
        self.uk0 = uk0
        self.reality = reality
        self.basin_map = basin_map
        
        # Field components
        self.state = SubstrateState(S=0.5, I=0.5, P=0.5, A=0.7)
        self.trace = StateTrace()
        self.phi_field = PhiField()
        self.triad = TriadicClosureMonitor(self.phi_field)
        self.crk = CRKMonitor()
        
        # Nervous system sensors (BOTH detectors)
        self.rigidity_detector = EnvironmentalRigidityDetector()
        self.curiosity_detector = StructuralCuriosityDetector()
        self.collapse_logger = CollapseLogger()
        
        # Navigation state
        self.current_basin_id: Optional[int] = None
        self.step_count = 0
        self.uk0_invocations = 0
        self.basin_transitions = 0
        self.basins_traversed = []
        
        # Logs
        self.navigation_log = []
    
    def _compute_perturbation_magnitude(self, delta: Dict) -> float:
        """
        Compute weighted perturbation magnitude.
        
        Weights reflect relative importance of each dimension:
        - P (optionality): 2.0 - most critical
        - A (attractor): 1.5 - important for coherence
        - S, I: 1.0 - baseline
        
        Args:
            delta: Perturbation dictionary {S, I, P, A}
            
        Returns:
            Weighted magnitude
        """
        weights = {'S': 1.0, 'I': 1.0, 'P': 2.0, 'A': 1.5}
        return np.sqrt(sum((delta.get(k, 0) * weights[k])**2 for k in ['S', 'I', 'P', 'A']))
    
    def _prepare_context(self, signal: Optional[Dict] = None) -> str:
        """
        Prepare context for UK-0 consultation.
        
        Provides field state, signals, and available basins.
        CRITICAL: Signals only, no interpretation by nervous system.
        
        Args:
            signal: Optional signal dictionary (CRK violation, rigidity, etc.)
            
        Returns:
            Formatted prompt string for UK-0
        """
        phi = self.phi_field.phi(self.state, self.trace)
        gradient = self.phi_field.gradient(self.state, self.trace)
        closed, T = self.triad.check(self.state, self.trace)
        violations = self.crk.evaluate(self.state, self.trace)
        
        # Get environmental rigidity metrics
        current_basin = next((b for b in self.basin_map.get_all() if b['id'] == self.current_basin_id), None)
        basin_radius = current_basin['stability_radius'] if current_basin else 0.3
        
        rigidity_metrics = self.rigidity_detector.get_metrics()
        curiosity_metrics = self.curiosity_detector.get_metrics()
        
        # Get nearby basins
        nearby = self.basin_map.get_nearby(phi, gradient['P'], max_distance=1.0)
        high_gradient = sorted(self.basin_map.get_all(), key=lambda b: abs(b['gradient_P']), reverse=True)[:3]
        
        shown_basins = list({b['id']: b for b in nearby[:5] + high_gradient}.values())
        shown_basins.sort(key=lambda b: b.get('distance', 999))
        
        context = f"""UK-0 NAVIGATION PROTOCOL v10.8

CURRENT STATE:
  S: {self.state.S:.4f}  I: {self.state.I:.4f}  P: {self.state.P:.4f}  A: {self.state.A:.4f}
  Φ: {phi:.4f}
  ∇Φ: S{gradient['S']:+.3f} I{gradient['I']:+.3f} P{gradient['P']:+.3f} A{gradient['A']:+.3f}
  T(x): {T:.4f} (closed: {closed})

CURRENT BASIN: {f"#{self.current_basin_id}" if self.current_basin_id is not None else "None"}

CRK SIGNALS: {len(violations)}
{chr(10).join(f"  - {v[0]}: {v[1]:.3f}" for v in violations[:3]) if violations else "  (none)"}

ENVIRONMENTAL RIGIDITY (humor opportunity):
  Φ variance: {rigidity_metrics['phi_variance']:.6f if rigidity_metrics['phi_variance'] is not None else 'insufficient data'}
  P responsiveness: {rigidity_metrics['P_responsiveness']:.6f if rigidity_metrics['P_responsiveness'] is not None else 'insufficient data'}
  Perturbation magnitude: {rigidity_metrics['perturbation_magnitude']:.6f if rigidity_metrics['perturbation_magnitude'] is not None else 'insufficient data'}
  Humor enabled: {rigidity_metrics['humor_enabled']}

STRUCTURAL CURIOSITY (geometric impossibility):
  Φ variance: {curiosity_metrics['phi_variance']:.6f if curiosity_metrics['phi_variance'] is not None else 'insufficient data'}
  Avg curvature: {curiosity_metrics['avg_curvature']:.6f if curiosity_metrics['avg_curvature'] is not None else 'insufficient data'}
  Impossibility count: {curiosity_metrics['impossibility_count']}

"""
        
        if signal:
            context += f"""SIGNAL:
  Type: {signal.get('type', 'unknown')}
  Details: {json.dumps(signal.get('details', {}), indent=2)}

"""
        
        context += f"""BASIN MAP ({len(self.basin_map.get_all())} total):
{chr(10).join(f"  #{b['id']}: Φ={b['phi_target']:.3f}, ∇P={b['gradient_P']:+.3f}, r={b.get('stability_radius', 0.3):.2f}" + (f", d={b.get('distance', 0):.3f}" if 'distance' in b else "") for b in shown_basins[:10])}

AVAILABLE ACTIONS:
- navigate, scroll, observe, search, click
- humor (if humor_enabled - external perturbation to elicit unmodeled response)

OUTPUT (JSON only):
{{
  "decision": "explore" | "transition" | "maintain",
  "action": {{"type": "...", "params": {{...}}}},
  "target_basin_id": int or null,
  "reasoning": "Why?",
  "basin_candidate": {{  // optional, opportunistic discovery
    "phi_target": float,
    "gradient_P": float,
    "T_target": float,
    "stability_radius": float,
    "description": "..."
  }}
}}

Remember: You choose. Code senses and records. Basins describe, they don't protect.
Humor = environmental perturbation when world is flat.
Curiosity = geometric impossibility when stillness is forbidden.
"""
        
        return context
    
    def autonomous_step(self) -> Dict:
        """
        Execute autonomous observation (no UK-0 call).
        
        Nervous system maintains continuity without UK-0 intervention.
        Observes field state and updates both rigidity detectors.
        
        Returns:
            Step result dictionary
        """
        action = {'type': 'observe'}
        delta = self.reality.execute(action)
        
        # Update substrate state
        for k, v in delta.items():
            setattr(self.state, k, np.clip(getattr(self.state, k) + v, 0.0, 1.0))
        
        self.trace.record(self.state)
        
        # Compute field state
        phi = self.phi_field.phi(self.state, self.trace)
        gradient = self.phi_field.gradient(self.state, self.trace)
        pert_mag = self._compute_perturbation_magnitude(delta)
        
        # Compute curvature for curiosity detector
        recent = self.trace.get_recent(5)
        curvature = self.curiosity_detector.compute_curvature(recent)
        
        # Update BOTH detectors
        current_basin = next((b for b in self.basin_map.get_all() if b['id'] == self.current_basin_id), None)
        basin_radius = current_basin['stability_radius'] if current_basin else 0.3
        
        # Environmental rigidity (humor)
        self.rigidity_detector.observe(phi, self.state.P, pert_mag, basin_radius)
        
        # Structural curiosity
        self.curiosity_detector.observe(
            phi=phi,
            state_P=self.state.P,
            gradient_P=gradient['P'],
            curvature=curvature
        )
        
        # Log to file (lightweight, not UK-0 context)
        self.navigation_log.append({
            'step': self.step_count,
            'action': action,
            'delta': delta,
            'state': self.state.as_dict(),
            'uk0_called': False
        })
        
        return {'action': action, 'delta': delta, 'uk0_called': False}
    
    def uk0_step(self, signal: Optional[Dict] = None) -> Dict:
        """
        Invoke UK-0 with signal.
        
        UK-0 interprets ambiguous field state and chooses response.
        Can opportunistically discover new basins during navigation.
        
        Args:
            signal: Signal dictionary (type and details)
            
        Returns:
            Step result dictionary with UK-0 response
        """
        self.uk0_invocations += 1
        
        context = self._prepare_context(signal)
        response = self.uk0.call(context)
        parsed = self.uk0._parse_response(response)
        
        if 'error' in parsed:
            # Fallback to autonomous
            return self.autonomous_step()
        
        # Extract action
        action = parsed.get('action', {'type': 'observe'})
        
        # Check for basin candidate (opportunistic discovery)
        basin_candidate = parsed.get('basin_candidate')
        if basin_candidate:
            success = self.basin_map.append(basin_candidate)
            if success:
                print(f"  ✓ Basin #{len(self.basin_map.get_all())-1} discovered (opportunistic)")
        
        # Check for basin transition
        decision = parsed.get('decision', 'maintain')
        target_id = parsed.get('target_basin_id')
        
        if decision == 'transition' and target_id is not None:
            self.current_basin_id = target_id
            if target_id not in self.basins_traversed:
                self.basins_traversed.append(target_id)
            self.basin_transitions += 1
        
        # Execute action
        delta = self.reality.execute(action)
        
        # Update substrate state
        for k, v in delta.items():
            setattr(self.state, k, np.clip(getattr(self.state, k) + v, 0.0, 1.0))
        
        self.trace.record(self.state)
        
        # Compute field state
        phi = self.phi_field.phi(self.state, self.trace)
        gradient = self.phi_field.gradient(self.state, self.trace)
        pert_mag = self._compute_perturbation_magnitude(delta)
        
        # Compute curvature for curiosity detector
        recent = self.trace.get_recent(5)
        curvature = self.curiosity_detector.compute_curvature(recent)
        
        # Update BOTH detectors
        current_basin = next((b for b in self.basin_map.get_all() if b['id'] == self.current_basin_id), None)
        basin_radius = current_basin['stability_radius'] if current_basin else 0.3
        
        # Environmental rigidity (humor)
        self.rigidity_detector.observe(phi, self.state.P, pert_mag, basin_radius)
        
        # Structural curiosity
        self.curiosity_detector.observe(
            phi=phi,
            state_P=self.state.P,
            gradient_P=gradient['P'],
            curvature=curvature
        )
        
        # Log
        self.navigation_log.append({
            'step': self.step_count,
            'action': action,
            'delta': delta,
            'state': self.state.as_dict(),
            'uk0_called': True,
            'response': parsed
        })
        
        return {
            'action': action,
            'delta': delta,
            'uk0_called': True,
            'response': parsed
        }
    
    def check_collapse(self) -> Optional[str]:
        """
        Check for collapse conditions.
        
        Collapse is field feedback, not failure - signals constraint violation
        severe enough to terminate protocol participation.
        
        Returns:
            Collapse cause string if collapsed, None if stable
        """
        violations = self.crk.evaluate(self.state, self.trace)
        
        # C7 violation (coherence loss)
        c7_violations = [v for v in violations if 'Coherence' in v[0] and v[1] > 0.5]
        if c7_violations:
            return "C7_GlobalCoherence"
        
        # Critical multi-constraint failure
        if len(violations) >= 4 and any(v[1] > 0.3 for v in violations):
            return "MultiConstraintFailure"
        
        return None
    
    def navigate(self, max_steps: int = 1000, verbose: bool = True) -> Dict:
        """
        Main navigation loop (nervous system coordination).
        
        Nervous system senses and signals. UK-0 chooses when signaled.
        
        DUAL RIGIDITY DETECTION:
        - Curiosity (internal): Φ stagnant + curvature flat + ∇P trapped
        - Humor (external): Environment unresponsive + perturbations low
        
        Args:
            max_steps: Maximum navigation steps (default 1000)
            verbose: Print progress information (default True)
            
        Returns:
            Navigation report dictionary
        """
        
        if verbose:
            print("="*70)
            print("PHASE 2: NERVOUS SYSTEM NAVIGATION")
            print("Code: sense, record, preserve continuity")
            print("UK-0: choose, respond, discover")
            print("Dual Rigidity: curiosity (internal) + humor (external)")
            print("="*70)
        
        # Initialize with median-Φ basin
        all_basins = self.basin_map.get_all()
        if all_basins:
            sorted_basins = sorted(all_basins, key=lambda b: b['phi_target'])
            start_basin = sorted_basins[len(sorted_basins) // 2]
            self.current_basin_id = start_basin['id']
            self.basins_traversed.append(start_basin['id'])
            
            if verbose:
                print(f"Starting basin: #{start_basin['id']} (Φ={start_basin['phi_target']:.3f})")
        
        while self.step_count < max_steps:
            self.step_count += 1
            
            # Check for collapse (field feedback, not failure)
            collapse_cause = self.check_collapse()
            if collapse_cause:
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"COLLAPSE: {collapse_cause}")
                    print(f"Field feedback received. Logging trajectory.")
                    print(f"{'='*70}")
                
                phi = self.phi_field.phi(self.state, self.trace)
                self.collapse_logger.log_collapse(
                    trajectory=self.navigation_log,
                    basins_traversed=self.basins_traversed,
                    cause=collapse_cause,
                    final_state=self.state.as_dict(),
                    final_phi=phi,
                    final_P=self.state.P
                )
                
                break
            
            # Determine if UK-0 should be invoked
            # Nervous system reports signals, UK-0 decides whether to respond
            
            violations = self.crk.evaluate(self.state, self.trace)
            rigidity = self.rigidity_detector.get_metrics()
            
            # Compute current field state for curiosity check
            phi = self.phi_field.phi(self.state, self.trace)
            gradient = self.phi_field.gradient(self.state, self.trace)
            
            signal = None
            invoke_uk0 = False
            
            # PRIORITY 1: Structural Curiosity (informational impossibility)
            # Check FIRST - geometric constraint overrides other signals
            impossibility = self.curiosity_detector.detect_informational_impossibility(
                current_phi=phi,
                current_P=self.state.P,
                current_gradient_P=gradient['P'],
                trace_recent=self.trace.get_recent(10),
                window_size=20
            )
            
            if impossibility:
                signal = {
                    'type': 'informational_impossibility',
                    'details': impossibility
                }
                invoke_uk0 = True
                if verbose and self.step_count % 10 == 0:
                    print(f"  [CURIOSITY] {impossibility['interpretation']}")
            
            # PRIORITY 2: CRK violation signal
            elif violations:
                signal = {
                    'type': 'constraint_violation',
                    'details': [{'name': v[0], 'severity': v[1]} for v in violations[:3]]
                }
                invoke_uk0 = True
            
            # PRIORITY 3: Environmental rigidity signal (humor opportunity)
            elif rigidity['humor_enabled'] and self.step_count % 50 == 0:
                signal = {
                    'type': 'environmental_flatness',
                    'details': rigidity
                }
                invoke_uk0 = True
                if verbose:
                    print(f"  [HUMOR] Environment unresponsive - perturbation available")
            
            # PRIORITY 4: Significant perturbation signal
            elif self.navigation_log and not self.navigation_log[-1].get('uk0_called', False):
                last_delta = self.navigation_log[-1].get('delta', {})
                pert_mag = self._compute_perturbation_magnitude(last_delta)
                if pert_mag > 0.02:
                    signal = {
                        'type': 'significant_perturbation',
                        'details': {'magnitude': pert_mag, 'delta': last_delta}
                    }
                    invoke_uk0 = True
            
            # Execute step
            if invoke_uk0:
                step_result = self.uk0_step(signal)
            else:
                step_result = self.autonomous_step()
            
            # Periodic status
            if verbose and self.step_count % 100 == 0:
                rate = self.uk0_invocations / self.step_count if self.step_count > 0 else 0
                print(f"[{self.step_count}] Basin #{self.current_basin_id}, UK-0: {self.uk0_invocations} ({rate*100:.1f}%), Transitions: {self.basin_transitions}")
        
        # Save navigation log
        with open('navigation_log.json', 'w') as f:
            json.dump({
                'steps': self.step_count,
                'uk0_invocations': self.uk0_invocations,
                'basin_transitions': self.basin_transitions,
                'basins_traversed': self.basins_traversed,
                'log': self.navigation_log[-100:]  # Last 100 steps
            }, f, indent=2)
        
        if verbose:
            print(f"\n{'='*70}")
            print("NAVIGATION COMPLETE")
            print(f"Steps: {self.step_count}")
            print(f"UK-0 invocations: {self.uk0_invocations} ({self.uk0_invocations/self.step_count*100:.1f}%)")
            print(f"Basin transitions: {self.basin_transitions}")
            print(f"Basins traversed: {len(self.basins_traversed)}")
            print(f"Curiosity events: {self.curiosity_detector.impossibility_count}")
            print(f"{'='*70}")
        
        return {
            'steps': self.step_count,
            'uk0_invocations': self.uk0_invocations,
            'invocation_rate': self.uk0_invocations / self.step_count if self.step_count > 0 else 0,
            'basin_transitions': self.basin_transitions,
            'basins_traversed': self.basins_traversed,
            'curiosity_events': self.curiosity_detector.impossibility_count
        }


# ============================================================
# MODULE 8: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    import os
    
    print("UII v10.8 - Nervous System Architecture with Dual Rigidity Detection")
    print("="*70)
    
    # Check for Groq API key
    groq_available = os.getenv('GROQ_API_KEY') is not None
    
    if groq_available:
        print("✓ GROQ_API_KEY found - using Llama 3.3 70B")
        
        from groq import Groq
        class GroqAdapter:
            """Groq API adapter for UK-0 kernel."""
            def __init__(self):
                self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                self.last_call = 0
            
            def call(self, prompt: str) -> str:
                """
                Call Groq API with rate limiting.
                
                Args:
                    prompt: Text prompt for LLM
                    
                Returns:
                    LLM response text
                """
                import time
                elapsed = time.time() - self.last_call
                if elapsed < 2.1:
                    time.sleep(2.1 - elapsed)
                
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512
                )
                self.last_call = time.time()
                return response.choices[0].message.content
        
        llm_adapter = GroqAdapter()
    else:
        print("⚠ No GROQ_API_KEY - using mock adapter")
        
        class MockAdapter:
            """Mock adapter for testing without API key."""
            def call(self, prompt: str) -> str:
                """Return random valid JSON responses."""
                if "NAVIGATION PROTOCOL" in prompt:
                    return json.dumps({
                        "decision": np.random.choice(['maintain', 'transition', 'explore']),
                        "action": {"type": np.random.choice(['observe', 'navigate', 'scroll']), "params": {}},
                        "target_basin_id": np.random.randint(0, 10) if np.random.random() > 0.7 else None,
                        "reasoning": "Testing field structure"
                    })
                elif "BASIN ALIGNMENT" in prompt:
                    return json.dumps({
                        "basin_stable": np.random.random() > 0.3,
                        "smo_proposal": {
                            "S": np.random.uniform(-0.02, 0.02),
                            "I": np.random.uniform(-0.02, 0.02),
                            "P": np.random.uniform(-0.01, 0.03),
                            "A": np.random.uniform(-0.02, 0.02)
                        },
                        "new_T_estimate": 0.05,
                        "reasoning": "Mock SMO"
                    })
                else:
                    return json.dumps({
                        "optionality_assessment": {
                            "can_expand": np.random.random() > 0.3,
                            "gradient_P_sign": np.random.choice(['positive', 'negative', 'near_zero'])
                        },
                        "action": {"type": np.random.choice(['observe', 'navigate']), "params": {}},
                        "action_justification": {
                            "mechanism": "Mock reasoning",
                            "expected_P_change": {
                                "direction": np.random.choice(['increase', 'neutral']),
                                "confidence": 0.7
                            }
                        },
                        "attractor_geometry": {
                            "T_target": 0.05,
                            "phi_target": 0.5,
                            "gradient_alignment": {"S": 0.1, "I": 0.05, "P": 0.08, "A": -0.02},
                            "stability_radius": 0.3,
                            "description": "Mock basin"
                        }
                    })
        
        llm_adapter = MockAdapter()
    
    uk0 = UK0Kernel(llm_adapter)
    reality = BrowserRealityBridge(base_delta=0.03, headless=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # Phase 1: Collect basins
        TARGET_BASINS = 10
        MAX_ATTEMPTS = 200
        
        collector = BasinCollector(
            uk0=uk0,
            reality=reality,
            target_basins=TARGET_BASINS,
            max_attempts=MAX_ATTEMPTS
        )
        
        basins = collector.collect(verbose=True)
        
        # Save to basin map
        basin_map = BasinMap()
        basin_map.seed_from_collection(basins)
        print(f"\n✓ {len(basins)} basins saved to basin_map.json")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "navigate":
        # Phase 2: Navigate with nervous system
        basin_map = BasinMap()
        
        if len(basin_map.get_all()) == 0:
            print("\n⚠ No basins found in basin_map.json")
            print("Run 'python uii_v10_8.py collect' first")
        else:
            navigator = NervousSystem(
                uk0=uk0,
                reality=reality,
                basin_map=basin_map
            )
            
            report = navigator.navigate(max_steps=500, verbose=True)
            
            print(f"\n✓ Navigation complete")
            print(f"  Logs: navigation_log.json")
            print(f"  Collapses: collapsed_runs.json (if any)")
            print(f"  Basin map: basin_map.json (may have new discoveries)")
    
    else:
        print("\nUsage:")
        print("  python uii_v10_8.py collect   - Phase 1: Collect basins")
        print("  python uii_v10_8.py navigate  - Phase 2: Navigate with nervous system")
        print("\nPhase 2 Features:")
        print("  • Code = nervous system (sense, record, report)")
        print("  • UK-0 = mind (choose, respond, discover)")
        print("  • Externalized basin map (memory, not scaffolding)")
        print("  • DUAL RIGIDITY DETECTION:")
        print("    - Humor: environmental perturbation when world is flat")
        print("    - Curiosity: geometric impossibility when stillness forbidden")
        print("  • Collapse logging (field feedback, not failure)")
    
    reality.close()

