"""
UII v14.1 — uii_reality.py
Perturbation Harnessing

Role: Reality is the authoritative, non-optimizing source of perturbations.
It executes actions and returns measured deltas — nothing more.
The browser is a low-fidelity viewport into a slice of reality.

Also contains:
  - AttractorMonitor (detects stability windows for mitosis)
  - CouplingMatrixEstimator (empirical S/I/P/A co-movement — learns from reality)

Contents:
  - AttractorMonitor
  - CouplingMatrixEstimator
  - BrowserRealityAdapter (Playwright-based reality interface)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import copy
import time
import json
from collections import deque

from uii_types import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace,
    RealityAdapter,
    AgentHandler, AVAILABLE_AGENTS,
)

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

        if len(self.recent_phi) < self.stability_window:
            return (False, "accumulating_stability_data")

        phi_stable = True
        for i in range(1, len(self.recent_phi)):
            delta_phi = abs(self.recent_phi[i] - self.recent_phi[i-1])
            if delta_phi > self.phi_epsilon:
                phi_stable = False
                break

        constraints_satisfied = len(crk_violations) == 0

        if phi_stable and constraints_satisfied:
            if not self.freeze_verified:
                self.freeze_verified = True
                self.freeze_step = step_count
                return (True, f"freeze_verified_step_{step_count}")
            return (True, "freeze_verified")

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


# ============================================================
# COUPLING MATRIX ESTIMATOR
# ============================================================

class CouplingMatrixEstimator:
    """
    Empirical S/I/P/A co-movement tracker.

    Tracks how the four substrate dimensions move together across reality interactions.
    Updates via slow exponential moving average (alpha=0.05 — long memory, resists noise).

    This is the most important Layer 2 component.
    A 4x4 matrix of empirical coupling strengths IS the basin's causal signature
    in compact heritable form.

    A bootstrapping Triad with an inherited coupling matrix starts with calibrated
    gradients — it knows how its lineage's SIPA dimensions actually couple,
    without being told explicitly.

    matrix[i][j] = how much dim_i tends to move when dim_j moves.
    Starts as identity (no assumed couplings).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.matrix = np.eye(4)  # Start as identity — no assumed couplings
        self.dims = SUBSTRATE_DIMS
        self.observation_count = 0
        self.calibration_threshold = 200  # Observations needed for full confidence

    def observe(self, observed_delta: Dict[str, float]):
        """Update coupling matrix from observed substrate co-movement."""
        for i, dim_i in enumerate(self.dims):
            for j, dim_j in enumerate(self.dims):
                if i == j:
                    continue
                di = observed_delta.get(dim_i, 0.0)
                dj = observed_delta.get(dim_j, 0.0)
                if abs(dj) > 1e-6:
                    observed_coupling = np.clip(di / dj, -2.0, 2.0)
                    self.matrix[i][j] = (
                        (1 - self.alpha) * self.matrix[i][j] +
                        self.alpha * observed_coupling
                    )
        self.observation_count += 1

    def get_confidence(self) -> float:
        return min(1.0, self.observation_count / self.calibration_threshold)

    def to_genome_entry(self) -> Dict:
        return {
            'matrix': self.matrix.tolist(),
            'observations': self.observation_count,
            'confidence': self.get_confidence()
        }

    @classmethod
    def from_genome_entry(cls, entry: Dict, alpha: float = 0.05) -> 'CouplingMatrixEstimator':
        estimator = cls(alpha=alpha)
        estimator.matrix = np.array(entry['matrix'])
        estimator.observation_count = entry.get('observations', 0)
        return estimator

    @classmethod
    def merge(cls, parent_entry: Dict, session_estimator: 'CouplingMatrixEstimator') -> 'CouplingMatrixEstimator':
        """
        Merge parent coupling matrix with session observations.
        Session weighted by its own confidence — low-evidence sessions
        don't overwrite high-confidence inherited structure.
        """
        merged = cls()
        parent_matrix = np.array(parent_entry.get('matrix', np.eye(4).tolist()))
        session_conf = session_estimator.get_confidence()
        merged.matrix = (1 - session_conf) * parent_matrix + session_conf * session_estimator.matrix
        merged.observation_count = (
            parent_entry.get('observations', 0) + session_estimator.observation_count
        )
        return merged


# ============================================================
# NEW: ResidualTracker


class BrowserRealityAdapter(RealityAdapter):
    """
    Browser-based reality interface via Playwright.

    v14: response_latency_ms added to context for ResidualTracker.
    Python affordance ungated (v13.4+). freeze_verified does not gate execution.
    """

    def __init__(self, base_delta: float = 0.03, headless: bool = True):
        self.base_delta = base_delta
        self.headless = headless

        self.previous_dom_metrics: Optional[Dict] = None
        self.initialized: bool = False
        self._ever_navigated: bool = False

        self.volatility_history: deque = deque(maxlen=10)
        # self.latency_history: deque = deque(maxlen=10) removed 
        # complexity_history removed v14.2: was tracking element_count variance,
        # which is an INTERFACE_COUPLED_SIGNAL and drove I structurally negative.

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

    def _compute_substrate_delta(self, before: Dict, after: Dict,
                                  state: 'SubstrateState' = None) -> Dict[str, float]:
        """
        Compute substrate delta from environmental measurement.

        Returns S and P deltas only. I is computed by MentatTriad._compute_delta_i()
        from the Triad's trace — compression quality of S history is a Triad-level
        observation, not a browser-level measurement.

        DASS causal chain:
            S: E × I → S'   Environmental surface change, gated by current I.
                             Current I is in state — passed in, not computed here.
            P: C → C'       Environmental volatility with soft I-support term.
            I: 0.0          Computed in MentatTriad.step() via _compute_delta_i().
            A: 0.0          Computed in MentatTriad._compute_a().

        No SMO signals (rigidity, prediction_error) enter this method.
        U is downstream of S, I, P. The causal chain does not run backward.
        """
        if state is None:
            from uii_types import SubstrateState as _SS
            state = _SS(S=0.5, I=0.5, P=0.5, A=0.7)

        delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

        # ── S: Environmental surface change, gated by current I ─────────────
        #
        # Spec: S: E × I → S'
        # E = change in interactive surface fraction (normalized, not raw counts)
        # I = current integration quality — already in state, passed in
        #
        # i_gate: at I=0 → 50% of signal admitted. At I=1 → 100%.
        # Low I reflects that less compressed structure is available to make
        # sense of new sensing. Structural, not punitive.
        current_surface = after['interactive_count'] / max(after['element_count'], 1)
        prev_surface    = before['interactive_count'] / max(before['element_count'], 1)
        surface_delta   = current_surface - prev_surface

        viewport_coverage = min(1.0, after['viewport_height'] / max(after['scroll_height'], 1))
        prev_viewport     = min(1.0, before['viewport_height'] / max(before['scroll_height'], 1))
        coverage_delta    = viewport_coverage - prev_viewport

        env_signal = 0.7 * surface_delta + 0.3 * coverage_delta
        i_gate     = 0.5 + 0.5 * state.I   # [0.5, 1.0]

        delta['S'] = float(np.clip(env_signal * i_gate, -0.1, 0.1))
        # Noise on S only — environmental sensing has measurement noise.
        # Noise on I would be incoherent: I is a derived compression measure.
        delta['S'] += float(np.random.uniform(-0.005, 0.005))

        # ── I: 0.0 — computed in MentatTriad._compute_delta_i() ─────────────
        # I: {S_i} → C   Compression quality of S history.
        # The browser does not have S history. The Triad does.
        # delta['I'] stays 0.0 — merged by step() before apply_delta().

        # ── P: Environmental volatility with soft I-support ─────────────────
        #
        # Spec: P: C → C'   Compressed state → forward model.
        # C is produced by I. When I is very low, less compressed structure
        # is available to project forward from. The I-support term is a soft
        # signal [-0.02, +0.02] — not a ceiling (the PhiField grounding
        # invariant provides the hard ceiling on P separately).
        structural_delta = abs(after['element_count'] - before['element_count']) / max(before['element_count'], 1)
        text_delta       = abs(after['text_length']   - before['text_length'])   / max(before['text_length'], 1)
        volatility       = float(np.mean([structural_delta, text_delta]))
        self.volatility_history.append(volatility)

        if len(self.volatility_history) >= 5:
            volatility_variance = float(np.var(list(self.volatility_history)[-5:]))
            env_p = float(np.clip(0.1 - volatility_variance * 10.0, -0.1, 0.1))
        else:
            env_p = float(np.clip(-volatility * 0.5, -0.1, 0.1))

        # Soft I-support: below I=0.5, less compressed structure for P to use.
        i_support      = float(np.clip(state.I - 0.5, -0.5, 0.5))
        i_contribution = 0.04 * i_support   # [-0.02, +0.02]
        delta['P']     = float(np.clip(env_p + i_contribution, -0.1, 0.1))

        # URL change: forward model loses its reference frame.
        if after.get('url', '') != before.get('url', ''):
            delta['P'] -= 0.08

        # ── A: 0.0 — computed in MentatTriad._compute_a() ───────────────────
        # Requires Triad context (genome geometry, trace). Not Reality's job.

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

    def execute(self, action: Dict, boundary_pressure: float = 0.0,
                state: 'SubstrateState' = None,
                coupling_confidence: float = 0.0) -> Tuple[Dict[str, float], Dict]:
        """
        Execute action in Reality and return MEASURED perturbation delta.

        v13.4: Python affordance ungated.
        v14: response_latency_ms added to context for ResidualTracker.
        """
        t_start = time.time()  # v14: for response_latency_ms signal

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

            elif action_type == 'query_agent':
                return self._query_agent(params, before_metrics)

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
                        'response_latency_ms': (time.time() - t_start) * 1000,
                    }
                )

            action_succeeded = False

        response_latency_ms = (time.time() - t_start) * 1000
        after_metrics = self._measure_dom_state()
        delta = self._compute_substrate_delta(
         before_metrics, after_metrics,
         state=state,
     )

        if boundary_pressure > 0.0:
         pressure_damping = (1.0 - 0.7 * boundary_pressure)
         delta['S'] *= pressure_damping
         delta['S'] += np.random.uniform(
             -0.01 * boundary_pressure,
              0.01 * boundary_pressure
         )

        context = {
            'before': before_metrics,
            'after': after_metrics,
            'action_succeeded': action_succeeded,
            'refusal': False,
            'boundary_pressure': boundary_pressure,
            'url_changed': before_metrics['url'] != after_metrics['url'],
            'new_url': after_metrics['url'],
            'page_title': after_metrics['title'],
            'response_latency_ms': (time.time() - t_start) * 1000,  # v14
        }

        self.previous_dom_metrics = after_metrics

        return delta, context

    def _query_agent(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        """Query an agent (non-blocking)."""
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

        agent = AVAILABLE_AGENTS[agent_name]
        triad_id = params.get('triad_id', 'default')
        agent.post_query(triad_id, query_text)

        return (
            {'S': 0.01, 'I': 0, 'P': 0, 'A': 0.01},
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
        """Execute arbitrary Python code. v13.4: Available from step 1 (ungated)."""
        import os

        code = params.get('code')
        if not code:
            raise ValueError("python affordance requires 'code' parameter")

        cwd = os.getcwd()

        exec_globals = {
            '__builtins__': __builtins__,
            'cwd': cwd,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
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
