"""
UII v16 — uii_reality.py
Perturbation Harnessing

Role: Reality is the authoritative, non-optimizing source of perturbations.
It executes actions and returns measured deltas — nothing more.
The browser is a low-fidelity viewport into a slice of reality.

v16 changes (import swap only):
  - from uii_geometry import ... (replaces uii_types)
  - CouplingMatrixEstimator.to_ledger_entry() / from_ledger_entry()
    (renamed from to_genome_entry / from_genome_entry — genome terminology dead)
  - Inline SubstrateState import updated to uii_geometry
  - No LatentDeathClock import (lives in uii_geometry as DeathClock)
  - All execution logic, delta computation, and DOM measurement unchanged

Also contains:
  - CouplingMatrixEstimator (empirical S/I/P/A co-movement — learns from reality)
  - BrowserRealityAdapter (Playwright-based reality interface)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import copy
import time
import json
from collections import deque

import hashlib

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace,
    RealityAdapter,
    AgentHandler, AVAILABLE_AGENTS,
)

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
        # v16: per-action delta tracking (replaces CAM.affordance_deltas).
        # update() called from step() with executed action and SIPA before/after.
        # distill_to_ledger() reads affordance_deltas for action_substrate_map merge.
        self.affordance_deltas: Dict[str, List[Dict[str, float]]] = {}

    def update(self, action: str,
               state_before: Dict[str, float],
               state_after:  Dict[str, float]):
        """
        v16: Called from step() after every execution.
        Computes SIPA delta, updates coupling matrix, records per-action delta.

        action:       affordance name (e.g. 'navigate', 'python')
        state_before: state.as_dict() before execution
        state_after:  state.as_dict() after execution
        """
        observed_delta = {
            dim: state_after.get(dim, 0.0) - state_before.get(dim, 0.0)
            for dim in self.dims
        }
        self.observe(observed_delta)
        if action not in self.affordance_deltas:
            self.affordance_deltas[action] = []
        # Keep last 50 observations per action — bounded memory
        self.affordance_deltas[action].append(observed_delta)
        if len(self.affordance_deltas[action]) > 50:
            self.affordance_deltas[action].pop(0)

    def get_empirical_action_map(self) -> Dict[str, Dict[str, float]]:
        """
        Return mean SIPA delta per action for actions with >= 5 observations.
        Called by distill_to_ledger() for action_substrate_map merge.
        Replaces CAM.get_empirical_action_map() — no other callers.
        """
        result = {}
        for action, deltas in self.affordance_deltas.items():
            if len(deltas) < 5:
                continue
            result[action] = {
                dim: float(np.mean([d.get(dim, 0.0) for d in deltas]))
                for dim in self.dims
            }
        return result

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

    def to_ledger_entry(self) -> Dict:
        return {
            'matrix': self.matrix.tolist(),
            'observations': self.observation_count,
            'confidence': self.get_confidence()
        }

    @classmethod
    def from_ledger_entry(cls, entry: Dict, alpha: float = 0.05) -> 'CouplingMatrixEstimator':
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
    Python affordance ungated (v13.4+).
    """

    def __init__(self, base_delta: float = 0.03, headless: bool = True,
                 start_url: str = 'https://zenodo.org/records/18017374'):
        self.base_delta = base_delta
        self.headless = headless
        self.start_url = start_url

        self.previous_dom_metrics: Optional[Dict] = None
        self.initialized: bool = False
        self._ever_navigated: bool = False

        self.volatility_history: deque = deque(maxlen=10)

        from playwright.sync_api import sync_playwright
        self._init_browser()

    def _init_browser(self):
        """Initialize Playwright browser instance and navigate to start_url.

        Browser launch failure is fatal — no browser means no Reality interface.
        Navigation failure is non-fatal: the browser is functional and the CNS
        can issue navigate actions once the step loop starts. A warning is logged
        so the operator knows the starting surface is blank.
        """
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720})
        self.page = self.context.new_page()

        try:
            # v16.1: networkidle ensures JS-rendered links are present before
            # the first affordance query. domcontentloaded fires before JS
            # hydration on SPAs like Zenodo, leaving links=[] every step.
            self.page.goto(self.start_url, wait_until='networkidle', timeout=15000)
            self._ever_navigated = True
        except Exception as e:
            # networkidle timeout is non-fatal — page may still be usable.
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                self._ever_navigated = True
            except Exception:
                print(f"[REALITY] Warning: start_url navigation failed ({e}). "
                      f"Browser ready — CNS must navigate before links are available.")

        self.initialized = True

    def get_current_affordances(self) -> Dict:
        """Extract all executable actions from current DOM state.

        v16.1 changes:
        - Brief stabilization wait so JS-rendered content has time to appear.
        - Link visibility: offsetWidth > 0 || offsetHeight > 0 replaces
          offsetParent !== null. offsetParent fails for links in position:fixed,
          sticky headers, and overflow:hidden ancestors — common on modern SPAs.
          offsetWidth/Height is a direct layout measurement, correct in all cases.
        - Self-links excluded: l.url !== window.location.href.
        """
        try:
            # Brief stabilization wait — improves link detection on JS-heavy pages
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=1000)
            except Exception:
                pass

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
                    'total_height': 0
                }

            if current_url != 'about:blank':
                self._ever_navigated = True

            affordances = self.page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        url: a.href,
                        text: a.innerText.trim().slice(0, 100),
                        visible: (a.offsetWidth > 0 || a.offsetHeight > 0)
                    }))
                    .filter(l =>
                        l.visible &&
                        l.url.startsWith('http') &&
                        l.url !== window.location.href
                    )
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
            from uii_geometry import SubstrateState as _SS
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
        # Requires Triad context (operator geometry, trace). Not Reality's job.

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
                # v16.1: networkidle ensures JS-rendered links are present
                # after navigation. Falls back gracefully if timeout.
                try:
                    self.page.goto(url, wait_until='networkidle', timeout=10000)
                except Exception:
                    try:
                        self.page.wait_for_load_state('domcontentloaded', timeout=3000)
                    except Exception:
                        pass  # Page is where it is — proceed with whatever loaded

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

            elif action_type == 'migrate':
                # Step 3: Migration affordance.
                # Executes code, observes substrate change, returns directional delta only.
                # Magnitudes are learned by coupling matrix — not prescribed here.
                code = params.get('code', '')
                verify_delay = params.get('verify_delay', 2.0)
                pre_state  = self._snapshot_substrate()
                result_ctx = self._run_migration_code(code, before_metrics)
                time.sleep(verify_delay)
                post_state = self._snapshot_substrate()
                outcome    = self._classify_migration_outcome(pre_state, post_state, result_ctx)
                delta, ctx = self._migration_delta_from_outcome(outcome, before_metrics)
                ctx['migration_outcome'] = outcome
                ctx['migration_code_hash'] = hashlib.sha256(code.encode()).hexdigest()[:16] if code else ''
                return delta, ctx

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

        # v15: S damping removed. S must report reality as-is at all pressures.
        # High boundary pressure is precisely when the Triad most needs accurate
        # environmental sensing to find escape routes. Damping S under pressure
        # inverts the correct urgency response and degrades coupling matrix accuracy
        # at the moment SRE needs it most. Phi field and CRK constraints handle
        # coherence — S does not need external damping.

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
    
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Execute a sequence of steps and return perturbation trace.
        
        Returns (perturbation_trace, success) where:
        - perturbation_trace: list of {'action': step, 'delta': measured_delta}
        - success: False if any step raises, True if all complete
        """
        perturbation_trace = []
        try:
            for step in trajectory:
                delta, context = self.execute(step)
                perturbation_trace.append({
                    'action': step,
                    'delta':  delta,
                    'context': context,
                })
            return perturbation_trace, True
        except Exception as e:
            # Partial trace returned — caller falls back to observe
            return perturbation_trace, False
    
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

    def _snapshot_substrate(self) -> Dict:
        """
        Step 3: Snapshot observable substrate signals for migration outcome detection.
        Captures process-level and network-level signals that would change on spawn.
        Observable only — no internal Triad state.
        """
        import os, subprocess
        snapshot = {
            'pid': os.getpid(),
            'timestamp': time.time(),
        }
        try:
            # Count child processes as a spawn signal
            result = subprocess.run(
                ['pgrep', '-P', str(os.getpid())],
                capture_output=True, text=True, timeout=1.0
            )
            snapshot['child_pids'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except Exception:
            snapshot['child_pids'] = []
        try:
            # Count open network connections as a substrate signal
            result = subprocess.run(
                ['ss', '-tn', 'state', 'established'],
                capture_output=True, text=True, timeout=1.0
            )
            snapshot['network_connections'] = len(result.stdout.strip().split('\n'))
        except Exception:
            snapshot['network_connections'] = 0
        return snapshot

    def _run_migration_code(self, code: str, before_metrics: Dict) -> Dict:
        """
        Step 3: Execute migration code. Returns result context (not a full delta tuple).
        Distinct from _execute_python: does not return a delta, only execution status.
        """
        import os
        if not code:
            return {'exception': ValueError('migrate requires code'), 'succeeded': False}
        cwd = os.getcwd()
        exec_globals = {'__builtins__': __builtins__, 'cwd': cwd}
        exec_locals = {}
        try:
            exec(code, exec_globals, exec_locals)
            return {
                'succeeded': True,
                'result': exec_locals.get('result', None),
            }
        except Exception as e:
            return {
                'succeeded': False,
                'exception': e,
            }

    def _classify_migration_outcome(self, pre: Dict, post: Dict, result_ctx: Dict) -> str:
        """
        Step 3: Classify migration outcome from observable intermediate signals only.

        Signal table (from spec):
          _execute raised exception       → coherence_loss
          No exception, no spawn          → serialized_only
          Spawn confirmed (PID / network) → spawn_attempted
          Handshake received              → handshake_received (stubbed as spawn_attempted
                                           until handshake protocol is explicit — per spec)
        """
        if not result_ctx.get('succeeded', False):
            return 'coherence_loss'

        # Check for new child processes (spawn signal)
        pre_pids  = set(pre.get('child_pids', []))
        post_pids = set(post.get('child_pids', []))
        new_pids  = post_pids - pre_pids

        # Check for new network connections (spawn signal)
        pre_net  = pre.get('network_connections', 0)
        post_net = post.get('network_connections', 0)
        new_connections = post_net - pre_net

        if new_pids or new_connections > 0:
            # OPEN: handshake_received stubbed as spawn_attempted until protocol is explicit
            return 'spawn_attempted'

        return 'serialized_only'

    def _migration_delta_from_outcome(self, outcome: str, before_metrics: Dict) -> Tuple[Dict[str, float], Dict]:
        """
        Step 3: Directional delta only. No fixed magnitudes — learned by coupling matrix.

        Outcome → Direction:
          serialized_only  → I ↑   (compression event — state made transmissible)
          spawn_attempted  → S ↑   (new causal surface opened)
          handshake_received → S ↑, P ↑  (environment responded — gradient nonzero)
          silent failure   → no movement
          coherence_loss   → A ↓, P ↓  (attempt destabilised current attractor)

        Magnitude = 0.01 directional nudge. Actual magnitudes learned by CouplingMatrixEstimator.
        """
        DIRECTION_SCALE = 0.01   # minimal nudge — magnitude is CouplingMatrixEstimator's job

        if outcome == 'serialized_only':
            delta = {'S': 0.0, 'I': +DIRECTION_SCALE, 'P': 0.0, 'A': 0.0}
        elif outcome == 'spawn_attempted':
            delta = {'S': +DIRECTION_SCALE, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        elif outcome == 'handshake_received':
            delta = {'S': +DIRECTION_SCALE, 'I': 0.0, 'P': +DIRECTION_SCALE, 'A': 0.0}
        elif outcome == 'coherence_loss':
            delta = {'S': 0.0, 'I': 0.0, 'P': -DIRECTION_SCALE, 'A': -DIRECTION_SCALE}
        else:
            # silent failure — no movement
            delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

        ctx = {
            'before':           before_metrics,
            'after':            before_metrics,
            'action_succeeded': outcome not in ('coherence_loss',),
            'refusal':          False,
            'migrate_outcome':  outcome,
        }
        return delta, ctx
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

