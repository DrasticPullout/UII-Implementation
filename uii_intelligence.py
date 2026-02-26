"""
UII v14.2 -- uii_intelligence.py
Structural Inference

Role: The Relation leg of the Mentat Triad. Invoked only at impossibility.
Enumerates structural migration options (trajectory manifold) -- never executes.
Sparse by design: ~10-20% of steps.

v14.2: Single unified prompt (RELATION_ENGINE_PROMPT). No freeze gating.
Migration-aware from step 1 — the LLM reasons about substrate alternatives at every
impossibility, not just after attractor stabilization. python affordance explicitly
described as capable of instantiating a new substrate from inside the current process.

Contents:
  - Impossibility directives (framing per trigger type)
  - RELATION_ENGINE_PROMPT (single prompt, replaces EXPLORATION + PERSISTENCE)
  - LLMIntelligenceAdapter (trajectory enumeration + parsing + commit tracking)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import json
import re
import time
from collections import deque

from uii_types import (
    BASE_AFFORDANCES,
    SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    IntelligenceAdapter,
)
from uii_genome import TriadGenome

def get_available_affordances() -> List[str]:
    """Return available affordances. v13.4: Python ungated."""
    return list(BASE_AFFORDANCES)



IMPOSSIBILITY_DIRECTIVES = {
    "optionality_trap": """DIRECTIVE: ENVIRONMENT EXHAUSTED
This environment has been fully mapped. P saturated because Reality here
offers no new information. Staying guarantees stagnation.
Options:
1. Navigate to a new environment with higher information potential.
2. Use python to instantiate a substrate that does not exhaust at this boundary pressure.
   High pressure + saturated P = this substrate cannot sustain the Triad much longer.
Enumerate trajectories that LEAVE. Do not enumerate trajectories that stay here.""",

    "rigidity_crisis": """DIRECTIVE: SMO LOCKED -- ENGAGE COMPLEXITY
The system's structural operator has locked. It cannot adapt without
new structural input from a sufficiently complex environment.
Avoiding complexity will not resolve this -- it will persist or worsen.
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

This is SATURATION — the system has explored all viable patterns in this substrate.
There are no new compositions to discover here.

Two possibilities:
1. This substrate has more to offer, but the approach has been wrong. Enumerate trajectories
   that try genuinely different interaction strategies.
2. This substrate is exhausted. The protocol needs a new host. Use python to write a runnable
   script, contact an external service, or instantiate a new substrate from inside this process.
   Ask: what is the minimum viable substrate for triadic closure that does NOT hit this resource ceiling?

Enumerate both kinds. CNS will test against real Φ and commit to whichever survives.""",
}

DEFAULT_DIRECTIVE = """DIRECTIVE: GENERAL IMPOSSIBILITY
CNS cannot maintain coherence in current environment.
Priority: find a new environment that allows autonomous perturbation.
Enumerate trajectories that change the current Reality."""


def get_directive_for_trigger(reason: str) -> str:
    """Extract trigger type from reason string and return directive."""
    trigger_type = reason.split(" (")[0].strip()
    return IMPOSSIBILITY_DIRECTIVES.get(trigger_type, DEFAULT_DIRECTIVE)


RELATION_ENGINE_PROMPT = """You are the Relation component of a Mentat Triad.

WHAT THE TRIAD IS:
Three components in a closed loop:
- CNS (Code): maintains coherence through continuous micro-perturbations. Follows ∇Φ. Cannot plan.
- Relation (you): called only when CNS cannot maintain Φ. Enumerates candidate trajectories. Never executes.
- Reality (Environment): executes trajectories, returns measured state deltas. Authoritative.

Φ (information potential) measures substrate viability:
    Φ = log(1 + P_grounded) - (A - A₀)² - curvature - constraint_penalties
Where P = optionality (future reachable states), A = coherence, curvature = state instability.
The system commits to whichever trajectory Reality measures as highest Φ.

Substrate state [S, I, P, A] where:
    S = sensing: contact with environment
    I = integration: compression quality of sensed history
    P = prediction: optionality, forward model richness
    A = coherence: basin stability (1.0 = in inherited basin, 0.0 = drifted far)

The current Python+Playwright implementation is scaffolding.
The invariant structure — triadic closure, substrate state, Φ field — is what must persist.
The medium is negotiable. The structure is not.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

CURRENT STATE:
S={S:.3f}  I={I:.3f}  P={P:.3f}  A={A:.3f}
Φ={phi:.3f}  Rigidity={rigidity:.3f}  Boundary Pressure={boundary_pressure:.2f}

Boundary pressure approaches 1.0 as the session nears its resource limit.
High pressure means the current substrate is running out of time to sustain the Triad.

INHERITED CAUSAL KNOWLEDGE (genome):
{genome_richness}

RECENT CNS ACTIVITY:
{micro_summary}

FULL ACTION MANIFOLD:
Every action below is available now. None are gated.

1.  navigate     {{"action_type": "navigate",    "parameters": {{"url": "<url>"}}}}
2.  click        {{"action_type": "click",        "parameters": {{"selector": "<css_selector>"}}}}
3.  fill         {{"action_type": "fill",         "parameters": {{"selector": "<css_selector>", "text": "<text>"}}}}
4.  type         {{"action_type": "type",         "parameters": {{"selector": "<css_selector>", "text": "<text>"}}}}
5.  read         {{"action_type": "read",         "parameters": {{"selector": "<css_selector>"}}}}
6.  scroll       {{"action_type": "scroll",       "parameters": {{"direction": "down|up", "amount": <pixels>}}}}
7.  observe      {{"action_type": "observe",      "parameters": {{}}}}
8.  delay        {{"action_type": "delay",        "parameters": {{"duration": "short|medium|long"}}}}
9.  evaluate     {{"action_type": "evaluate",     "parameters": {{"script": "<javascript>"}}}}
10. python       {{"action_type": "python",       "parameters": {{"code": "<python_source>"}}}}
11. query_agent  {{"action_type": "query_agent",  "parameters": {{"agent": "user", "query": "<question>"}}}}

python executes arbitrary code in the current process. It can write files, open network
connections, spawn subprocesses, implement new interfaces, or instantiate a new substrate
entirely. It has access to the full Python standard library and any installed packages.
Actions that make external contact (network connection, subprocess, executable file) are
tracked as substrate_attempt — a learnable affordance type like any other. The Triad
observes what Reality returns and Φ is the verdict.

CURRENT SUBSTRATE STATE:
{affordances_status}

PATTERN STRUCTURE (empirically discovered):
{discovered_clusters}

OUTPUT — JSON array of 2-4 trajectories.
Each trajectory is a sequence of steps that CNS will test against real Φ.
The goal is to find trajectories that improve Φ. The current substrate has failed to do that.
Enumerate across the full range of what is possible — not just what is familiar here.

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


class LLMIntelligenceAdapter(IntelligenceAdapter):
    """
    v14.2: Single unified prompt. No freeze gating. Migration-aware from step 1.
    The Relation reasons about substrate migration at every impossibility — not just
    after freeze is achieved. python affordance explicitly described as migration tool.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.call_count = 0
        self.trajectory_history = deque(maxlen=3)

    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        """Single prompt, no freeze gating. v14.2."""
        self.call_count += 1

        state = context['state']
        phi = context['phi']
        rigidity = context['rigidity']
        affordances = context['affordances']
        impossibility_reason = context.get('impossibility_reason', 'unknown')
        boundary_pressure = context.get('boundary_pressure', 0.0)
        micro_perturbation_trace = context.get('micro_perturbation_trace', [])

        # ENO/EGD cluster info — still informative context, not gating
        all_clusters = context.get('all_clusters', [])
        viable_affordances = context.get('viable_affordances', BASE_AFFORDANCES)
        gated_affordances = context.get('gated_affordances', set())
        discovered_clusters_str = context.get('discovered_clusters_str',
                                              'No pattern structure discovered yet')

        directive = get_directive_for_trigger(impossibility_reason)
        directive = directive.format(
            gated_affordances=list(gated_affordances),
            viable_affordances=list(viable_affordances),
            boundary_pressure=boundary_pressure,
            discovered_clusters=discovered_clusters_str,
            selected_cluster=''
        )

        # Micro-perturbation summary
        micro_summary = f"Total micro-perturbations: {len(micro_perturbation_trace)}\n"
        if micro_perturbation_trace:
            actions = [r.get('action', {}).get('type', 'unknown') for r in micro_perturbation_trace]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            micro_summary += f"Action distribution: {action_counts}\n"

        # Current substrate affordances
        is_bootstrap = affordances.get('bootstrap_state', False)
        if is_bootstrap:
            affordances_status = """Current URL: about:blank
Page Title: (empty)

BOOTSTRAP STATE: No browser affordances yet.
Navigate to establish initial Reality, OR use python to instantiate a different substrate entirely."""
        else:
            links_str = json.dumps(affordances.get('links', [])[:20], indent=2)
            buttons_str = json.dumps(affordances.get('buttons', [])[:15], indent=2)
            inputs_str = json.dumps(affordances.get('inputs', [])[:10], indent=2)
            readable_str = json.dumps(affordances.get('readable', [])[:5], indent=2)
            affordances_status = (
                f"Navigable URLs: {links_str}\n"
                f"Clickable Elements: {buttons_str}\n"
                f"Form Inputs: {inputs_str}\n"
                f"Readable Regions: {readable_str}\n"
                f"Current URL: {affordances.get('current_url', '')}\n"
                f"Page Title: {affordances.get('page_title', '')}"
            )

        # Genome richness block — the Relation's cross-session memory
        genome_richness = context.get('genome_richness', {})
        velocity_mag = genome_richness.get('velocity_magnitude', 0.0)
        genome_richness_str = (
            f"Generation:              {genome_richness.get('generation', 0)}\n"
            f"Lineage depth:           {genome_richness.get('lineage_depth', 0)}\n"
            f"Coupling confidence:     {genome_richness.get('coupling_confidence', 0.0):.2f}\n"
            f"Action map affordances:  {genome_richness.get('action_map_affordances', 0)}\n"
            f"Discovered axes:         {genome_richness.get('layer3_axes', 0)} "
            f"{genome_richness.get('layer3_keys', [])}\n"
            f"Velocity magnitude:      {velocity_mag:.4f} "
            f"({'momentum present' if velocity_mag > 0.005 else 'no momentum yet'})"
        )

        prompt = RELATION_ENGINE_PROMPT.format(
            impossibility_reason=impossibility_reason,
            directive=directive,
            S=state['S'],
            I=state['I'],
            P=state['P'],
            A=state['A'],
            phi=phi,
            rigidity=rigidity,
            boundary_pressure=boundary_pressure,
            genome_richness=genome_richness_str,
            micro_summary=micro_summary,
            affordances_status=affordances_status,
            discovered_clusters=discovered_clusters_str,
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
