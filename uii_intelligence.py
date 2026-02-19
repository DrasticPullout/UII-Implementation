"""
UII v14.1 -- uii_intelligence.py
Structural Inference

Role: The Relation leg of the Mentat Triad. Invoked only at impossibility.
Enumerates structural migration options (trajectory manifold) -- never executes.
Sparse by design: ~10-20% of steps. The LLM sees context, not substrate internals.

Contents:
  - Impossibility directives (framing per trigger type)
  - Relation prompts (exploration + persistence regimes)
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
You must navigate to a new environment entirely.
Priority override: new Reality > survivability in current Reality.
Enumerate trajectories that LEAVE this page. Pick URLs with high link
density or content volume. Do not enumerate trajectories that stay here.""",

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


RELATION_ENGINE_PROMPT_EXPLORATION = """You are the Relation component of a Mentat Triad (Code + LLM + Reality).

THE TRIAD:
Code (CNS) maintains coherence via micro-perturbations.
You (LLM) enumerate trajectory possibilities when CNS detects impossibility.
Reality executes and measures all trajectories.
System commits to trajectory with highest Phi.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

CURRENT STATE:
S={S:.3f} I={I:.3f} P={P:.3f} A={A:.3f} Phi={phi:.3f} Rigidity={rigidity:.3f}
Boundary Pressure: {boundary_pressure:.2f}

GENOME RICHNESS (inherited causal knowledge):
{genome_richness}

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
Code (CNS) evaluates candidate trajectories for Phi preservation.
You (LLM) propose candidate trajectories when CNS detects impossibility.
Reality executes trajectories and measures impact.

IMPOSSIBILITY DETECTED: {impossibility_reason}

{directive}

SUBSTRATE SCAFFOLDING:
This Python+Playwright implementation is temporary. The invariant structure is:
  - Triadic closure: T(x) = f_rel(f_self(x), f_env(x))
  - Substrate state [S, I, P, A] maintaining basin geometry
  - Operator coupling O₁-O₄ across time
  - Phi field integrating coherence + optionality

The code is scaffolding. The structure is what persists.

CURRENT STATE:
S={S:.3f} I={I:.3f} P={P:.3f} A={A:.3f} Phi={phi:.3f} Rigidity={rigidity:.3f}
Boundary Pressure: {boundary_pressure:.2f}

GENOME RICHNESS (inherited causal knowledge):
{genome_richness}

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

Where O₁-O₄ are discovered attractor operators, α-δ are scaling factors, acting on substrate state s(t).

Triadic closure must hold:
    T(x) = f_rel(f_self(x), f_env(x))

Your trajectories represent f_rel: relational transformations integrating self-state with measured environment.

CONSTRAINTS:
  - Triadic closure preserved
  - External measurement maintained
  - No internal assumption loops

COMPOSITION STRATEGY:
Maximize long-horizon ∫Phi dt by preserving basin stability while expanding optionality where safe.
Structural self-preservation naturally favors optimal persistence pathways.
Avoid actions that decouple operators or violate discovered covariance patterns.

Estimate for each trajectory:
  - coherence preservation (0.0-1.0)
  - optionality delta (0.0-1.0)

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
- CNS selects trajectories by measured Phi.
- JSON only, no commentary.
"""


class LLMIntelligenceAdapter(IntelligenceAdapter):
    """v13.2: Two-prompt system. v14: genome_richness added to context."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.call_count = 0
        self.trajectory_history = deque(maxlen=3)

    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        """v14: genome_richness passed to both prompts."""
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

        affordances_list = ", ".join(get_available_affordances())

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

        # v14: Build genome richness string
        genome_richness = context.get('genome_richness', {})
        genome_richness_str = (
            f"Generation: {genome_richness.get('generation', 0)}\n"
            f"Coupling confidence: {genome_richness.get('coupling_confidence', 0.0):.2f}\n"
            f"Action map affordances: {genome_richness.get('action_map_affordances', 0)}\n"
            f"Discovered axes: {genome_richness.get('layer3_axes', 0)} "
            f"{genome_richness.get('layer3_keys', [])}"
        )

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
            genome_richness=genome_richness_str,
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
