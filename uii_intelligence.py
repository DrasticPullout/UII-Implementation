"""
UII v15.0 — uii_intelligence.py
Structural Inference / Symbol Grounding

Role: The Relation leg of the Mentat Triad. Invoked only at impossibility.
Enumerates trajectory candidates — never executes.

v15.0: Two-part Relation adapter.
  - StructuralRelationEngine (uii_structural.py) runs first at zero token cost.
  - SymbolGroundingAdapter (this file) called only when SRE sets resolution_tier=TIER_3_LLM.
  - RelationAdapter wires both together behind the IntelligenceAdapter interface.

LLM role is now narrowed to symbol grounding only:
  - SRE has already determined WHAT to do (backward + forward pass over σ(x) outputs)
  - LLM fills in concrete tokens: URLs, Python code, CSS selectors
  - LLM receives a compact geometry package, not a general impossibility dump

llm_query is now a member of BASE_AFFORDANCES. Its outcome (SIPA delta from the
committed trajectory) feeds CouplingMatrixEstimator like any other affordance.

Contents:
  - SYMBOL_GROUNDING_PROMPT      (minimal — geometry pre-solved by SRE)
  - SymbolGroundingAdapter       (replaces LLMIntelligenceAdapter)
  - RelationAdapter              (two-part wrapper: SRE + SymbolGroundingAdapter)
  - LLMIntelligenceAdapter       (DEPRECATED — alias for backward compat with v14 logs)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np
import json
import re
from collections import deque

from uii_types import (
    BASE_AFFORDANCES,
    SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    IntelligenceAdapter,
)
from uii_genome import TriadGenome
from uii_structural import (
    StructuralRelationEngine,
    CausalDiagnosis,
    TrajectoryShape,
    shapes_to_manifold,
)


# ============================================================
# SYMBOL GROUNDING PROMPT
# ============================================================
# Much smaller than v14 RELATION_ENGINE_PROMPT.
# SRE has already solved the geometry. The LLM fills symbols only.
# No Φ definition, no SIPA semantics, no Triad explanation needed.

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


def _format_shapes_block(shapes: List[TrajectoryShape]) -> str:
    """Format TrajectoryShape list for the LLM prompt."""
    lines = []
    for i, shape in enumerate(shapes, 1):
        lines.append(f"Shape {i}: {shape.strategy_class}")
        lines.append(f"  Sequence:  {' → '.join(shape.action_sequence)}")
        lines.append(f"  Target:    {shape.target_dimension}")
        lines.append(f"  Rationale: {shape.rationale}")
        pred = shape.predicted_delta
        delta_str = ', '.join(
            f"{d}={pred.get(d, 0):+.3f}"
            for d in ['S', 'I', 'P', 'A']
            if abs(pred.get(d, 0)) > 0.001
        )
        if delta_str:
            lines.append(f"  Predicted: {delta_str}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# SYMBOL GROUNDING ADAPTER
# ============================================================

class SymbolGroundingAdapter:
    """
    v15.0: Symbol grounding layer. Called only when SRE sets TIER_3_LLM.

    Receives a CausalDiagnosis (geometry already solved) and fills in
    concrete symbols — URLs, Python code, CSS selectors — using the LLM's
    pretrained token field.

    The LLM no longer reasons about Φ, SIPA, impossibility theory, or the
    Triad architecture. It does one job: map abstract action types to
    concrete tokens given the current environment.

    Parsing, validation, and trajectory history are unchanged from v14.
    """

    def __init__(self, llm_client):
        self.llm             = llm_client
        self.call_count      = 0
        self.trajectory_history = deque(maxlen=3)

    def ground_trajectories(self,
                             diagnosis: CausalDiagnosis,
                             context: Dict) -> TrajectoryManifold:
        """
        Build minimal prompt from CausalDiagnosis, call LLM, parse result.
        context provides current affordances for symbol resolution.

        Step 5: token_budget_remaining included in prompt so LLM can adapt
        code complexity and target count. Degrades gracefully — if LLM ignores
        the budget hint, no hard dependency.
        """
        self.call_count += 1

        affordances       = context.get('affordances', {})
        boundary_pressure = context.get('boundary_pressure', 0.0)
        shapes            = diagnosis.trajectory_shapes

        # Step 5: Compute token budget remaining from context
        token_budget    = context.get('token_budget', None)
        token_pressure  = context.get('token_pressure', None)
        binding_constraint = context.get('binding_constraint', 'steps')

        if token_budget is not None and token_pressure is not None:
            remaining = int(token_budget * (1.0 - token_pressure))
            token_budget_remaining = f"{remaining:,} (binding: {binding_constraint})"
        else:
            token_budget_remaining = "unknown"

        # Step 5: Reduce shapes for low-budget / emergency scenarios
        migration_urgency = getattr(diagnosis, 'migration_urgency', 'focused')
        if migration_urgency == 'emergency' or (
            token_pressure is not None and token_pressure > 0.8
        ):
            shapes = shapes[:1]  # single target only under emergency / high token pressure

        # Format evidence lines
        evidence_lines = "\n".join(f"    - {e}" for e in diagnosis.evidence)

        # Format affordances — keep compact
        links   = json.dumps(affordances.get('links', [])[:15])
        buttons = json.dumps(affordances.get('buttons', [])[:10])
        inputs  = json.dumps(affordances.get('inputs', [])[:8])

        is_bootstrap = affordances.get('bootstrap_state', False)
        current_url  = 'about:blank' if is_bootstrap else affordances.get('current_url', '')
        page_title   = '' if is_bootstrap else affordances.get('page_title', '')

        shapes_block = _format_shapes_block(shapes)

        prompt = SYMBOL_GROUNDING_PROMPT.format(
            binding_dim           = diagnosis.binding_dim,
            cause_class           = diagnosis.cause_class,
            evidence_lines        = evidence_lines,
            migration_indicated   = diagnosis.migration_indicated,
            symbol_requirement    = diagnosis.symbol_requirement or 'none',
            migration_urgency     = migration_urgency,
            shapes_block          = shapes_block,
            current_url           = current_url,
            page_title            = page_title,
            links                 = links,
            buttons               = buttons,
            inputs                = inputs,
            boundary_pressure     = boundary_pressure,
            token_budget_remaining= token_budget_remaining,
            n_shapes              = len(shapes),
        )

        response, tokens_used = self.llm.call(prompt)
        candidates = self._parse_trajectories(response)

        return TrajectoryManifold(
            candidates          = candidates,
            enumeration_context = {
                'tokens_used':  tokens_used,
                'source':       'SRE_TIER3_LLM',
                'binding_dim':  diagnosis.binding_dim,
                'cause_class':  diagnosis.cause_class,
                'migration':    diagnosis.migration_indicated,
                **context,
            }
        )

    # ---- Parsing (unchanged from v14) ----

    def _parse_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        """Parse LLM response with progressive degradation."""
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

        cleaned = cleaned.replace('}\n{', '},\n{')
        cleaned = cleaned.replace('} {', '}, {')
        cleaned = cleaned.strip()

        if cleaned.startswith('{') and not cleaned.startswith('['):
            cleaned = f'[{cleaned}]'

        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            return None

    def _extract_partial_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        candidates = []
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.findall(object_pattern, response):
            try:
                obj = json.loads(match)
                if 'steps' in obj and isinstance(obj['steps'], list):
                    candidate = TrajectoryCandidate(
                        steps=obj.get('steps', []),
                        rationale=obj.get('rationale', 'Partially recovered trajectory'),
                        estimated_coherence_preservation=float(
                            obj.get('estimated_coherence_preservation', 0.5)),
                        estimated_optionality_delta=float(
                            obj.get('estimated_optionality_delta', 0.0)),
                        reversibility_point=int(obj.get('reversibility_point', 0))
                    )
                    if len(candidate.steps) > 0:
                        candidates.append(candidate)
            except Exception:
                continue
        return candidates

    def _generate_fallback_trajectory(self) -> List[TrajectoryCandidate]:
        return [TrajectoryCandidate(
            steps=[{'action_type': 'observe', 'parameters': {}}],
            rationale='Symbol grounding fallback — observation only',
            estimated_coherence_preservation=0.3,
            estimated_optionality_delta=0.0,
            reversibility_point=0,
        )]

    def _validate_and_convert(self, trajectory_dicts) -> List[TrajectoryCandidate]:
        if not isinstance(trajectory_dicts, list):
            return []

        candidates = []
        for traj_dict in trajectory_dicts:
            try:
                candidate = TrajectoryCandidate(
                    steps=traj_dict.get('steps', []),
                    rationale=traj_dict.get('rationale', 'No rationale'),
                    estimated_coherence_preservation=float(
                        traj_dict.get('estimated_coherence_preservation', 0.5)),
                    estimated_optionality_delta=float(
                        traj_dict.get('estimated_optionality_delta', 0.0)),
                    reversibility_point=int(traj_dict.get('reversibility_point', 0))
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
                                     phi_final: float):
        """Record committed trajectory for context in future calls."""
        self.trajectory_history.append({
            'steps':     trajectory.steps,
            'rationale': trajectory.rationale,
            'phi_final': phi_final,
        })


# ============================================================
# RELATION ADAPTER (two-part wrapper)
# ============================================================

class RelationAdapter(IntelligenceAdapter):
    """
    v15.0: Two-part Relation leg.

    Stage 1 — StructuralRelationEngine (zero tokens, every impossibility):
      Reads σ(x) outputs (coupling matrix, action_substrate_map, trace).
      Produces CausalDiagnosis with tier routing.

    Stage 2 — SymbolGroundingAdapter (tokens, TIER_3_LLM only):
      Receives pre-solved geometry. Fills in concrete symbols.
      Called only when SRE sets requires_symbols=True.

    Tier 1 (CNS weight bias) is handled in uii_triad.py before this adapter
    is called — see MentatTriad.step() Tier 1 short-circuit.
    This adapter handles Tier 2 (shapes_to_manifold) and Tier 3 (LLM) only.
    """

    def __init__(self,
                 structural_engine: StructuralRelationEngine,
                 symbol_grounder: SymbolGroundingAdapter):
        self.structural = structural_engine
        self.grounder   = symbol_grounder

        # Expose call_count for MentatTriad logging compatibility
        self._last_tier: Optional[str] = None

    @property
    def call_count(self) -> int:
        """Token-spending calls only (Tier 3). Used by MentatTriad for LLM rate tracking."""
        return self.grounder.call_count

    def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
        """
        Run SRE diagnosis then route to Tier 2 or Tier 3.
        Tier 1 is short-circuited in MentatTriad.step() before this is called.

        context must include all keys required by StructuralRelationEngine.diagnose()
        plus 'trajectory_history' for shapes_to_manifold Tier 2 navigate fallback.
        """
        diagnosis = self.structural.diagnose(context)
        self._last_tier = diagnosis.resolution_tier

        # Inject trajectory history for Tier 2 navigate fallback
        context_with_history = {
            **context,
            'trajectory_history': list(self.grounder.trajectory_history),
        }

        if diagnosis.resolution_tier == 'TIER_3_LLM':
            return self.grounder.ground_trajectories(diagnosis, context_with_history)

        # Tier 2: heuristic param fill, no LLM
        return shapes_to_manifold(diagnosis.trajectory_shapes, context_with_history)

    def record_committed_trajectory(self,
                                     trajectory: TrajectoryCandidate,
                                     phi_final: float):
        """
        Pass through to SymbolGroundingAdapter trajectory history.
        Used by Tier 3 to provide recent committed trajectories as context.
        Also used by Tier 2 navigate fallback (last committed URL).
        """
        self.grounder.record_committed_trajectory(trajectory, phi_final)

    def get_last_tier(self) -> Optional[str]:
        """Most recent tier used. For StepLog / session_end diagnostics."""
        return self._last_tier


# ============================================================
# DEPRECATED — backward compatibility only
# ============================================================

class LLMIntelligenceAdapter(SymbolGroundingAdapter):
    """
    DEPRECATED in v15.0.

    v14.x bundled structural reasoning and symbol grounding into one class.
    In v15.0 these are separated: StructuralRelationEngine handles reasoning,
    SymbolGroundingAdapter handles symbol grounding.

    This alias is retained so any v14 code that instantiates
    LLMIntelligenceAdapter directly still runs without modification.
    It will behave as SymbolGroundingAdapter — i.e., without structural reasoning.
    Migrate callers to RelationAdapter for full v15 behaviour.
    """
    pass


# ============================================================
# LEGACY — retained for v14 log readers
# ============================================================

# IMPOSSIBILITY_DIRECTIVES and RELATION_ENGINE_PROMPT are no longer used
# by the main execution path. Retained here so any tooling that imports
# them for log analysis or debugging does not break.

IMPOSSIBILITY_DIRECTIVES: Dict = {}   # v15: directives moved into SRE CausalDiagnosis.evidence

RELATION_ENGINE_PROMPT: str = (
    "# DEPRECATED v15.0 — structural reasoning moved to StructuralRelationEngine\n"
    "# Symbol grounding prompt: see SYMBOL_GROUNDING_PROMPT\n"
)
