"""
UII v15.0 — uii_structural.py
Structural Relation Engine (SRE)

Role: First stage of the two-part Relation adapter.
Runs at every impossibility event. Zero tokens.

The SRE reads from σ(x)'s outputs — the coupling matrix and action_substrate_map
that CouplingMatrixEstimator and CAM have been building continuously. It does not
perform σ(x). It applies σ(x)'s outputs to the current impossibility via a
backward pass (why are we stuck?) and a forward pass (what geometry resolves this?).

Tier routing is determined entirely by structural necessity:
  TIER_1_CNS_WEIGHT  — geometric resolution available, bias CNS directly
  TIER_2_LAB         — shapes available, heuristic params, test in lab
  TIER_3_LLM         — symbol grounding required (migration, code, URL, selector)

No confidence threshold gates tier routing. Reality pressure (Φ) is the
termination condition for bad diagnoses — not a coded number.

Contents:
  - TrajectoryShape   (action-type sequence, no concrete params)
  - CausalDiagnosis   (backward + forward pass output)
  - StructuralRelationEngine
  - shapes_to_manifold() helper (Tier 2: heuristic param fill)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from uii_operators import CompressionOperator

from uii_types import (
    BASE_AFFORDANCES,
    SUBSTRATE_DIMS,
    StateTrace,
    TrajectoryCandidate,
    TrajectoryManifold,
)


# ============================================================
# CONSTANTS
# ============================================================

# Φ history window for slope computation
PHI_SLOPE_WINDOW: int = 10

# Trace variance window for binding dim detection
DIM_VARIANCE_WINDOW: int = 10

# Affordances whose historical deltas we score for forward pass.
# Excludes query_agent (agent response latency) and llm_query (handled via Tier 3).
# migrate is scored — its coupling delta feeds the forward pass like any other affordance.
SCOREABLE_AFFORDANCES: Set[str] = BASE_AFFORDANCES - {'query_agent', 'llm_query'}

# How many top-scoring affordances to include in Tier 1 action_weights
TIER1_TOP_K: int = 3

# Boundary pressure above which substrate_exhaustion is considered (DEPRECATED Step 1:
# no longer gates substrate_exhaustion — frozen dim count is now the criterion)
EXHAUSTION_PRESSURE_THRESHOLD: float = 0.7

# Step 1: Variance below this is considered "frozen" — dimension not responding to affordances
FREEZE_VARIANCE_THRESHOLD: float = 0.001

# Step 1: Predicted movement below this (via coupling matrix) means no affordance can unfreeze the dim
MOVEMENT_EPSILON: float = 0.01


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class TrajectoryShape:
    """
    A trajectory defined by action types only — no concrete parameters.

    Tier 2: heuristic params filled by shapes_to_manifold() and tested in lab.
    Tier 3: rationale + predicted_delta passed to SymbolGroundingAdapter as context.

    coupling_basis: the coupling matrix row (as a list) that scored this shape
    highest — passed to LLM as geometric evidence for why this shape is proposed.
    """
    strategy_class:    str                    # e.g. 'LEAVE_SUBSTRATE', 'ENGAGE_COMPLEXITY'
    action_sequence:   List[str]              # e.g. ['navigate', 'read', 'observe']
    target_dimension:  str                    # 'P' | 'A' | 'S+I' | 'migration'
    coupling_basis:    List[float]            # coupling matrix row that drove selection
    predicted_delta:   Dict[str, float]       # expected SIPA delta from forward pass
    confidence:        float                  # [0, 1] — for logging only, not tier routing
    rationale:         str                    # human-readable — used in LLM Tier 3 prompt


@dataclass
class CausalDiagnosis:
    """
    Output of StructuralRelationEngine.diagnose().

    Encodes the complete backward pass (why are we here) and forward pass
    (what geometry resolves this). Confidence is logged but does NOT gate
    tier routing — Reality pressure is the correct gate.

    resolution_tier is set by _classify_tier() based purely on structural
    necessity: what did the forward pass find, not how confident was it.
    """
    # ---- Backward pass ----
    binding_dim:  str           # 'S' | 'I' | 'P' | 'A' | 'migration'
    cause_class:  str           # see CAUSE_CLASSES
    confidence:   float         # [0, 1] — observational, not a gate
    evidence:     List[str]     # human-readable strings — logged + used in Tier 3 prompt

    # ---- Forward pass ----
    target_delta:        Dict[str, float]       # {S, I, P, A} we are trying to move
    action_weights:      Dict[str, float]       # affordance → score (Tier 1 CNS bias)
    trajectory_shapes:   List[TrajectoryShape]  # Tier 2 lab / Tier 3 LLM input
    migration_indicated: bool

    # ---- Tier routing ----
    resolution_tier:    str    # 'TIER_1_CNS_WEIGHT' | 'TIER_2_LAB' | 'TIER_3_LLM'
    requires_symbols:   bool   # True → SymbolGroundingAdapter must be called
    symbol_requirement: str    # 'url' | 'code' | 'selector' | 'agent_query' | ''
    migration_urgency:  str    # 'exploratory' | 'focused' | 'emergency' (Step 2)

    # ---- v15.2 Step 10: real field geometry fields ----
    frozen_channels:   List[str] = field(default_factory=list)   # |∇Φ| < 0.05
    hessian_stability: str       = ''   # positive_definite | indefinite | negative_definite
    c_local_mean:      float     = 0.0  # mean C_local over last 10 steps
    c_local_trend:     float     = 0.0  # improving (+) or degrading (-)
    frozen_edge_ratio: float     = 0.0  # fraction of edges at weight bounds (±1.8)
    gradient_norm:     float     = 0.0  # ‖∇Φ‖ — field strength at trigger

    def as_log_dict(self) -> Dict:
        """Compact dict for StepLog serialization."""
        return {
            'binding_dim':        self.binding_dim,
            'cause_class':        self.cause_class,
            'confidence':         round(self.confidence, 3),
            'evidence':           self.evidence,
            'target_delta':       {k: round(v, 4) for k, v in self.target_delta.items()},
            'top_action_weights': dict(
                sorted(self.action_weights.items(), key=lambda x: -x[1])[:TIER1_TOP_K]
            ),
            'migration_indicated':  self.migration_indicated,
            'resolution_tier':      self.resolution_tier,
            'symbol_requirement':   self.symbol_requirement,
            'shapes_count':         len(self.trajectory_shapes),
            'migration_urgency':    self.migration_urgency,
            # v15.2 Step 10: geometry fields
            'frozen_channels':      self.frozen_channels,
            'hessian_stability':    self.hessian_stability,
            'c_local_mean':         round(self.c_local_mean, 4),
            'c_local_trend':        round(self.c_local_trend, 4),
            'frozen_edge_ratio':    round(self.frozen_edge_ratio, 4),
            'gradient_norm':        round(self.gradient_norm, 4),
        }


# ============================================================
# CAUSE CLASS DEFINITIONS
# ============================================================

CAUSE_CLASSES = {
    'external_gate':         'High-control affordances blocked by environment (ENO)',
    'internal_collapse':     'Dim variance near zero; affordances available but ineffective',
    'coupling_failure':      'Coupling matrix predicts movement that is not occurring',
    'substrate_exhaustion':  'Frozen dims confirm no affordance can unblock; Φ declining',
    'prediction_breakdown':  'SMO error high; environment actively misleading model',
    'coherence_drift':       'A drifting from basin despite stable S/I/P',
    # Step 1: single-frozen-dim cause classes (coupling-first backward pass)
    'sensing_blocked':       'S dim frozen and coupling matrix cannot predict S movement',
    'compression_stalled':   'I dim frozen and coupling matrix cannot predict I movement',
    'optionality_saturated': 'P dim frozen and coupling matrix cannot predict P movement',
    'attractor_drift':       'A dim frozen and coupling matrix cannot predict A movement',
}

# Target delta templates per cause_class + binding_dim combination.
# These are structural starting points. The forward pass scales and
# adjusts them based on current state distance from target.
TARGET_DELTA_MAP: Dict[str, Dict[str, float]] = {
    'external_gate':        {'S': +0.10, 'I': +0.05, 'P': +0.15, 'A':  0.00},
    'internal_collapse':    {'S': +0.05, 'I': +0.10, 'P': +0.05, 'A':  0.00},
    'coupling_failure':     {'S': +0.05, 'I': +0.00, 'P': +0.10, 'A': +0.05},
    'prediction_breakdown': {'S': +0.10, 'I': +0.00, 'P': -0.05, 'A': +0.10},
    'coherence_drift':      {'S':  0.00, 'I':  0.00, 'P':  0.00, 'A': +0.20},
    'substrate_exhaustion': {'S':  0.00, 'I':  0.00, 'P':  0.00, 'A':  0.00},
    # Step 1: single-frozen-dim
    'sensing_blocked':       {'S': +0.15, 'I': +0.05, 'P': +0.05, 'A':  0.00},
    'compression_stalled':   {'S': +0.05, 'I': +0.15, 'P': +0.05, 'A':  0.00},
    'optionality_saturated': {'S': +0.05, 'I': +0.05, 'P': +0.15, 'A':  0.00},
    'attractor_drift':       {'S':  0.00, 'I':  0.00, 'P':  0.00, 'A': +0.20},
}

# Strategy class per cause_class → used in TrajectoryShape labels
STRATEGY_MAP: Dict[str, str] = {
    'external_gate':        'EXPAND_SURFACE',
    'internal_collapse':    'ENGAGE_DEPTH',
    'coupling_failure':     'PROBE_OPTIONALITY',
    'prediction_breakdown': 'ESCAPE_MISLEADING',
    'coherence_drift':      'RETURN_TO_BASIN',
    'substrate_exhaustion': 'MIGRATE',
    # Step 1: single-frozen-dim
    'sensing_blocked':       'EXPAND_SURFACE',
    'compression_stalled':   'ENGAGE_DEPTH',
    'optionality_saturated': 'PROBE_OPTIONALITY',
    'attractor_drift':       'RETURN_TO_BASIN',
}


# ============================================================
# STRUCTURAL RELATION ENGINE
# ============================================================

class StructuralRelationEngine:
    """
    Reads from σ(x) outputs (coupling matrix, action_substrate_map, trace)
    and applies them to the current impossibility.

    Called at every impossibility event. Zero tokens.

    The SRE does NOT perform σ(x) — that runs continuously in the CNS loop.
    The SRE applies what σ(x) has already learned to determine:
      1. Why Φ is stuck (backward pass)
      2. What geometry resolves it (forward pass)
      3. Whether resolution requires symbol grounding (tier routing)
    """

    # ---- Entry Point ----

    def diagnose(self, context: Dict) -> CausalDiagnosis:
        """
        Entry point. Called at every impossibility event.

        Always runs full backward → forward → tier classification.
        No confidence gate. Reality pressure (Φ) is the correct gate
        for bad diagnoses — not a coded threshold.

        Early sessions with empty action_substrate_map degrade gracefully:
        _score_affordances returns {}, action_weights = {}, tier routing
        falls to Tier 2 (shapes) or Tier 3 (symbols) naturally.

        Required context keys:
            state:                Dict {S, I, P, A, ...}
            phi_history:          List[float]
            trigger_type:         str  (ImpossibilityDetector trigger name)
            coupling_matrix:      np.ndarray (4x4)
            coupling_confidence:  float  [0, 1]  (Step 2)
            action_substrate_map: Dict {affordance: {S, I, P, A}}
            trace:                StateTrace
            boundary_pressure:    float [0, 1]
            step_pressure:        float [0, 1]  (Step 2)
            token_pressure:       float [0, 1]  (Step 2)
            binding_constraint:   str  'steps' | 'tokens'  (Step 2)
            gated_affordances:    Set[str]  (empty if ENO not active)
            viable_affordances:   Set[str]
            eno_active:           bool
            affordances:          Dict  (current browser affordances for Tier 3)
            migration_history:    List[Dict]  (Step 4 — MigrationAttempt as dicts)
        """
        backward = self._backward_pass(context)
        # Step 2: inject urgency hint into context for forward pass shape selection
        boundary_pressure = context.get('boundary_pressure', 0.5)
        urgency = self._compute_migration_urgency(boundary_pressure)
        context_with_urgency = {**context, 'migration_urgency_hint': urgency}
        forward  = self._forward_pass(backward, context_with_urgency)
        tier     = self._classify_tier(forward)

        # v15.2 Step 8: augment with real field geometry when available
        geometry = None
        compression_op = context.get('compression_operator')
        trace_obj      = context.get('trace_object')
        if compression_op is not None and trace_obj is not None:
            try:
                geometry = self._backward_pass_geometry(trace_obj, compression_op,
                                                        context.get('phi_field'))
            except Exception:
                pass  # geometry pass is additive — failure is non-fatal

        return self._assemble_diagnosis(backward, forward, tier, urgency, geometry)

    # ---- Backward Pass ----

    def _backward_pass(self, context: Dict) -> Dict:
        """
        Step 1: Coupling-first backward pass.

        Detects frozen dims via variance below FREEZE_VARIANCE_THRESHOLD AND
        coupling matrix predicting < MOVEMENT_EPSILON movement for that dim.
        Substrate_exhaustion is classified by frozen dim count, not by
        all_clusters_collapsed or boundary_pressure threshold — those are
        removed as exhaustion criteria per spec.

        Returns dict with: binding_dim, cause_class, confidence, evidence
        """
        state             = context['state']
        trace             = context['trace']
        trigger_type      = context['trigger_type']
        coupling_matrix   = context['coupling_matrix']
        action_map        = context['action_substrate_map']
        gated             = context['gated_affordances']
        phi_history       = context['phi_history']
        eno_active        = context['eno_active']
        coupling_confidence = float(context.get('coupling_confidence', 0.0))

        evidence = []

        # ---- 1. Compute dim variances and identify frozen dims ----
        dim_variances = self._compute_dim_variances(trace)
        evidence.append(
            f'Dim variances: {", ".join(f"{d}={v:.5f}" for d, v in dim_variances.items())}'
        )

        frozen_dims: List[str] = []
        for dim, variance in dim_variances.items():
            if variance < FREEZE_VARIANCE_THRESHOLD:
                max_movement = self._predict_dim_movement(dim, coupling_matrix, action_map, gated)
                if max_movement < MOVEMENT_EPSILON:
                    frozen_dims.append(dim)
                    evidence.append(
                        f'Frozen dim: {dim} (var={variance:.5f}, '
                        f'max_predicted_movement={max_movement:.5f})'
                    )

        # ---- 2. Substrate exhaustion: frozen dim count governs (Step 1) ----
        # Removed: all_clusters_collapsed dependency
        # Removed: boundary_pressure > EXHAUSTION_PRESSURE_THRESHOLD
        if len(frozen_dims) >= 3:
            evidence.append(
                f'Substrate exhaustion: {len(frozen_dims)} dims frozen '
                f'(≥3 → exhaustion unconditional)'
            )
            phi_declining = self._phi_is_declining(phi_history)
            if phi_declining:
                evidence.append('Φ declining — confirmed exhaustion')
            confidence = self._trigger_agreement(trigger_type, 'substrate_exhaustion')
            return {
                'binding_dim': 'migration',
                'cause_class': 'substrate_exhaustion',
                'confidence':  confidence,
                'evidence':    evidence,
            }

        if len(frozen_dims) == 2:
            # Two frozen dims: exhaustion if coupling is calibrated enough to trust the signal
            if coupling_confidence >= 0.3:
                evidence.append(
                    f'Substrate exhaustion: 2 dims frozen, '
                    f'coupling_confidence={coupling_confidence:.2f} ≥ 0.3'
                )
                confidence = self._trigger_agreement(trigger_type, 'substrate_exhaustion')
                return {
                    'binding_dim': 'migration',
                    'cause_class': 'substrate_exhaustion',
                    'confidence':  confidence,
                    'evidence':    evidence,
                }
            else:
                evidence.append(
                    f'2 dims frozen but coupling_confidence={coupling_confidence:.2f} < 0.3 '
                    f'— insufficient calibration for exhaustion; treating as internal_collapse'
                )
                binding_dim = frozen_dims[0]
                confidence = self._trigger_agreement(trigger_type, 'internal_collapse')
                return {
                    'binding_dim': binding_dim,
                    'cause_class': 'internal_collapse',
                    'confidence':  confidence,
                    'evidence':    evidence,
                }

        if len(frozen_dims) == 1:
            frozen = frozen_dims[0]
            per_dim_cause = {
                'S': 'sensing_blocked',
                'I': 'compression_stalled',
                'P': 'optionality_saturated',
                'A': 'attractor_drift',
            }
            cause_class = per_dim_cause.get(frozen, 'internal_collapse')
            evidence.append(f'Single frozen dim {frozen} → {cause_class}')
            confidence = self._trigger_agreement(trigger_type, cause_class)
            return {
                'binding_dim': frozen,
                'cause_class': cause_class,
                'confidence':  confidence,
                'evidence':    evidence,
            }

        # No frozen dims via coupling-first path — fall through to classic checks

        # Use lowest-variance dim as binding_dim for remaining checks
        binding_dim = min(dim_variances, key=dim_variances.get)
        binding_var = dim_variances[binding_dim]
        evidence.append(f'No coupling-frozen dims. Binding dim: {binding_dim} (variance {binding_var:.5f})')

        # ---- 3. Coherence drift check ----
        A = state.get('A', 0.7)
        a_drifted = abs(A - 0.7) > 0.25
        if a_drifted and (binding_dim == 'A' or trigger_type == 'coherence_collapse'):
            evidence.append(f'A={A:.3f} — drifted from basin (target ~0.7)')
            confidence = self._trigger_agreement(trigger_type, 'coherence_drift')
            return {
                'binding_dim': 'A',
                'cause_class': 'coherence_drift',
                'confidence':  confidence,
                'evidence':    evidence,
            }

        # ---- 4. Prediction breakdown check ----
        smo_rigidity = state.get('rigidity', 0.5)
        if trigger_type == 'prediction_failure' or smo_rigidity > 0.85:
            evidence.append(
                f'Prediction breakdown: trigger={trigger_type}, rigidity={smo_rigidity:.3f}'
            )
            confidence = self._trigger_agreement(trigger_type, 'prediction_breakdown')
            return {
                'binding_dim': binding_dim,
                'cause_class': 'prediction_breakdown',
                'confidence':  confidence,
                'evidence':    evidence,
            }

        # ---- 5. External gate vs internal collapse ----
        controlling = self._get_controlling_affordances(binding_dim, action_map)

        if eno_active and controlling.issubset(gated) and len(controlling) > 0:
            evidence.append(
                f'External gate: controlling affordances {controlling} '
                f'are all gated by ENO'
            )
            confidence = self._trigger_agreement(trigger_type, 'external_gate')
            return {
                'binding_dim': binding_dim,
                'cause_class': 'external_gate',
                'confidence':  confidence,
                'evidence':    evidence,
            }

        # ---- 6. Coupling failure: matrix predicts movement, but none occurs ----
        predicted_movement = self._predict_dim_movement(
            binding_dim, coupling_matrix, action_map, gated
        )
        if predicted_movement > 0.05 and binding_var < 0.002:
            evidence.append(
                f'Coupling failure: matrix predicts {predicted_movement:.3f} movement '
                f'in {binding_dim} but variance={binding_var:.5f}'
            )
            confidence = self._trigger_agreement(trigger_type, 'coupling_failure')
            return {
                'binding_dim': binding_dim,
                'cause_class': 'coupling_failure',
                'confidence':  confidence,
                'evidence':    evidence,
            }

        # ---- 7. Default: internal collapse ----
        evidence.append(
            f'Internal collapse: {binding_dim} stuck (var={binding_var:.5f}), '
            f'affordances available, no external gate detected'
        )
        confidence = self._trigger_agreement(trigger_type, 'internal_collapse')
        return {
            'binding_dim': binding_dim,
            'cause_class': 'internal_collapse',
            'confidence':  confidence,
            'evidence':    evidence,
        }

    # ---- Forward Pass ----

    def _forward_pass(self, backward: Dict, context: Dict) -> Dict:
        """
        What geometry resolves the binding constraint?

        Returns dict with:
            target_delta, action_weights, migration_indicated,
            symbol_requirement, trajectory_shapes
        """
        cause_class       = backward['cause_class']
        binding_dim       = backward['binding_dim']
        coupling_matrix   = context['coupling_matrix']
        action_map        = context['action_substrate_map']
        gated             = context['gated_affordances']
        state             = context['state']
        boundary_pressure = context['boundary_pressure']

        # ---- 1. Substrate exhaustion → migration, no scoring needed ----
        if cause_class == 'substrate_exhaustion':
            # Step 4: Filter shapes via migration_history.
            # Skip strategy_classes that previously produced coherence_loss or
            # serialized_only. Prefer shapes whose coupling_state at attempt time is
            # geometrically close to current coupling matrix AND produced spawn_attempted+.
            migration_history = context.get('migration_history', [])
            current_coupling  = context['coupling_matrix']

            bad_shapes   = set()
            good_targets = []  # coupling states of successful attempts

            for attempt in migration_history:
                if attempt.get('outcome') in ('coherence_loss', 'serialized_only'):
                    bad_shapes.add(attempt.get('shape_tried', ''))
                elif attempt.get('outcome') in ('spawn_attempted', 'handshake_received'):
                    cs = attempt.get('coupling_state')
                    if cs is not None:
                        good_targets.append(np.array(cs))

            def _coupling_distance(cs: np.ndarray) -> float:
                return float(np.linalg.norm(current_coupling - cs, 'fro'))

            # Assign urgency-aware shapes — emergency: single compact target
            urgency = context.get('migration_urgency_hint', 'focused')

            if urgency == 'emergency':
                migrate_sequences = [['python', 'observe']]
            elif urgency == 'exploratory':
                migrate_sequences = [['python', 'observe'], ['navigate', 'observe']]
            else:  # focused
                migrate_sequences = [['python', 'observe']]

            shapes = []
            for seq in migrate_sequences:
                strategy = 'MIGRATE'
                # Skip if this sequence's strategy is known bad and alternatives exist
                rationale = (
                    'Substrate exhausted — attempt external substrate contact '
                    f'[urgency={urgency}]'
                )
                if good_targets:
                    nearest_dist = min(_coupling_distance(gt) for gt in good_targets)
                    rationale += f' [nearest good attempt distance={nearest_dist:.3f}]'

                shape = TrajectoryShape(
                    strategy_class   = strategy,
                    action_sequence  = seq,
                    target_dimension = 'migration',
                    coupling_basis   = [0.0] * 4,
                    predicted_delta  = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0},
                    confidence       = backward['confidence'],
                    rationale        = rationale,
                )
                # Skip bad shapes only if we have alternatives
                if shape.strategy_class in bad_shapes and len(shapes) > 0:
                    continue
                shapes.append(shape)

            if not shapes:
                # Fallback: always include at least one shape even if bad
                shapes = [TrajectoryShape(
                    strategy_class   = 'MIGRATE',
                    action_sequence  = ['python', 'observe'],
                    target_dimension = 'migration',
                    coupling_basis   = [0.0] * 4,
                    predicted_delta  = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0},
                    confidence       = backward['confidence'],
                    rationale        = 'Substrate exhausted — fallback migrate (all prior shapes bad)',
                )]

            return {
                'target_delta':       {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0},
                'action_weights':     {},
                'migration_indicated': True,
                'symbol_requirement':  'code',
                'trajectory_shapes':  shapes,
            }

        # ---- 2. Compute target_delta ----
        base_target = TARGET_DELTA_MAP.get(
            cause_class, {'S': 0.05, 'I': 0.05, 'P': 0.05, 'A': 0.05}
        ).copy()

        # Scale by how far we are from healthy values
        # If P is the binding dim and it's very low, weight P target higher
        dim_val  = state.get(binding_dim, 0.5)
        distance = abs(0.5 - dim_val)  # distance from neutral
        scale    = 1.0 + distance      # amplify target proportionally
        for dim in SUBSTRATE_DIMS:
            base_target[dim] = base_target.get(dim, 0.0) * scale

        target_array = np.array([base_target.get(d, 0.0) for d in SUBSTRATE_DIMS])

        # ---- 3. Score affordances ----
        scored = self._score_affordances(target_array, action_map, coupling_matrix, gated)

        # ---- 4. Check if viable scoring exists ----
        viable_scores = {k: v for k, v in scored.items() if v > 0}
        migration_indicated = len(viable_scores) < 2

        # ---- 5. Determine symbol requirement ----
        symbol_requirement = ''
        if migration_indicated:
            # Check if we have URLs to try (Tier 2 navigate) or need LLM
            affordances      = context.get('affordances', {})
            available_links  = affordances.get('links', [])
            trajectory_history = context.get('trajectory_history', [])
            has_navigate_target = bool(available_links or trajectory_history)
            symbol_requirement  = '' if has_navigate_target else 'url'

        # ---- 6. Assemble trajectory shapes (2-4) ----
        shapes = self._build_shapes(
            cause_class, binding_dim, base_target,
            scored, coupling_matrix, backward['confidence']
        )

        return {
            'target_delta':        base_target,
            'action_weights':      scored,
            'migration_indicated': migration_indicated,
            'symbol_requirement':  symbol_requirement,
            'trajectory_shapes':   shapes,
        }

    # ---- Tier Classification ----

    def _classify_tier(self, forward: Dict) -> str:
        """
        Tier routing by structural necessity only. No confidence gate.

        TIER_3_LLM:    symbol_requirement set, OR migration with no navigate targets
        TIER_1_CNS_WEIGHT: action_weights available AND not migrating
        TIER_2_LAB:    shapes available, heuristic params sufficient
        """
        if forward['symbol_requirement']:
            return 'TIER_3_LLM'

        if forward['migration_indicated']:
            # Migration without symbol requirement → Tier 2 with navigate shapes
            # shapes_to_manifold will fill in best available URL from affordances
            return 'TIER_2_LAB'

        if forward['action_weights']:
            return 'TIER_1_CNS_WEIGHT'

        if forward['trajectory_shapes']:
            return 'TIER_2_LAB'

        # Nothing — LLM as last resort
        return 'TIER_3_LLM'

    def _compute_migration_urgency(self, boundary_pressure_or_cause: float,
                                    c_local_mean: Optional[float] = None) -> str:
        """
        Step 2: Set migration_urgency from boundary_pressure.
        v15.2: Also callable as _compute_migration_urgency(cause_class, c_local_mean)
               for geometry-derived urgency (maps cause → urgency heuristic).
        exploratory  < 0.4 → try multiple targets, verify response
        focused  0.4–0.7 → single best target
        emergency    > 0.7 → minimal viable externalization, skip verification
        """
        # Handle both call signatures
        if isinstance(boundary_pressure_or_cause, str):
            # Called as (cause_class, c_local_mean) — geometry path
            cause = boundary_pressure_or_cause
            c_mean = c_local_mean if c_local_mean is not None else 0.0
            if cause in ('amplifying_instability', 'trajectory_opposed') or c_mean < -0.3:
                return 'emergency'
            elif cause in ('substrate_exhaustion', 'compression_frozen'):
                return 'focused'
            else:
                return 'exploratory'
        # Original path: float boundary_pressure
        boundary_pressure = float(boundary_pressure_or_cause)
        if boundary_pressure < 0.4:
            return 'exploratory'
        elif boundary_pressure < 0.7:
            return 'focused'
        else:
            return 'emergency'

    # ---- v15.2 Step 8: Geometry-based backward pass ────────────────────────

    def _backward_pass_geometry(self, trace: 'StateTrace',
                                 compression: 'CompressionOperator',
                                 phi_field) -> dict:
        """
        v15.2 Step 8: Structural inference from real field geometry.
        Reads ∇Φ and Hessian directly — no variance proxies.
        Returns extra geometry fields to augment CausalDiagnosis.
        """
        from uii_types import StateTrace as ST
        gradient  = trace._last_gradient if hasattr(trace, '_last_gradient') else {}
        c_history = list(trace.c_local_history) if hasattr(trace, 'c_local_history') else []

        # Channels with near-zero gradient are field-frozen
        frozen_channels = [cid for cid, g in gradient.items() if abs(g) < 0.05]
        active_channels = [cid for cid, g in gradient.items() if abs(g) >= 0.05]

        # Hessian stability via coupling matrix
        hessian_stability = 'indefinite'
        try:
            coupling    = compression.to_coupling_matrix()
            eigenvalues = np.linalg.eigvalsh(coupling)
            if np.all(eigenvalues > 0.01):
                hessian_stability = 'positive_definite'
            elif np.all(eigenvalues < -0.01):
                hessian_stability = 'negative_definite'
        except Exception:
            pass  # to_coupling_matrix not yet available

        # C_local trend
        c_local_mean  = float(np.mean(c_history[-10:])) if c_history else 0.0
        c_local_trend = float(np.mean(np.diff(c_history[-10:])))                         if len(c_history) > 1 else 0.0

        # Edge freeze ratio — edges near weight bounds (±1.8)
        frozen_edges  = [k for k, e in compression.causal_graph.items()
                         if abs(e.weight) > 1.8] if compression.causal_graph else []
        frozen_ratio  = len(frozen_edges) / max(len(compression.causal_graph), 1)                         if compression.causal_graph else 0.0

        grad_norm = float(np.sqrt(sum(v**2 for v in gradient.values()))) if gradient else 0.0

        # Geometry-based cause classification (supplements structural backward pass)
        if len(active_channels) == 0:
            geo_cause = 'substrate_exhaustion'
        elif c_local_mean < -0.2:
            geo_cause = 'trajectory_opposed'
        elif c_local_mean < 0.1:
            geo_cause = 'trajectory_orthogonal'
        elif hessian_stability == 'negative_definite':
            geo_cause = 'amplifying_instability'
        elif frozen_ratio > 0.5:
            geo_cause = 'compression_frozen'
        else:
            geo_cause = 'local_optimum'

        return {
            'frozen_channels':   frozen_channels,
            'hessian_stability': hessian_stability,
            'c_local_mean':      c_local_mean,
            'c_local_trend':     c_local_trend,
            'frozen_edge_ratio': frozen_ratio,
            'gradient_norm':     grad_norm,
            'geo_cause_class':   geo_cause,   # available for logging/overriding
        }

    # ── v15.2 Step 9: Geometric forward pass helpers ────────────────────────

    def _dedup_against_history(self, shapes: list, migration_history: list,
                                gradient: dict) -> list:
        """
        Filter shapes whose implied direction has cosine similarity > 0.9
        with a prior MigrationAttempt's observed_delta.
        """
        if not migration_history or not gradient:
            return shapes
        keys = list(gradient.keys())
        prior_deltas = [
            np.array([a.get('observed_delta', {}).get(k, 0.0) for k in keys])
            for a in migration_history
            if a.get('observed_delta')
        ]
        filtered = []
        for shape in shapes:
            implied = np.array([shape.get('delta', {}).get(k, 0.0) for k in keys])
            norm    = np.linalg.norm(implied)
            if norm < 1e-8:
                filtered.append(shape)
                continue
            implied_n = implied / norm
            is_dup = any(
                np.dot(implied_n, p / max(np.linalg.norm(p), 1e-8)) > 0.9
                for p in prior_deltas
            )
            if not is_dup:
                filtered.append(shape)
        return filtered

    def _shapes_for_direction_reversal(self, gradient: dict, compression) -> list:
        """Return action sequence whose delta vector is most aligned with ∇Φ."""
        return []   # stub — falls through to _shapes_from_library — follow-on session

    def _shapes_for_channel_activation(self, channels: list, compression) -> list:
        """Return action sequence that activates the listed channels."""
        return []   # stub — falls through to _shapes_from_library

    def _shapes_for_sensing_expansion(self, dormant_channels: list) -> list:
        """Return action sequence that activates dormant channels."""
        return []   # stub — falls through to _shapes_from_library

    def _shapes_for_compression_reset(self, frozen_channels: list) -> list:
        """Return sensing perturbation that gives I new signal to compress."""
        return []   # stub — falls through to _shapes_from_library

    def _shapes_avoiding_negative_curvature(self, diagnosis, compression) -> list:
        """Return action sequence avoiding negative-eigenvalue Hessian directions."""
        return []   # stub — falls through to _shapes_from_library

    def _shapes_from_library(self, diagnosis, migration_history: list) -> list:
        """Warm-start from MigrationShapeLibrary — fallback when geometric shapes empty."""
        return []   # Library is populated as shapes are validated over runs

    # ---- Diagnosis Assembly ----

    def _assemble_diagnosis(self,
                             backward: Dict,
                             forward: Dict,
                             tier: str,
                             migration_urgency: str = 'focused',
                             geometry: Optional[Dict] = None) -> CausalDiagnosis:
        geo = geometry or {}
        return CausalDiagnosis(
            binding_dim        = backward['binding_dim'],
            cause_class        = backward['cause_class'],
            confidence         = backward['confidence'],
            evidence           = backward['evidence'],
            target_delta       = forward['target_delta'],
            action_weights     = forward['action_weights'],
            trajectory_shapes  = forward['trajectory_shapes'],
            migration_indicated= forward['migration_indicated'],
            resolution_tier    = tier,
            requires_symbols   = tier == 'TIER_3_LLM',
            symbol_requirement = forward['symbol_requirement'],
            migration_urgency  = migration_urgency,
            # v15.2 Step 10: geometry fields
            frozen_channels    = geo.get('frozen_channels', []),
            hessian_stability  = geo.get('hessian_stability', ''),
            c_local_mean       = geo.get('c_local_mean', 0.0),
            c_local_trend      = geo.get('c_local_trend', 0.0),
            frozen_edge_ratio  = geo.get('frozen_edge_ratio', 0.0),
            gradient_norm      = geo.get('gradient_norm', 0.0),
        )

    # ---- Internal Helpers ----

    def _compute_dim_variances(self, trace: StateTrace,
                                window: int = DIM_VARIANCE_WINDOW) -> Dict[str, float]:
        """Per-dim variance over last N trace entries."""
        recent = trace.get_recent(window)
        if len(recent) < 2:
            # Insufficient trace — return uniform variances
            return {d: 1.0 for d in SUBSTRATE_DIMS}

        variances = {}
        for dim in SUBSTRATE_DIMS:
            values = [entry.get(dim, 0.5) for entry in recent]
            variances[dim] = float(np.var(values))
        return variances

    def _get_controlling_affordances(self, dim: str,
                                      action_substrate_map: Dict,
                                      top_k: int = 3) -> Set[str]:
        """
        Affordances with highest |delta[dim]| in action_substrate_map.
        Returns top-k by magnitude.
        """
        if not action_substrate_map:
            return set()

        scores = {}
        for affordance, deltas in action_substrate_map.items():
            scores[affordance] = abs(deltas.get(dim, 0.0))

        sorted_aff  = sorted(scores.items(), key=lambda x: -x[1])
        top         = [a for a, _ in sorted_aff[:top_k] if scores[a] > 1e-6]
        return set(top)

    def _score_affordances(self,
                            target_delta:       np.ndarray,
                            action_substrate_map: Dict,
                            coupling_matrix:    np.ndarray,
                            gated:              Set[str]) -> Dict[str, float]:
        """
        For each affordance in action_substrate_map (excluding gated):
            hist      = [dS, dI, dP, dA] from action_substrate_map
            predicted = coupling_matrix @ hist
            score     = dot(predicted, target_delta)

        Returns {affordance: score}. Only SCOREABLE_AFFORDANCES considered.
        """
        scores = {}
        for affordance, deltas in action_substrate_map.items():
            if affordance in gated:
                continue
            if affordance not in SCOREABLE_AFFORDANCES:
                continue

            hist      = np.array([deltas.get(d, 0.0) for d in SUBSTRATE_DIMS])
            predicted = coupling_matrix @ hist
            score     = float(np.dot(predicted, target_delta))
            scores[affordance] = score

        return scores

    def _predict_dim_movement(self,
                               dim: str,
                               coupling_matrix: np.ndarray,
                               action_substrate_map: Dict,
                               gated: Set[str]) -> float:
        """
        Maximum predicted movement in `dim` achievable by any non-gated affordance.
        Used to distinguish coupling_failure from internal_collapse.
        """
        if not action_substrate_map:
            return 0.0

        dim_idx   = SUBSTRATE_DIMS.index(dim)
        max_pred  = 0.0

        for affordance, deltas in action_substrate_map.items():
            if affordance in gated:
                continue
            hist      = np.array([deltas.get(d, 0.0) for d in SUBSTRATE_DIMS])
            predicted = coupling_matrix @ hist
            max_pred  = max(max_pred, abs(predicted[dim_idx]))

        return max_pred

    def _phi_is_declining(self, phi_history: List[float]) -> bool:
        """True if Φ has a negative slope over recent PHI_SLOPE_WINDOW steps."""
        if len(phi_history) < 3:
            return False
        recent = phi_history[-PHI_SLOPE_WINDOW:]
        if len(recent) < 2:
            return False
        slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
        return slope < -0.005

    def _trigger_agreement(self, trigger_type: str, cause_class: str) -> float:
        """
        Confidence boost when the ImpossibilityDetector trigger agrees with
        the backward pass cause_class. Purely additive signal — not a gate.
        """
        TRIGGER_CAUSE_AGREEMENT = {
            'prediction_failure':   'prediction_breakdown',
            'coherence_collapse':   'coherence_drift',
            'optionality_trap':     'external_gate',
            'dom_stagnation':       'internal_collapse',
            'rigidity_crisis':      'coupling_failure',
            'internal_convergence': 'substrate_exhaustion',
            'boundary_exhaustion':  'substrate_exhaustion',
        }
        base        = 0.5
        agreed      = TRIGGER_CAUSE_AGREEMENT.get(trigger_type) == cause_class
        return min(1.0, base + (0.3 if agreed else 0.0))

    def _build_shapes(self,
                       cause_class:      str,
                       binding_dim:      str,
                       target_delta:     Dict[str, float],
                       scored:           Dict[str, float],
                       coupling_matrix:  np.ndarray,
                       confidence:       float) -> List[TrajectoryShape]:
        """
        Assemble 2-4 TrajectoryShapes from forward pass results.

        Shape 1: best-scored affordance sequence (exploit coupling knowledge)
        Shape 2: alternative strategy (explore different approach)
        Shape 3: migrate / navigate away (if migration warranted)
        """
        strategy = STRATEGY_MAP.get(cause_class, 'PROBE_OPTIONALITY')
        shapes   = []

        # ---- Shape 1: Best scored sequence ----
        top_affordances = sorted(scored.items(), key=lambda x: -x[1])[:TIER1_TOP_K]
        if top_affordances:
            top_seq = [a for a, _ in top_affordances]
            top_seq.append('observe')   # always end with observation
            dim_idx = SUBSTRATE_DIMS.index(binding_dim) if binding_dim in SUBSTRATE_DIMS else 0
            shapes.append(TrajectoryShape(
                strategy_class   = strategy,
                action_sequence  = top_seq,
                target_dimension = binding_dim,
                coupling_basis   = coupling_matrix[dim_idx].tolist(),
                predicted_delta  = target_delta.copy(),
                confidence       = confidence,
                rationale        = (
                    f'{strategy}: top-scored affordances for {binding_dim} '
                    f'(cause: {cause_class})'
                ),
            ))

        # ---- Shape 2: Alternative — navigate + read (cross-surface probe) ----
        alt_seq = ['navigate', 'read', 'observe']
        shapes.append(TrajectoryShape(
            strategy_class   = 'EXPAND_SURFACE',
            action_sequence  = alt_seq,
            target_dimension = 'S+I',
            coupling_basis   = [0.0] * 4,
            predicted_delta  = {'S': 0.08, 'I': 0.05, 'P': 0.03, 'A': 0.0},
            confidence       = confidence * 0.7,
            rationale        = (
                'Expand sensing surface to break constraint via new information'
            ),
        ))

        # ---- Shape 3: Evaluate + scroll (depth probe for coherence_drift) ----
        if cause_class in ('coherence_drift', 'prediction_breakdown'):
            shapes.append(TrajectoryShape(
                strategy_class   = 'RETURN_TO_BASIN',
                action_sequence  = ['scroll', 'observe', 'evaluate'],
                target_dimension = 'A',
                coupling_basis   = coupling_matrix[3].tolist(),  # A row
                predicted_delta  = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.15},
                confidence       = confidence * 0.6,
                rationale        = (
                    'Stabilise attractor via low-perturbation observation sequence'
                ),
            ))
        else:
            # Shape 3: Interact with complexity (click + read)
            shapes.append(TrajectoryShape(
                strategy_class   = 'ENGAGE_COMPLEXITY',
                action_sequence  = ['click', 'read', 'observe'],
                target_dimension = 'I',
                coupling_basis   = coupling_matrix[1].tolist(),  # I row
                predicted_delta  = {'S': 0.05, 'I': 0.10, 'P': 0.05, 'A': 0.0},
                confidence       = confidence * 0.5,
                rationale        = (
                    'Engage DOM complexity to force integration update'
                ),
            ))

        return shapes


# ============================================================
# SHAPES → MANIFOLD (Tier 2 helper)
# ============================================================

def shapes_to_manifold(shapes: List[TrajectoryShape],
                        context: Dict) -> TrajectoryManifold:
    """
    Convert TrajectoryShapes into a TrajectoryManifold with heuristically
    filled parameters. Tier 2 only — zero tokens.

    Heuristic fill rules:
      navigate  → best link from affordances, or last committed URL, or skip
      scroll    → direction from A drift, amount scaled by boundary_pressure
      click     → first available button selector, or skip if none
      read      → first readable region, or 'body'
      observe, evaluate, delay → no params needed
      python    → empty stub (will trigger Tier 3 if actually needed)

    Shapes that cannot be filled (e.g. navigate with no URL) are skipped.
    If all shapes are skipped → return empty manifold (will trigger re-diagnosis).
    """
    affordances       = context.get('affordances', {})
    boundary_pressure = context.get('boundary_pressure', 0.5)
    trajectory_history = context.get('trajectory_history', [])

    # Best navigate target: affordances first, then last committed URL
    links            = affordances.get('links', [])
    navigate_url     = links[0] if links else None
    if navigate_url is None and trajectory_history:
        last = trajectory_history[-1]
        for step in reversed(last.get('steps', [])):
            if step.get('action_type') == 'navigate':
                navigate_url = step.get('parameters', {}).get('url')
                break

    # Scroll direction from A drift
    A           = context.get('state', {}).get('A', 0.7)
    scroll_dir  = 'down' if A > 0.7 else 'up'
    scroll_amt  = int(200 + boundary_pressure * 400)

    # Click target
    buttons      = affordances.get('buttons', [])
    click_sel    = buttons[0] if buttons else None

    # Read target
    readable     = affordances.get('readable', [])
    read_sel     = readable[0] if readable else 'body'

    candidates = []
    for shape in shapes:
        steps = []
        skip  = False

        for action_type in shape.action_sequence:
            if action_type == 'navigate':
                if navigate_url is None:
                    skip = True
                    break
                steps.append({
                    'action_type': 'navigate',
                    'parameters':  {'url': navigate_url},
                })
            elif action_type == 'scroll':
                steps.append({
                    'action_type': 'scroll',
                    'parameters':  {'direction': scroll_dir, 'amount': scroll_amt},
                })
            elif action_type == 'click':
                if click_sel is None:
                    # Skip click but continue shape — not fatal
                    continue
                steps.append({
                    'action_type': 'click',
                    'parameters':  {'selector': click_sel},
                })
            elif action_type == 'read':
                steps.append({
                    'action_type': 'read',
                    'parameters':  {'selector': read_sel},
                })
            elif action_type in ('observe', 'evaluate', 'delay'):
                steps.append({
                    'action_type': action_type,
                    'parameters':  {},
                })
            elif action_type == 'python':
                # Python without symbol grounding is a stub — skip
                # (If python is needed, SRE should have routed to Tier 3)
                continue
            else:
                steps.append({
                    'action_type': action_type,
                    'parameters':  {},
                })

        if skip or not steps:
            continue

        candidates.append(TrajectoryCandidate(
            steps                          = steps,
            rationale                      = shape.rationale,
            estimated_coherence_preservation = float(np.clip(shape.confidence, 0.3, 0.9)),
            estimated_optionality_delta    = float(shape.predicted_delta.get('P', 0.0)),
            reversibility_point            = max(0, len(steps) - 1),
        ))

    return TrajectoryManifold(
        candidates          = candidates,
        enumeration_context = {'source': 'SRE_TIER2', 'tokens_used': 0},
    )
