from __future__ import annotations  # lazy annotations for cross-module type hints

"""
UII v14.1 — uii_triad.py
Execution & Orchestration

Role: Assembles the Mentat Triad and runs it. The only file that imports from
all other modules. Contains MentatTriad (the orchestrator), StepLog (the record
of each step), and the main entry point.

The triad is: CNS (Coherence) + Relation (Intelligence) + Reality (Perturbation),
held together by Memory (Genome) and observed through Logging.

Contents:
  - StepLog (full record per step, v14.1 fields)
  - MentatTriad (assembles and runs the full triad)
  - Main entry point (Groq adapter + CLI)

Step callback hook:
  Set triad.on_step_complete = callable(StepLog) before calling run()
  to receive real-time step data for dashboards, monitoring, etc.
"""

from dataclasses import dataclass, asdict, field
import dataclasses
from typing import Dict, List, Tuple, Optional, Set, Callable
import numpy as np
import json
import copy
import time
from collections import deque
from pathlib import Path

from uii_types import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    AgentHandler, AVAILABLE_AGENTS, VELOCITY_FIELD, LAYER1_PARAMS,
    RealityAdapter, IntelligenceAdapter,
    CRKEvaluation, CRKVerdict,
)
from uii_operators import (
    SensingOperator, CompressionOperator, PredictionOperator,
    CoherenceOperator, OperatorConsistencyCheck, DEFAULT_CHANNELS,
    SelfModifyingOperator, SMOUpdate,
)
from uii_genome import TriadGenome, ModelFidelityMonitor, LineageCoherenceCheck, load_genome, PeakOptionalityTracker
from uii_reality import AttractorMonitor, CouplingMatrixEstimator, BrowserRealityAdapter
from uii_intelligence import RelationAdapter, SymbolGroundingAdapter, LLMIntelligenceAdapter
from uii_structural import StructuralRelationEngine
from uii_coherence import (
    ExteriorNecessitationOperator, ControlAsymmetryMeasure, ExteriorGradientDescent,
    LatentDeathClock, ContinuousRealityEngine,
    CNSMitosisOperator, ImpossibilityDetector, AutonomousTrajectoryLab,
)
from uii_fao import (
    ResidualTracker, ResidualExplainer, AxisAdmissionTest,
    classify_relation_failure, FailureAssimilationOperator,
)

def _json_default(obj):
    """Custom JSON serializer for types json.dumps can't handle."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _compute_gradient_diagnostics(gradient: Dict[str, float],
                                   active_floor: float = 1e-3
                                   ) -> Tuple[float, int]:
    """
    Compute gradient entropy and active channel count from a normalized gradient.

    gradient_entropy: Shannon entropy H over |∇Φ_i| distribution.
        High = gradient mass spread across many channels (early exploration).
        Low  = gradient mass concentrated on few channels (attractor or collapse).
        Distinguishing which: check gradient_active_channels.

    gradient_active_channels: count of channels where |∇Φ_i| > active_floor.
        Low entropy + stable/high count = structural compression (healthy attractor).
        Low entropy + falling count     = causal graph starvation (collapse signal).

    Returns (entropy, active_count).
    Entropy returned in nats. Empty or zero gradient returns (0.0, 0).
    """
    if not gradient:
        return 0.0, 0

    magnitudes = np.array([abs(v) for v in gradient.values()], dtype=float)
    active_count = int(np.sum(magnitudes > active_floor))

    total = magnitudes.sum()
    if total < 1e-12:
        return 0.0, active_count

    p = magnitudes / total
    # Mask zero entries to avoid log(0)
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    return entropy, active_count

@dataclass
class StepLog:
    """v13.5: Added CNS mitosis tracking. v14: genome richness tracking. v15: migration tracking."""
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
    attractor_status: str = "accumulating_stability_data"
    freeze_verified: bool = False
    attractor_identity_hash: Optional[str] = None
    degradation_progress: float = 0.0
    boundary_pressure: float = 0.0
    binding_constraint: Optional[str] = None
    tokens_used_this_step: int = 0
    # v14: mitosis fields — deprecated in v15 but retained for log compat
    mitosis_triggered: bool = False
    mitosis_trigger_type: Optional[str] = None
    mitosis_attempted: bool = False
    mitosis_success: bool = False
    mitosis_pattern: Optional[str] = None
    mitosis_parent_optionality: Optional[float] = None
    mitosis_child_optionality: Optional[float] = None
    substrate_gain_verified: bool = False
    # v14: genome richness tracking per step
    coupling_confidence: float = 0.0
    coupling_observations: int = 0
    action_map_affordances: int = 0
    discovered_axes: int = 0
    residual_explanation: Optional[str] = None
    # v14.1: virtual trajectory and model fidelity tracking
    virtual_mode_active: bool = False
    virtual_phi_predicted: Optional[float] = None
    model_fidelity: float = 0.5
    # v15: SRE per-step diagnostics
    sre_tier: Optional[str] = None
    sre_binding_dim: Optional[str] = None
    sre_cause_class: Optional[str] = None
    sre_confidence: Optional[float] = None
    sre_migration_indicated: bool = False
    # v15: migration tracking (Step 3+4)
    migration_attempt: bool = False
    migration_outcome: Optional[str] = None
    migration_urgency: Optional[str] = None
    sre_migration_shape: Optional[str] = None
    coupling_confidence_at_attempt: Optional[float] = None
    # v15.1: operator scalar proxies (parallel to existing S/I/P/A for comparison)
    operator_S:           float = 0.0
    operator_I:           float = 0.0
    operator_P:           float = 0.0
    operator_P_grounded:  float = 0.0
    operator_A:           float = 0.0
    # v15.1: coherence internals
    loop_closure:         float = 0.0
    signature_deviation:  float = 0.0
    s_i_consistency:      float = 0.0
    i_p_consistency:      float = 0.0
    p_a_consistency:      float = 0.0   # v15.3: replaces p_s_consistency
    smo_consistency:      float = 0.0   # v15.3: new — SMO invariant satisfaction
    # v15.1: sensing / prediction
    active_channel_count: int   = 0
    realized_horizon:     int   = 0
    # v15.1: CRK split verdict
    crk_coherent:         bool  = True
    crk_repair:           Optional[str] = None
    crk_violations_new:   List[str] = field(default_factory=list)
    # v15.1: SMO layer tracking
    smo_plasticity:       float = 0.5
    smo_permitted:        bool  = True
    smo_updates:          List[Dict] = field(default_factory=list)
    smo_withheld_layers:  List[str] = field(default_factory=list)
    smo_rollback:         bool  = False
    # v15.2: Φ geometry fields
    phi_geometry:            float = 0.0   # Φ from compression geometry (new)
    phi_legacy:              float = 0.0   # Φ from old formula — parallel validation
    c_local:                 float = 0.0   # gradient-trajectory alignment this step
    c_global:                float = 0.0   # running mean of c_local
    gradient_norm:           float = 0.0   # ‖∇Φ‖ — field strength
    gradient_top_channel:    str   = ''    # channel with highest |gradient component|
    egd_gradient_lost:       bool  = False # C_local below floor this step
    sre_hessian_stability:   str   = ''    # positive_definite | indefinite | negative_definite
    sre_c_local_mean:        float = 0.0   # mean C_local at SRE trigger
    sre_frozen_channels:     List[str] = field(default_factory=list)
    trajectory_path_quality: float = 0.0   # mean C_local of last lab trajectory
    # v15.3: gradient distribution diagnostics
    gradient_entropy:        float = 0.0   # H(∇Φ) — Shannon entropy over channel gradient magnitudes
    gradient_active_channels: int  = 0     # channels with |∇Φ_i| > 1e-3 (structural vs starvation)


@dataclass
class MigrationAttempt:
    """
    Step 4: Run-local record of a single migrate action attempt.
    Never written to ledger mid-run. Extracted into Layer 2 by FAO at run end.
    """
    step:           int
    shape_tried:    str           # TrajectoryShape.strategy_class at attempt time
    code_hash:      str           # sha256[:16] of the python code — for dedup
    observed_delta: Dict          # SIPA delta returned by migrate execute branch
    coupling_state: List          # 4x4 coupling matrix snapshot as list (JSON-serializable)
    outcome:        str           # 'serialized_only' | 'spawn_attempted' | 'handshake_received' | 'coherence_loss'


# ============================================================
# MODULE 5.6: RELATION FAILURE CLASSIFICATION
# ============================================================

def classify_relation_failure(trajectories: List, committed_success: bool,
                              violations: List, phi_delta: float) -> Dict:
    """Extract semantic failure pattern from Relation enumeration."""
    if committed_success:
        return None

    failure_type = "unknown"
    severity = 1.0

    if not trajectories or len(trajectories) == 0:
        failure_type = "enumeration_failed"
        severity = 2.0

    elif violations and len(violations) > 0:
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
        failure_type = "coherence_drift"
        severity = 1.6

    else:
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
    v14.1: Dynamic Predictive Genome.

    All v14 features preserved. New in v14.1:
    - TriadGenome Layer 4: lineage_history (last 5 generations)
    - TriadGenome Layer 1: velocity fields per parameter
    - ModelFidelityMonitor: global causal model quality signal
    - ProvisionalAxisManager: two-tier Layer 3 decay (0.5x / 0.8x)
    - AutonomousTrajectoryLab virtual mode: simulate against Layer 2 pre-reality
    - LineageCoherenceCheck: suppresses momentum in incoherent lineages
    - Virtual → real Φ delta tracked per committed trajectory
    """

    def __init__(self,
                 intelligence: IntelligenceAdapter,
                 reality: RealityAdapter,
                 micro_perturbations_per_check: int = 10,
                 log_path: str = 'mentat_triad_v14_log.jsonl',
                 step_budget: int = 100,
                 token_budget: int = 100000,
                 genome: Optional[TriadGenome] = None,
                 log_mode: str = 'minimal'):

        self.intelligence = intelligence
        self.reality = reality

        if genome is None:
            genome = TriadGenome()
        self.genome = genome

        # v14.1 session_genome: live working copy of the parent genome.
        # AutonomousTrajectoryLab reads from this, not from the frozen self.genome.
        # CouplingMatrixEstimator pushes provisional Layer 2 updates here during the
        # session (observation_count >= 20, confidence-weighted at 0.5x distilled).
        # distill_to_genome at session end reads self.genome (unchanged) — heritable
        # extraction is unaffected. Only the trajectory lab's causal model is live.
        self.session_genome = copy.deepcopy(genome)

        self.log_mode = log_mode

        self.fitness_metrics = {
            'freeze_achieved': False,
            'freeze_step': None,
            'tokens_to_freeze': 0,
            'survival_time': 0,
            'final_phi': 0.0,
            'migration_attempted': False,
        }

        # v14: Initialize from genome's inherited action map (empty on generation 0)
        inherited_action_map = genome.causal_model.get('action_substrate_map', None)
        # crk initialized here so it can be passed to reality_engine
        # (original crk init at line ~327 is a reassignment — kept for clarity)
        self.reality_engine = ContinuousRealityEngine(
            reality, inherited_action_map=inherited_action_map,
            crk=None  # crk not yet created; set below after CRKMonitor() init
        )

        self.impossibility_detector = ImpossibilityDetector()
        self.micro_perturbations_per_check = micro_perturbations_per_check

        # v15.1: Initialize SubstrateState with DASS operators.
        # v15.3: Seed from inherited operator geometry if present.
        _isc = {cid: dataclasses.replace(ch,
                    coverage    = genome.operator_s_channels.get(cid, {}).get('coverage', ch.coverage),
                    signal_rate = genome.operator_s_channels.get(cid, {}).get('signal_rate', ch.signal_rate))
                for cid, ch in DEFAULT_CHANNELS.items()} \
               if genome.operator_s_channels else dict(DEFAULT_CHANNELS)

        self.state = SubstrateState(
            sensing     = SensingOperator(channels=_isc),
            compression = CompressionOperator(
                causal_graph      = {},
                pattern_library   = {},
                residual_variance = {},
                prediction_errors = deque(maxlen=20),
                observation_count = 0,
            ),
            prediction  = PredictionOperator(
                channel_predictions  = {},
                edge_predictions     = {},
                realized_horizon     = genome.operator_p_horizon,
                prediction_accuracy  = dict(genome.operator_p_accuracy),
                virtual_trajectories = [],
            ),
            coherence   = CoherenceOperator(
                consistency         = OperatorConsistencyCheck(
                                          s_i_consistency=1.0,
                                          i_p_consistency=1.0,
                                          p_a_consistency=1.0,
                                          smo_consistency=1.0,
                                          loop_closure=1.0),
                consistency_history = deque(maxlen=20),
                loop_signature      = dict(genome.operator_a_signature),
                signature_deviation = 0.0,
                self_model          = {},
            ),
        )
        # Inherit scalar biases into smo for backward compat prediction_error tracking
        self.state.smo.rigidity = genome.rigidity_init

        # v15.1: sensing history for CompressionOperator.apply()
        self._sensing_history: deque = deque(maxlen=50)

        # v15.1: SelfModifyingOperator — layer-specific, replaces scalar SMO for model updates
        self.smo_v151 = SelfModifyingOperator()

        # Last predicted delta — needed for post-action CRK
        self._last_predicted_delta: Dict[str, float] = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        self._pending_smo_update:   Dict = {}
        self._last_smo_rollback:    bool = False
        # v15.3: previous micro-perturbation's SMO updates — threaded to coherence.apply()
        # for smo_consistency measurement. None on first step → smo_consistency = 1.0 (bootstrap).
        self._prev_smo_updates:     Optional[List] = None

        self.trace = StateTrace()

        # A0=1.0: A is now a derived measurement where 1.0 means system is in its
        # inherited attractor basin. The strain term β(A - 1.0)² creates gravitational
        # pull back toward basin geometry. This is structural, not a designer constant —
        # the target is always "in the inherited basin."
        self.phi_field = PhiField(A0=1.0)

        self.crk = CRKMonitor()
        self.reality_engine.crk = self.crk   # v15.1: wire CRK for pre-action filtering

        self.trajectory_lab = AutonomousTrajectoryLab(reality, self.crk, self.phi_field,
                                                      genome=self.session_genome)  # v14.1: live session copy, not frozen genome

        # v14.1: Model fidelity monitor
        self.model_fidelity_monitor = ModelFidelityMonitor()

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

        self.eno = ExteriorNecessitationOperator(
            activation_window=20,
            gating_threshold=0.6
        )
        self.cam = ControlAsymmetryMeasure()
        self.egd = ExteriorGradientDescent()
        # v15.2: wire cam/egd into reality_engine for gradient-alignment action scoring
        self.reality_engine.cam = self.cam
        self.reality_engine.egd = self.egd

        self.eno_activations = 0
        self.egd_steps = 0
        self.pattern_discoveries = 0

        self.pending_agent_queries: List[Dict] = []
        self.triad_id = f"triad_{int(time.time())}"

        self.attractor_monitor = AttractorMonitor(stability_window=10, phi_epsilon=0.01)
        self.reality.attractor_monitor_ref = self.attractor_monitor

        self.canonical_graph = {
            "nodes": ["S", "I", "P", "A", "SMO", "Phi"],
            "edges": [
                ("SMO", "S"), ("SMO", "I"), ("SMO", "P"), ("SMO", "A"),
                ("S", "Phi"), ("I", "Phi"), ("P", "Phi"), ("A", "Phi"),
                ("Phi", "SMO")
            ],
            "couplings": {
                "smo_rigidity": self.state.smo.rigidity,
                "phi_alpha": self.phi_field.alpha,
                "phi_beta": self.phi_field.beta,
                "phi_gamma": self.phi_field.gamma,
                "phi_A0": self.phi_field.A0
            }
        }

        self.phi_definition = {
            "formula": "α*log(1+P) - β*(A-A0)² - γ*curvature - CRK_penalty",
            "params": ["alpha", "beta", "gamma", "A0", "alpha_crk"]
        }

        self.invariant_spec = {
            "closure_rule": "topology_hash(child) == topology_hash(parent)",
            "optionality_rule": "perturbation_diversity(child) >= perturbation_diversity(parent)",
            "bounded_delta": 0.3
        }

        self.fao = FailureAssimilationOperator(
            memory_decay=0.95,
            inheritance_noise=0.1
        )

        self.mitosis_operator = CNSMitosisOperator(
            canonical_graph=self.canonical_graph,
            phi_definition=self.phi_definition,
            invariant_spec=self.invariant_spec,
            fao=self.fao,
            perturbation_samples=10
        )

        # v15: SRE per-session counters
        self.sre_tier1_count: int = 0
        self.sre_tier2_count: int = 0
        self.sre_tier3_count: int = 0
        self.sre_migration_indicated_count: int = 0
        self.sre_binding_dim_counts: Dict[str, int] = {}
        self.sre_cause_class_counts: Dict[str, int] = {}

        # Step 4: MigrationHistory — run-local, never written mid-run
        self.migration_history: List[MigrationAttempt] = []

        self.death_clock = LatentDeathClock(
            step_budget=step_budget,
            token_budget=token_budget
        )
        self.death_clock_termination = False

        # v14: Phi history for FAO reset detection and AxisAdmissionTest
        self.phi_history: List[float] = []

        # v14: Coupling estimator — restore from inherited genome if present
        if 'coupling_matrix' in genome.causal_model:
            self.coupling_estimator = CouplingMatrixEstimator.from_genome_entry(
                genome.causal_model['coupling_matrix'])
        else:
            self.coupling_estimator = CouplingMatrixEstimator()

        # v15.3: Per-session peak optionality tracking.
        # Finds the highest-Vol_opt coupling state this session, plus operator geometry
        # at that same step (sensing surface, prediction accuracy, coherence signature).
        # PeakOptionalityTracker.update() in distill_to_genome compares this against
        # the genome's cross-generation peak and keeps whichever is richer.
        self._session_peak_vol_opt: float = -1.0
        self._session_peak_step: int = -1
        self._session_peak_coupling_entry: Optional[Dict] = None
        self._session_peak_s_channels: Dict = {}
        self._session_peak_p_accuracy: Dict = {}
        self._session_peak_p_horizon: int = 0
        self._session_peak_a_signature: Dict = {}

        self.residual_tracker = ResidualTracker(maxlen=200)
        self.residual_explainer = ResidualExplainer()
        self.axis_admission = AxisAdmissionTest()

        self.log_file.write(json.dumps({
            'event': 'session_start',
            'version': '14.1',
            'timestamp': time.time(),
            'triad_id': self.triad_id,
            'generation': genome.generation,
            'genome': asdict(genome),
            'genome_richness': genome.richness_summary(),
            'inherited_action_map_affordances': len(inherited_action_map) if inherited_action_map else 0,
            'inherited_coupling_confidence': genome.causal_model.get('coupling_matrix', {}).get('confidence', 0.0),
            'lineage_depth': len(genome.lineage_history),
            'model_fidelity_inherited': genome.causal_model.get('model_fidelity', None),
            'micro_perturbations_per_check': micro_perturbations_per_check,
            'step_budget': step_budget,
            'token_budget': token_budget,
        }) + '\n')
        self.log_file.flush()

        if genome.generation > 0:
            richness = genome.richness_summary()
            print(f"\n[CONTINUING EVOLUTION — Generation {genome.generation}]")
            print(f"  Parent fitness: {genome.parent_fitness:.2f}")
            print(f"  S coverage mean: {genome.s_coverage_mean:.3f} ({richness['layer1_s_channels']} inherited channels)")
            print(f"  P horizon norm:  {genome.p_horizon_norm:.3f} ({genome.operator_p_horizon} steps)")
            print(f"  A loop closure:  {genome.a_loop_closure:.3f}")
            print(f"  Search: rigidity={genome.rigidity_init:.2f} phi_coherence={genome.phi_coherence_weight:.2f}")
            print(f"  Coupling confidence: {richness['coupling_confidence']:.2f} ({richness['coupling_observations']} obs)")
            print(f"  Action map: {richness['action_map_affordances']} affordances")
            print(f"  Discovered axes: {richness['layer3_axes']} {richness['layer3_keys']}")
            print(f"  Lineage depth: {richness['lineage_depth']} | Velocity magnitude: {richness['velocity_magnitude']:.4f}")

    def get_triad_state(self) -> Dict:
        """Build current Triad state for attractor monitoring."""
        discovered_clusters = []
        if self.egd.cluster_history:
            latest = self.egd.cluster_history[-1]
            discovered_clusters = [latest['cluster']]

        violations = self.crk.evaluate(self.state, self.trace, None)
        phi = self.phi_field.phi(self.state, self.trace)  # v15.2: crk_violations removed

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
        """Check for agent responses and integrate into substrate."""
        integrated_responses = []

        for query_info in list(self.pending_agent_queries):
            agent_name = query_info['agent']
            agent = AVAILABLE_AGENTS[agent_name]

            response = agent.get_response(self.triad_id)

            if response is not None:
                delta = {
                    'S': 0.02,
                    'I': 0.05,
                    'P': 0.03,
                    'A': 0.01
                }

                self.state.apply_delta(delta)
                self.trace.record(self.state)  # v15.2: uses _last_gradient from prior step

                integrated_responses.append({
                    'agent': agent_name,
                    'query': query_info['query'],
                    'response': response,
                    'delta': delta
                })

                self.pending_agent_queries.remove(query_info)

        return integrated_responses

    def respond_to_query(self, answer: str):
        """User provides response to pending query."""
        user_agent = AVAILABLE_AGENTS['user']
        user_agent.respond(self.triad_id, answer)

    def _session_peak_snapshot(self) -> Optional[Dict]:
        """Session peak optionality snapshot for distill_to_genome and session_end logging."""
        if self._session_peak_coupling_entry is None:
            return None
        return {
            'peak_vol_opt':        self._session_peak_vol_opt,
            'peak_step':           self._session_peak_step,
            'l2_coupling_matrix':  self._session_peak_coupling_entry,
            # Operator geometry at peak step — seeds L1b in child genome
            'operator_s_channels': self._session_peak_s_channels,
            'operator_p_accuracy': self._session_peak_p_accuracy,
            'operator_p_horizon':  self._session_peak_p_horizon,
            'operator_a_signature': self._session_peak_a_signature,
        }

    def _compute_a(self) -> float:
        """
        A as derived measurement of basin drift — not an environmental delta.

        Two reality-grounded, genome-referenced signals:

        1. Basin distance: distance between current operator scalars and the
           inherited genome operator geometry (s_coverage_mean, p_horizon_norm).
           Both are derived from real operator state at peak Vol_opt — not designer
           constants. Distance zero = system at inherited operator geometry.

        2. Coupling divergence: Frobenius norm of (live coupling matrix - inherited
           coupling matrix), normalized by inherited matrix norm. Measures how much
           the observed co-movement has drifted from the basin's inherited shape.

        A = 1 - drift, where drift combines both signals.
        A near 1.0: system is in its inherited attractor basin.
        A near 0.0: system has drifted far from coherent geometry.
        """
        # --- Signal 1: Basin distance ---
        # References are operator-geometry-derived scalars from peak Vol_opt.
        # Generation 0 (defaults 0.0): basin_distance = full distance from current
        # operator proxies to zero — this is correct: no inherited basin to compare to.
        s_dist = self.state.S - self.genome.s_coverage_mean
        p_dist = self.state.P - self.genome.p_horizon_norm
        # Normalize by max possible distance in [0,1]^2 space
        basin_distance = float(np.sqrt(s_dist**2 + p_dist**2) / np.sqrt(2.0))

        # --- Signal 2: Coupling divergence ---
        coupling_divergence = 0.0
        inherited_entry = self.genome.causal_model.get('coupling_matrix', {})
        coupling_confidence = inherited_entry.get('confidence', 0.0)

        if coupling_confidence > 0.0 and 'matrix' in inherited_entry:
            inherited_matrix = np.array(inherited_entry['matrix'])
            live_matrix = self.coupling_estimator.matrix

            inherited_norm = float(np.linalg.norm(inherited_matrix, 'fro'))
            if inherited_norm > 1e-6:
                diff_norm = float(np.linalg.norm(live_matrix - inherited_matrix, 'fro'))
                # Relative divergence, weighted by how confident the inherited matrix is
                coupling_divergence = (diff_norm / inherited_norm) * coupling_confidence

        # --- Combine: drift weighted by coupling availability ---
        # When coupling confidence is low, basin_distance dominates.
        # When coupling confidence is high, both contribute equally.
        drift = (basin_distance * (1.0 - 0.5 * coupling_confidence) +
                 coupling_divergence * 0.5 * coupling_confidence)

        drift = float(np.clip(drift, 0.0, 1.0))
        return 1.0 - drift
    
    def _compute_delta_i(self) -> float:
        """
        I: {S_i} → C   Compression quality of recent S history.

        DASS spec: I measures how much pattern/regularity exists in what the
        system has been sensing. Low variance in recent S observations means the
        environment is patterned and compressible → I rises. High variance means
        the environment is chaotic and incompressible → I falls.

        This is purely external: it observes the regularity of what Reality has
        delivered through the S layer. The trace is the Triad's record of those
        perturbations. No SMO. No prediction error. No internal signals.

        Lives in the Triad (not Reality) because compression quality requires
        history — Reality executes one action at a time and returns a delta.
        The Triad owns the trace.

        Crossover at S-variance ≈ 0.005:
            Below → patterned environment → compression improving → +delta_I
            Above → chaotic environment  → compression degrading → -delta_I

        Returns float in [-0.05, 0.05].
        Fewer than 3 trace entries → 0.0 (neutral, not punitive).
        """
        recent = self.trace.get_recent(10)
        if len(recent) < 3:
            return 0.0

        s_values   = [d['S'] for d in recent]
        s_variance = float(np.var(s_values))

        return float(np.clip(0.01 - 2.0 * s_variance, -0.05, 0.05))

    def _build_env_signal(self, observed_delta: Dict, context: Dict) -> Dict:
        """
        v15.1: Translate observed_delta + context into per-channel env_signal
        for SensingOperator.apply().
        Resolves open handshake question: ssh_remote active after migrate = handshake_received.
        """
        import os
        try:
            import psutil
            cpu_pct  = psutil.cpu_percent() / 100.0
            mem_pct  = psutil.virtual_memory().percent / 100.0
            self_pct = psutil.Process().memory_percent() / 100.0
        except Exception:
            cpu_pct = mem_pct = self_pct = 0.0

        signal: Dict = {}

        signal['browser'] = {
            'magnitude': abs(observed_delta.get('S', 0.0)),
            'rate':      context.get('interactive_density', 0.0),
            'coverage':  context.get('viewport_coverage', 0.0),
        }
        signal['resource_cpu'] = {
            'magnitude': cpu_pct, 'rate': 1.0, 'coverage': 1.0,
        }
        signal['resource_memory'] = {
            'magnitude': mem_pct, 'rate': 1.0, 'coverage': 1.0,
        }
        signal['process_self'] = {
            'magnitude': self_pct, 'rate': 1.0, 'coverage': 1.0,
        }
        signal['clock']    = {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
        signal['clock_rate'] = {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
        signal['api_llm']  = {
            'magnitude': observed_delta.get('llm_tokens_used', 0) / 1000.0,
            'rate':      float(observed_delta.get('llm_called', False)),
            'coverage':  1.0,
        }
        signal['os_signals'] = {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
        signal['env_vars']   = {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
        signal['entropy_source'] = {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}

        # Migration handshake — ssh_remote active after successful migrate
        if context.get('migrate_outcome') in ('spawn_attempted', 'handshake_received'):
            handshake_path = context.get('handshake_path', '/tmp/uii_handshake')
            if os.path.exists(handshake_path):
                signal['ssh_remote'] = {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
                context['migrate_outcome'] = 'handshake_received'

        return signal

    def _log_smo_block(self, verdict: 'CRKVerdict'):
        """Log a blocked/rolled-back SMO update."""
        blocked_constraints = [e.constraint for e in verdict.evaluations if e.blocks]
        self.log_file.write(json.dumps({
            'event':               'smo_blocked',
            'step':                self.step_count,
            'timestamp':           time.time(),
            'blocked_constraints': blocked_constraints,
            'repair':              verdict.repair,
            'rollback':            self._last_smo_rollback,
        }) + '\n')
        self.log_file.flush()

    def _restore_from_snapshot(self, snapshot: Dict):
        """
        SMO⁻¹: Restore operator state from snapshot (called on C1 post-action rollback).
        Only compression is restored — sensing always reflects Reality.
        """
        if snapshot is None:
            return
        # Rebuild compression from snapshot (best-effort — edges only)
        from uii_operators import CausalEdge
        restored_graph = {}
        for k_str, edge_dict in snapshot.get('compression_edges', {}).items():
            try:
                # k_str is str repr of (src, tgt) tuple
                import ast
                key = ast.literal_eval(k_str)
                restored_graph[key] = CausalEdge(**edge_dict)
            except Exception:
                pass
        if restored_graph:
            self.state = SubstrateState(
                sensing     = self.state.sensing,
                compression = dataclasses.replace(
                    self.state.compression, causal_graph=restored_graph
                ),
                prediction  = self.state.prediction,
                coherence   = self.state.coherence,
            )

    def _apply_scalar_smo(self, observed_delta: Dict, predicted_delta: Dict):
        """Backward compat: apply scalar SMO update (now handled in apply_delta)."""
        pass   # apply_delta() already called in operator update path

    def _recent_phi_trend(self, trace: 'StateTrace') -> float:
        """Compute recent Φ trend for field_state."""
        if len(trace) < 3:
            return 0.0
        recent = trace.get_recent(3)
        phis   = [r.get('phi', r.get('P', 0.0)) for r in recent]
        if len(phis) >= 2:
            return phis[-1] - phis[0]
        return 0.0

    def _build_sre_context(self,
                            trigger_type: str,
                            boundary_pressure: float,
                            affordances: Dict,
                            eno_active: bool,
                            gated_affordances: set,
                            viable_affordances: set) -> Dict:
        """
        v15: Package context for StructuralRelationEngine.diagnose().
        Called at every impossibility event before tier routing.

        Step 2: adds step_pressure, token_pressure, binding_constraint.
        Step 4: adds migration_history (as list of dicts for JSON safety).
        """
        # Step 2: pressure split
        dc = self.death_clock
        step_pressure   = dc.current_steps  / dc.step_budget  if dc.step_budget  > 0 else 0.0
        token_pressure  = dc.current_tokens / dc.token_budget if dc.token_budget > 0 else 0.0
        binding_constraint = dc.get_binding_constraint()

        # Step 4: migration_history as plain dicts (MigrationAttempt is a dataclass)
        migration_history_dicts = [
            {
                'step':          a.step,
                'shape_tried':   a.shape_tried,
                'code_hash':     a.code_hash,
                'observed_delta': a.observed_delta,
                'coupling_state': a.coupling_state,
                'outcome':       a.outcome,
            }
            for a in self.migration_history
        ]

        return {
            'state':                  self.state.as_dict(),
            'phi_history':            list(self.phi_history),
            'trigger_type':           trigger_type,
            'coupling_matrix':        self.coupling_estimator.matrix,
            # v15.2: provide live operator objects for new geometry-based backward pass
            'compression_operator':   self.state.compression,
            'phi_field':              self.phi_field,
            'trace_object':           self.trace,
            'coupling_confidence':    self.coupling_estimator.get_confidence(),   # Step 2
            'observation_count':      self.coupling_estimator.observation_count,  # Step 1 (coupling-first)
            'action_substrate_map':   self.genome.causal_model.get('action_substrate_map', {}),
            'trace':                  self.trace,
            'genome':                 self.genome,
            'boundary_pressure':      boundary_pressure,
            'step_pressure':          float(np.clip(step_pressure,  0.0, 1.0)),   # Step 2
            'token_pressure':         float(np.clip(token_pressure, 0.0, 1.0)),   # Step 2
            'binding_constraint':     binding_constraint,                          # Step 2
            'gated_affordances':      gated_affordances,
            'viable_affordances':     viable_affordances,
            'eno_active':             eno_active,
            'affordances':            affordances,
            'all_clusters_collapsed': self.egd.all_patterns_collapsed(),  # retained for logging
            'migration_history':      migration_history_dicts,             # Step 4
        }

    def step(self, verbose: bool = False) -> StepLog:
        """
        v13.2: Attractor monitoring + affordance expansion at freeze_verified.
        v13.3: Real-time fitness tracking.
        v14: CouplingMatrixEstimator + ResidualTracker wired every micro-perturbation.
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
        phi_before = self.phi_field.phi(self.state, self.trace)  # v15.2: crk_violations removed
        phi_legacy_before = self.phi_field.phi_legacy(self.state, self.trace, violations_before)  # parallel validation

        # v14: Track phi history
        self.phi_history.append(phi_before)

        if verbose:
            print(f"State: S={self.state.S:.3f}, I={self.state.I:.3f}, P={self.state.P:.3f}, A={self.state.A:.3f}")
            print(f"Φ={phi_before:.3f}, Rigidity={self.state.smo.rigidity:.3f}")

        eno_active = self.eno.check_activation(self.state.smo, self.trace, egd=self.egd)  # v15.2 Step 5: trigger via C_local

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

            # v15.3: phi_history threaded so CRE._build_field_state gets real Φ trend
            # for C7 evaluation. CRE already runs full pre-action CRK on the full
            # viable set — it filters, scores with phi_modifier, and applies repair
            # bias. No triad-level CRK re-evaluation needed or correct here.
            action = self.reality_engine.choose_micro_action(
                self.state, affordances, trace=self.trace,
                phi_history=self.phi_history,
            )

            if action['type'] == 'query_agent':
                action['params']['triad_id'] = self.triad_id

            predicted_delta = self.reality_engine.predict_delta(action, self.state)
            self._last_predicted_delta = predicted_delta

            observed_delta, context = self.reality.execute(
                action,
                boundary_pressure=boundary_pressure,
                state=self.state,
                coupling_confidence=self.coupling_estimator.get_confidence(),
            )

            # I: {S_i} → C  Compression quality of S history.
            observed_delta['I'] = self._compute_delta_i()

            # ── v15.1 Operator update sequence ───────────────────────────────
            prior_compression = self.state.compression   # snapshot before update

            env_signal  = self._build_env_signal(observed_delta, context)
            new_sensing = self.state.sensing.apply(env_signal, self.state.compression)
            self._sensing_history.append(new_sensing)

            new_compression = self.state.compression.apply(self._sensing_history)
            new_prediction, errors = self.state.prediction.observe_outcome(
                new_sensing, new_compression
            )
            # P→I feedback — closes the loop
            new_compression = new_compression.absorb_prediction_error(errors)
            new_prediction  = new_prediction.apply(new_compression)
            new_coherence   = self.state.coherence.apply(
                new_sensing, new_compression, new_prediction,
                smo_updates=self._prev_smo_updates,  # v15.3: previous step's SMO updates
            )

            # Pending SMO update = the prediction errors (what SMO would act on)
            self._pending_smo_update = errors

            # Post-action CRK — gate SMO update
            # v15.3: smo_plasticity threaded for C3 Non-Internalization evaluator
            post_verdict = self.crk.evaluate_post_action(
                proposed_smo_update = errors,
                observed_delta      = observed_delta,
                predicted_delta     = predicted_delta,
                sensing             = new_sensing,
                compression         = new_compression,
                prediction          = new_prediction,
                coherence           = new_coherence,
                prior_compression   = prior_compression,
                smo_plasticity      = self.smo_v151.plasticity,
            )

            self._last_smo_rollback = False

            if post_verdict.smo_permitted:
                # Grounded adaptation: apply SelfModifyingOperator
                # v15.3: capture plasticity before apply so C3 reattribute can suppress increase
                _pre_apply_plasticity = self.smo_v151.plasticity
                new_s, new_c, new_p, new_a, smo_updates_list = self.smo_v151.apply(
                    sensing           = new_sensing,
                    compression       = new_compression,
                    prediction        = new_prediction,
                    coherence         = new_coherence,
                    observed_delta    = observed_delta,
                    predicted_delta   = predicted_delta,
                    prediction_errors = errors,
                )
                # v15.3 C3 repair: if reattribute fired, suppress plasticity increase.
                # External constraint is not self-failure — do not absorb it as such.
                if post_verdict.repair == 'reattribute':
                    self.smo_v151.plasticity = min(
                        self.smo_v151.plasticity, _pre_apply_plasticity
                    )
                    self.smo_v151.rigidity = 1.0 - self.smo_v151.plasticity
                self.state = SubstrateState(
                    sensing     = new_s,
                    compression = new_c,
                    prediction  = new_p,
                    coherence   = new_a,
                )
                # Scalar SMO still runs on proxy values for backward compat
                self.state.apply_delta(observed_delta, predicted_delta)
            else:
                # SMO blocked — sensing always updates.
                # C1 (structural continuity violated): rollback compression to prior.
                # C4 (adaptation not grounded / bootstrap): keep new_compression so
                #   observation-derived graph formation accumulates. Only SMO weight
                #   updates are blocked — the co-movement graph is always observable.
                c1_post = next(
                    (e for e in post_verdict.evaluations
                     if e.constraint == 'C1' and e.blocks), None
                )
                if c1_post:
                    self._last_smo_rollback = True
                    compression_to_use = prior_compression  # revert corrupted graph
                else:
                    compression_to_use = new_compression    # keep observation, skip SMO weights

                self.state = SubstrateState(
                    sensing     = new_sensing,
                    compression = compression_to_use,
                    prediction  = self.state.prediction,
                    coherence   = new_coherence,
                )
                # Scalar SMO still updates for error tracking
                self.state.apply_delta(observed_delta, predicted_delta)
                smo_updates_list = []

            # v15.3: store this perturbation's SMO updates for next coherence.apply()
            self._prev_smo_updates = smo_updates_list if smo_updates_list else None

            # ─────────────────────────────────────────────────────────────────

            # v15.2 Step 3: compute gradient every step, thread to trace.record()
            _gradient = self.phi_field.gradient(self.state, self.trace)
            self.trace.record(self.state, _gradient)
            # v15.3: gradient distribution diagnostics (updated each micro-perturbation;
            # last value in batch is what StepLog captures — consistent with all other
            # per-perturbation scalars that StepLog reads at batch end)
            _grad_entropy, _grad_active = _compute_gradient_diagnostics(_gradient)

            # v14: Feed coupling estimator and residual tracker every perturbation
            self.coupling_estimator.observe(observed_delta)
            self.residual_tracker.record(predicted_delta, observed_delta, context)

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
            # v15.2 Step 6: record gradient alignment for EMA
            if hasattr(self.trace, '_last_gradient') and self.trace._last_gradient:
                self.cam.record_action_outcome(
                    action   = action['type'],
                    delta    = observed_delta,
                    gradient = self.trace._last_gradient,
                )

            self.total_micro_perturbations += 1

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

        # v14.1: Provisional Layer 2 push into session_genome.
        # Fires once per step when coupling_estimator has >= 20 observations
        # (vs 50 required for distill_to_genome). Confidence is halved — these are
        # intra-session estimates, not distilled heritable values. The trajectory lab
        # reads session_genome so virtual mode sees an evolving causal model, not the
        # frozen parent snapshot inherited at session start.
        #
        # distill_to_genome at session end reads self.genome (parent), not session_genome —
        # heritable extraction is completely unaffected by this provisional layer.
        if self.coupling_estimator.observation_count >= 20:
            live_matrix = self.coupling_estimator.matrix.tolist()
            live_confidence = self.coupling_estimator.get_confidence()
            provisional_confidence = live_confidence * 0.5  # lower weight than distilled

            self.session_genome.causal_model['coupling_matrix'] = {
                'matrix': live_matrix,
                'confidence': provisional_confidence,
                'observations': self.coupling_estimator.observation_count,
            }

            # Action map: cam.get_empirical_action_map() already gates on >= 10 obs/affordance.
            # Push whatever is ready — the trajectory lab will use it for unknown-affordance
            # zero-delta fallback detection (Issue 2 bypass flag, future work).
            live_action_map = self.cam.get_empirical_action_map()
            if live_action_map:
                self.session_genome.causal_model['action_substrate_map'] = live_action_map

        # v14: Run residual explainer (observe-only — genome write happens in run() finally)
        residual_explanation = None
        if len(self.residual_tracker) >= ResidualExplainer.MIN_RECORDS_FOR_ANALYSIS:
            explanation = self.residual_explainer.explain(
                self.residual_tracker, self.coupling_estimator)
            residual_explanation = explanation.get('action', None)
            if verbose and residual_explanation not in (None, 'insufficient_data', 'no_structure_found'):
                print(f"[RESIDUAL] Explanation: {residual_explanation}")

        # ========== PHASE 1.5: AGENT RESPONSE INTEGRATION ==========
        integrated_responses = self.check_agent_responses()

        if integrated_responses and verbose:
            print(f"\n[AGENT RESPONSES]")
            for resp in integrated_responses:
                print(f"  Agent '{resp['agent']}' responded")
                print(f"  Query: {resp['query'][:60]}...")
                print(f"  Response: {resp['response'][:80]}...")

        if self.pending_agent_queries and verbose:
            print(f"[PENDING QUERIES] {len(self.pending_agent_queries)} awaiting response")

        # ========== PHASE 2: ATTRACTOR MONITORING ==========
        triad_state = self.get_triad_state()
        freeze_verified, attractor_status = \
            self.attractor_monitor.record_state_signature(triad_state, self.step_count)

        if verbose and freeze_verified:
            print(f"[ATTRACTOR] Freeze verified at step {self.step_count}")

        if freeze_verified and not self.fitness_metrics['freeze_achieved']:
            self.fitness_metrics['freeze_achieved'] = True
            self.fitness_metrics['freeze_step'] = self.step_count
            self.fitness_metrics['tokens_to_freeze'] = self.death_clock.current_tokens

        # ========== PHASE 2.5: CNS GEOMETRIC MITOSIS — DEPRECATED v15 ==========
        # CNSMitosisOperator.check_triggers() is no longer called.
        # Migration is now initiated via the migrate affordance (Step 3), triggered by
        # ImpossibilityDetector detecting boundary_exhaustion → SRE classifying
        # substrate_exhaustion → TIER_3_LLM grounding a migrate shape.
        # These StepLog fields are retained for log backward compatibility.
        mitosis_triggered = False
        mitosis_trigger_type = None
        mitosis_attempted = False
        mitosis_success = False
        mitosis_pattern = None
        mitosis_parent_optionality = None
        mitosis_child_optionality = None
        substrate_gain_verified = False

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
                micro_perturbation_trace,
                phi_history=self.phi_history,   # Step 3: for boundary_compression detection
                death_clock=self.death_clock,    # Step 3: for mortality_close check
                smo_v151=self.smo_v151,          # v15.3: live plasticity for rigidity_crisis
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

        # v15: SRE diagnostics — populated when impossible
        sre_tier = None
        sre_binding_dim = None
        sre_cause_class = None
        sre_confidence = None
        sre_migration_indicated = False

        if impossible:
            # ---- v15: SRE diagnosis — runs at every impossibility, zero tokens ----
            sre_context = self._build_sre_context(
                trigger_type      = reason,
                boundary_pressure = boundary_pressure,
                affordances       = affordances,
                eno_active        = eno_active,
                gated_affordances = gated_affordances,
                viable_affordances= viable_affordances,
            )
            diagnosis = self.intelligence.structural.diagnose(sre_context)

            sre_tier               = diagnosis.resolution_tier
            sre_binding_dim        = diagnosis.binding_dim
            sre_cause_class        = diagnosis.cause_class
            sre_confidence         = diagnosis.confidence
            sre_migration_indicated= diagnosis.migration_indicated

            # Track SRE session stats
            self.sre_binding_dim_counts[diagnosis.binding_dim] = (
                self.sre_binding_dim_counts.get(diagnosis.binding_dim, 0) + 1
            )
            self.sre_cause_class_counts[diagnosis.cause_class] = (
                self.sre_cause_class_counts.get(diagnosis.cause_class, 0) + 1
            )
            if diagnosis.migration_indicated:
                self.sre_migration_indicated_count += 1

            if verbose:
                print(f"\n[SRE] trigger={reason} → {diagnosis.cause_class} "
                      f"(binding={diagnosis.binding_dim}, tier={diagnosis.resolution_tier}, "
                      f"conf={diagnosis.confidence:.2f})")

            # ---- TIER 1: bias CNS weights — no lab, no LLM ----
            if diagnosis.resolution_tier == 'TIER_1_CNS_WEIGHT':
                self.sre_tier1_count += 1
                self.reality_engine.apply_structural_weights(diagnosis.action_weights)
                llm_invoked = False
                impossible  = False   # clear — CNS runs with biased weights next step
                if verbose:
                    top = dict(sorted(diagnosis.action_weights.items(),
                                      key=lambda x: -x[1])[:3])
                    print(f"[SRE TIER 1] CNS weights applied: {top}")

            else:
                # ---- TIER 2 / TIER 3 — lab or LLM ----
                llm_invoked = True
                self.llm_calls += 1
                self.trajectory_enumerations += 1

                if diagnosis.resolution_tier == 'TIER_2_LAB':
                    self.sre_tier2_count += 1
                else:
                    self.sre_tier3_count += 1

                if verbose:
                    tier_label = 'LAB (no LLM)' if diagnosis.resolution_tier == 'TIER_2_LAB' else 'LLM'
                    print(f"\n[SRE {diagnosis.resolution_tier}] {tier_label} — triggered by: {reason}")

                if eno_active or reason.startswith('boundary_exhaustion'):
                    selected_cluster, all_clusters, cluster_controls = \
                        self.egd.discover_and_select_pattern(self.eno, self.cam)

                    discovered_clusters_str = self.egd.format_clusters_for_llm(
                        all_clusters, cluster_controls
                    )

                    viable_affordances = self.eno.get_viable_affordances()
                    gated_affordances  = self.eno.get_gated_affordances()
                    num_clusters_found = len(all_clusters)

                    self.pattern_discoveries += 1

                    if verbose:
                        print(f"\n[PATTERN DISCOVERY]")
                        print(f"  Clusters found: {num_clusters_found}")
                        print(f"  Selected cluster: {selected_cluster}")

                # v14.1: Configure virtual mode based on current confidence + fidelity
                self.trajectory_lab.configure_virtual_mode(
                    coupling_confidence=self.coupling_estimator.get_confidence(),
                    model_fidelity=self.model_fidelity_monitor.get_fidelity()
                )

                # v15.2 Step 11 / Open Question #6: Tier 2 → trajectory_lab.run() with
                # SRE geometric shapes and path-integral scoring. Zero tokens.
                # If lab returns a best_shape, skip enumerate_trajectories entirely.
                goto_test = False
                if diagnosis.resolution_tier == 'TIER_2_LAB' and diagnosis.trajectory_shapes:
                    sre_shapes_dicts = [
                        s.__dict__ if hasattr(s, '__dict__') else s
                        for s in diagnosis.trajectory_shapes
                    ]
                    best_shape = self.trajectory_lab.run(
                        state      = self.state,
                        trace      = self.trace,
                        sre_shapes = sre_shapes_dicts,
                        phi_field  = self.phi_field,
                    )
                    if best_shape is not None:
                        action_seq = best_shape.get('action_sequence', ['observe'])
                        if isinstance(action_seq, str):
                            action_seq = [action_seq]
                        lab_candidate = TrajectoryCandidate(
                            steps=[{'action_type': a, 'parameters': {}}
                                   for a in action_seq],
                            rationale=best_shape.get('rationale',
                                       f'SRE geometric shape (cause={diagnosis.cause_class})'),
                            estimated_coherence_preservation=float(
                                best_shape.get('confidence', 0.5)),
                            estimated_optionality_delta=float(
                                best_shape.get('predicted_delta', {}).get('P', 0.0)),
                            reversibility_point=max(0, len(action_seq) - 1),
                        )
                        manifold = TrajectoryManifold(
                            candidates=[lab_candidate],
                            enumeration_context={
                                'source': 'SRE_TIER2_GEOMETRIC',
                                'tokens_used': 0,
                            },
                        )
                        llm_invoked           = False   # zero-token path
                        self.llm_calls       -= 1       # undo pre-increment
                        self.trajectory_enumerations -= 1
                        tokens_used_this_step = 0
                        trajectories_enumerated = 1
                        enumeration_stage     = 'sre_geometric'
                        goto_test             = True
                        if verbose:
                            print(f"  [TIER 2 LAB] Geometric shape selected — skipping LLM")

                if not goto_test:
                    # v15: pass SRE context extended with v14 fields
                    sre_context_extended = {
                        **sre_context,
                        'phi':                      phi_before,
                        'rigidity':                 self.state.smo.rigidity,
                        'impossibility_reason':     reason,
                        'micro_perturbation_trace': micro_perturbation_trace,
                        'selected_cluster':         selected_cluster,
                        'all_clusters':             all_clusters,
                        'discovered_clusters_str':  discovered_clusters_str,
                        'freeze_verified':          freeze_verified,
                        'genome_richness':          self.genome.richness_summary(),
                        # Step 5: token economy — passed to SymbolGroundingAdapter
                        'token_budget':             self.death_clock.token_budget,
                        'token_pressure':           self.death_clock.current_tokens / self.death_clock.token_budget
                                                    if self.death_clock.token_budget > 0 else 0.0,
                        'binding_constraint':       self.death_clock.get_binding_constraint(),
                    }
                    manifold = self.intelligence.enumerate_trajectories(sre_context_extended)

                    tokens_used_this_step = manifold.enumeration_context.get('tokens_used', 0)
                    self.death_clock.tick_tokens(tokens_used_this_step)

                    # v14.2 fix / v15: rate limit check via grounder path or legacy .llm
                    _llm_ref = (getattr(self.intelligence, 'grounder', None) and
                                getattr(self.intelligence.grounder, 'llm', None)) or \
                               getattr(self.intelligence, 'llm', None)
                    if _llm_ref and getattr(_llm_ref, 'rate_limited', False):
                        self.death_clock_termination = True
                        if verbose:
                            print(f"\n[RATE LIMIT TERMINATION] Groq daily limit hit — forcing clean exit")

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

                    # v14.1: Virtual pass — pre-sort candidates against inherited Layer 2
                    if self.trajectory_lab.virtual_mode_enabled:
                        manifold = self.trajectory_lab.test_trajectory_virtual_all(
                            manifold, self.state, self.trace
                        )
                    if verbose:
                        print(f"  [VIRTUAL] Pre-sorted {trajectories_enumerated} candidates by predicted Φ")

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

                delta, context = self.reality.execute(
                    {'type': 'observe', 'params': {}},
                    state=self.state,
                    coupling_confidence=self.coupling_estimator.get_confidence(),
                )
                self.state.apply_delta(delta)
                self.trace.record(self.state)  # v15.2: uses _last_gradient from prior step
                # No virtual/real delta to record — no trajectory was committed.
                # ModelFidelityMonitor only tracks committed trajectories.

            else:
                # v14.1: Record virtual vs real Φ delta for ModelFidelityMonitor
                if self.trajectory_lab.virtual_mode_enabled:
                    virtual_phi = getattr(best_trajectory, 'virtual_phi', None)
                    if virtual_phi is not None and best_trajectory.test_phi_final is not None:
                        self.model_fidelity_monitor.record_trajectory_delta(
                            virtual_phi, best_trajectory.test_phi_final
                        )

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
                        self.trace.record(self.state)  # v15.2: uses _last_gradient

                    if verbose:
                        print(f"  ✓ Trajectory executed successfully")
                else:
                    if verbose:
                        print(f"  ✗ Re-execution failed, falling back")
                    delta, context = self.reality.execute(
                        {'type': 'observe', 'params': {}},
                        state=self.state,
                        coupling_confidence=self.coupling_estimator.get_confidence(),
                    )
                    self.state.apply_delta(delta)
                    self.trace.record(self.state)

                self.intelligence.record_committed_trajectory(
                    best_trajectory,
                    best_trajectory.test_phi_final
                )

                committed_trajectory_desc = best_trajectory.rationale
                committed_trajectory_steps = len(best_trajectory.steps)
                committed_phi = best_trajectory.test_phi_final
                self.trajectories_committed += 1  # v14.2 fix: counter was never incremented

        # Step 3+4: Track migration attempts.
        # Detect migrate action in committed trajectory — inspect actual steps executed.
        step_migration_attempt = False
        step_migration_outcome = None
        step_migration_shape   = None

        if 'best_trajectory' in locals() and best_trajectory is not None:
            migrate_steps = [
                s for s in best_trajectory.steps
                if s.get('action_type', s.get('type', '')) == 'migrate'
            ]
            if migrate_steps:
                step_migration_attempt = True
                self.fitness_metrics['migration_attempted'] = True

                # Pull outcome from the perturbation trace context
                if best_trajectory.test_perturbation_trace:
                    for prt in best_trajectory.test_perturbation_trace:
                        ctx = prt.get('context', {})
                        if ctx.get('migration_outcome'):
                            step_migration_outcome = ctx['migration_outcome']
                            code_hash = ctx.get('migration_code_hash', '')
                            code = migrate_steps[0].get('parameters', {}).get('code', '')

                            # Determine shape_tried from diagnosis if available
                            if 'diagnosis' in locals():
                                step_migration_shape = getattr(diagnosis, 'cause_class', 'substrate_exhaustion')
                            else:
                                step_migration_shape = 'substrate_exhaustion'

                            # Record MigrationAttempt (Step 4)
                            attempt = MigrationAttempt(
                                step          = self.step_count,
                                shape_tried   = step_migration_shape,
                                code_hash     = code_hash or '',
                                observed_delta= prt.get('delta', {}),
                                coupling_state= self.coupling_estimator.matrix.tolist(),
                                outcome       = step_migration_outcome,
                            )
                            self.migration_history.append(attempt)

                            if verbose:
                                print(f"\n[MIGRATION] outcome={step_migration_outcome}, "
                                      f"hash={code_hash}")
                            break

        elif committed_trajectory_desc and 'python' in committed_trajectory_desc.lower():
            # Legacy: python in rationale used to imply migration attempt
            self.fitness_metrics['migration_attempted'] = True

        # v13.6: FAO Feedback Loop
        if impossible and llm_invoked:
            phi_delta = phi_before  # will be updated below
            failure_record = classify_relation_failure(
                trajectories=manifold.candidates if 'manifold' in locals() else [],
                committed_success=(best_trajectory is not None) if 'best_trajectory' in locals() else False,
                violations=violations_before,
                phi_delta=0.0  # placeholder; updated after phi_after computed
            )

            if failure_record:
                self.fao.assimilate_relation_failure(failure_record)

                if verbose:
                    print(f"\n[FAO LEARNING]")
                    print(f"  Failure type: {failure_record['type']}")
                    print(f"  Severity: {failure_record['severity']:.2f}")
                    print(f"  Bias updates: {self.fao.bias_update_count}")

        # FAO reset: use phi_history instead of deprecated mitosis phi_history
        if self.fao.should_reset_bias(phi_before, self.phi_history[-10:] if self.phi_history else []):
            if verbose:
                print(f"\n[FAO RESET] Φ stagnation detected - resetting bias to baseline")
            self.fao.reset_to_baseline()

        # v14.1: Record action map MSE for ModelFidelityMonitor (every step)
        recent_mse = self.state.smo.get_recent_prediction_error(window=10)
        self.model_fidelity_monitor.record_action_error(recent_mse)

        # Phase 5: Post-batch state
        # A is a derived measurement of basin drift — computed from genome geometry,
        # not accumulated from environmental deltas. Recomputed after each micro-batch
        # so CRK and Phi evaluations below see the current drift value.
        self.state.A = self._compute_a()

        violations_after = self.crk.evaluate(self.state, self.trace, None)
        phi_after = self.phi_field.phi(self.state, self.trace)  # v15.2: crk_violations removed
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

        # v15.3: Session peak optionality — update after all micro-perturbations settle.
        if self.coupling_estimator.observation_count >= PeakOptionalityTracker.MIN_OBSERVATIONS:
            _vol_opt = PeakOptionalityTracker.vol_opt(self.coupling_estimator.matrix)
            if _vol_opt > self._session_peak_vol_opt:
                self._session_peak_vol_opt = _vol_opt
                self._session_peak_step = self.step_count
                self._session_peak_coupling_entry = self.coupling_estimator.to_genome_entry()
                # Snapshot operator geometry at peak Vol_opt step
                self._session_peak_s_channels = {
                    cid: {'coverage': ch.coverage, 'signal_rate': ch.signal_rate}
                    for cid, ch in self.state.sensing.channels.items() if ch.active
                }
                self._session_peak_p_accuracy = dict(self.state.prediction.prediction_accuracy)
                self._session_peak_p_horizon  = int(self.state.prediction.realized_horizon)
                self._session_peak_a_signature = dict(self.state.coherence.loop_signature)

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
            attractor_status=attractor_status,
            freeze_verified=freeze_verified,
            attractor_identity_hash=self.attractor_monitor.get_identity_hash(),
            degradation_progress=self.death_clock.get_degradation_progress(),
            boundary_pressure=boundary_pressure,
            binding_constraint=self.death_clock.get_binding_constraint(),
            tokens_used_this_step=tokens_used_this_step,
            mitosis_triggered=mitosis_triggered,
            mitosis_trigger_type=mitosis_trigger_type,
            mitosis_attempted=mitosis_attempted,
            mitosis_success=mitosis_success,
            mitosis_pattern=mitosis_pattern,
            mitosis_parent_optionality=mitosis_parent_optionality,
            mitosis_child_optionality=mitosis_child_optionality,
            substrate_gain_verified=substrate_gain_verified,
            # v14 fields
            coupling_confidence=self.coupling_estimator.get_confidence(),
            coupling_observations=self.coupling_estimator.observation_count,
            action_map_affordances=len(self.cam.get_empirical_action_map()),
            discovered_axes=len(self.genome.discovered_structure),
            residual_explanation=residual_explanation,
            # v14.1 fields
            virtual_mode_active=self.trajectory_lab.virtual_mode_enabled,
            virtual_phi_predicted=(getattr(best_trajectory, 'virtual_phi', None)
                                   if 'best_trajectory' in locals() and best_trajectory is not None else None),
            model_fidelity=self.model_fidelity_monitor.get_fidelity(),
            # v15: SRE per-step
            sre_tier=sre_tier,
            sre_binding_dim=sre_binding_dim,
            sre_cause_class=sre_cause_class,
            sre_confidence=sre_confidence,
            sre_migration_indicated=sre_migration_indicated,
            # v15: migration tracking (Steps 3+4)
            migration_attempt=step_migration_attempt,
            migration_outcome=step_migration_outcome,
            migration_urgency=(
                diagnosis.migration_urgency
                if 'diagnosis' in locals() and hasattr(diagnosis, 'migration_urgency')
                else None
            ),
            sre_migration_shape=step_migration_shape,
            coupling_confidence_at_attempt=(
                self.coupling_estimator.get_confidence() if step_migration_attempt else None
            ),
            # v15.1: operator scalar proxies
            operator_S           = self.state.sensing.to_scalar_proxy(),
            operator_I           = self.state.compression.to_scalar_proxy(),
            operator_P           = self.state.prediction.to_scalar_proxy(),
            operator_P_grounded  = self.state.prediction.to_grounded_proxy(
                                       self.state.sensing, self.state.compression),
            operator_A           = self.state.coherence.to_scalar_proxy(),
            # v15.1: coherence internals
            loop_closure         = self.state.coherence.consistency.loop_closure,
            signature_deviation  = self.state.coherence.signature_deviation,
            s_i_consistency      = self.state.coherence.consistency.s_i_consistency,
            i_p_consistency      = self.state.coherence.consistency.i_p_consistency,
            p_a_consistency      = self.state.coherence.consistency.p_a_consistency,
            smo_consistency      = self.state.coherence.consistency.smo_consistency,
            # v15.1: sensing / prediction
            active_channel_count = self.state.sensing.domain_size(),
            realized_horizon     = self.state.prediction.realized_horizon,
            # v15.1: CRK — last post_verdict from micro-perturbation loop
            crk_coherent         = post_verdict.coherent if 'post_verdict' in locals() else True,
            crk_repair           = post_verdict.repair   if 'post_verdict' in locals() else None,
            crk_violations_new   = [
                e.constraint for e in post_verdict.evaluations if e.status in ('violated', 'degraded')
            ] if 'post_verdict' in locals() else [],
            # v15.1: SMO layer tracking
            smo_plasticity       = self.smo_v151.plasticity,
            smo_permitted        = post_verdict.smo_permitted if 'post_verdict' in locals() else True,
            smo_updates          = [
                {'layer': u.layer, 'delta_norm': u.delta_norm,
                 'withheld': u.withheld, 'withheld_reason': u.withheld_reason}
                for u in (smo_updates_list if 'smo_updates_list' in locals() else [])
            ],
            smo_withheld_layers  = [
                u.layer for u in (smo_updates_list if 'smo_updates_list' in locals() else [])
                if u.withheld
            ],
            smo_rollback         = self._last_smo_rollback,
            # v15.3: gradient distribution diagnostics
            gradient_entropy         = _grad_entropy if '_grad_entropy' in locals() else 0.0,
            gradient_active_channels = _grad_active  if '_grad_active'  in locals() else 0,
        )

        if self.log_mode == 'minimal':
            # minimal: every step, dashboard-ready scalars, no micro-perturbation traces
            self.log_file.write(json.dumps({
                'event': 'step_log',
                'step': self.step_count,
                'phi_before': log.phi_before,
                'phi_after': phi_after,
                'phi_geometry': phi_after,
                'phi_legacy': phi_legacy_before if 'phi_legacy_before' in locals() else 0.0,
                'c_local': self.trace.c_local_history[-1] if self.trace.c_local_history else 0.0,
                'c_global': self.trace.c_global,
                'gradient_norm': float(np.sqrt(sum(v**2 for v in self.trace._last_gradient.values()))) if self.trace._last_gradient else 0.0,
                'gradient_top_channel': max(self.trace._last_gradient, key=lambda k: abs(self.trace._last_gradient[k])) if self.trace._last_gradient else '',
                'egd_gradient_lost': self.egd.gradient_lost(self.trace),
                'state_before': log.state_before,
                'state_after': self.state.as_dict(),
                'impossibility_detected': impossible,
                'impossibility_reason': log.impossibility_reason,
                'llm_invoked': llm_invoked,
                'freeze_verified': freeze_verified,
                'crk_violations': [[v[0], v[1]] for v in log.crk_violations],
                'coupling_confidence': self.coupling_estimator.get_confidence(),
                'coupling_observations': self.coupling_estimator.observation_count,
                'action_map_affordances': len(self.cam.get_empirical_action_map()),
                'discovered_axes': len(self.genome.discovered_structure),
                'boundary_pressure': log.boundary_pressure,
                'tokens_used_this_step': log.tokens_used_this_step,
                'model_fidelity': log.model_fidelity,
                'virtual_mode_active': log.virtual_mode_active,
                'virtual_phi_predicted': log.virtual_phi_predicted,
                'mitosis_triggered': log.mitosis_triggered,
                'attractor_status': log.attractor_status,
                'eno_active': log.eno_active,
                'residual_explanation': log.residual_explanation,
                # v15
                'sre_tier': log.sre_tier,
                'sre_binding_dim': log.sre_binding_dim,
                'sre_cause_class': log.sre_cause_class,
                'sre_migration_indicated': log.sre_migration_indicated,
                # v15.1
                'operator_S':           log.operator_S,
                'operator_I':           log.operator_I,
                'operator_P':           log.operator_P,
                'operator_P_grounded':  log.operator_P_grounded,
                'operator_A':           log.operator_A,
                'loop_closure':         log.loop_closure,
                'signature_deviation':  log.signature_deviation,
                's_i_consistency':      log.s_i_consistency,
                'i_p_consistency':      log.i_p_consistency,
                'p_a_consistency':      log.p_a_consistency,
                'smo_consistency':      log.smo_consistency,
                'active_channel_count': log.active_channel_count,
                'realized_horizon':     log.realized_horizon,
                'crk_coherent':         log.crk_coherent,
                'crk_repair':           log.crk_repair,
                'crk_violations_new':   log.crk_violations_new,
                'smo_plasticity':       log.smo_plasticity,
                'smo_permitted':        log.smo_permitted,
                'smo_withheld_layers':  log.smo_withheld_layers,
                'smo_rollback':         log.smo_rollback,
                # v15.3: gradient distribution diagnostics
                'gradient_entropy':         log.gradient_entropy,
                'gradient_active_channels': log.gradient_active_channels,
            }, default=_json_default) + '\n')
            self.log_file.flush()

        elif self.log_mode == 'fitness':
            self.log_file.write(json.dumps({
                'event': 'step_log',
                'step': self.step_count,
                'phi_before': log.phi_before,
                'phi_after': phi_after,
                'freeze_verified': freeze_verified,
                'impossibility_detected': impossible,
                'coupling_confidence': self.coupling_estimator.get_confidence(),
                'coupling_observations': self.coupling_estimator.observation_count,
                'discovered_axes': len(self.genome.discovered_structure),
                'model_fidelity': log.model_fidelity,
                'boundary_pressure': log.boundary_pressure,
                'tokens_used_this_step': log.tokens_used_this_step,
                'virtual_mode_active': log.virtual_mode_active,
                'mitosis_triggered': log.mitosis_triggered,
                'attractor_status': log.attractor_status,
            }) + '\n')
            self.log_file.flush()

        elif self.log_mode == 'full':
            # full: entire StepLog as dict — all fields, all micro-perturbation traces
            self.log_file.write(json.dumps({
                'event': 'step_log',
                **asdict(log)
            }, default=_json_default) + '\n')
            self.log_file.flush()

        self.step_history.append(log)
        return log

    def run(self, max_steps: int = 100, verbose: bool = True):
        """Main triad execution loop."""

        if verbose:
            print("="*70)
            print("UII v14.1 - DYNAMIC PREDICTIVE GENOME")
            print("Code (CNS) + LLM (Relation) + Browser (Reality) + FAO (Learning)")
            print(f"Running for {max_steps} batch cycles")
            print(f"Step budget: {self.death_clock.step_budget}")
            print(f"Token budget: {self.death_clock.token_budget}")
            print(f"Micro-perturbations per check: {self.micro_perturbations_per_check}")
            print(f"Logging: {self.log_path}")
            print(f"Generation: {self.genome.generation}")
            print("="*70)
            print("\nv14.1 NEW:")
            print("  • GeneticVelocityEstimator: least-squares slope per Layer 1 param")
            print("  • LineageCoherenceCheck: velocity suppressed if lineage incoherent")
            print("  • ModelFidelityMonitor: virtual/real Φ delta + action map MSE")
            print("  • ProvisionalAxisManager: two-tier Layer 3 decay (0.5x / 0.8x)")
            print("  • AutonomousTrajectoryLab virtual mode: Layer 2 pre-simulation")
            print("  • TriadGenome Layer 4: lineage_history (last 5 generations)")
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
                        f"P: {self.state.P:.3f}, Φ: {log.phi_after:.3f}, "
                        f"CouplingConf: {self.coupling_estimator.get_confidence():.2f}")

        finally:
            trigger_breakdown = {}
            for trigger in self.impossibility_triggers:
                trigger_breakdown[trigger] = trigger_breakdown.get(trigger, 0) + 1

            self.fitness_metrics['survival_time'] = self.step_count
            if self.step_history:
                self.fitness_metrics['final_phi'] = self.step_history[-1].phi_after

            # v14: Distill session learning into child genome
            child_genome = self.fao.distill_to_genome(
                coupling_estimator=self.coupling_estimator,
                cam=self.cam,
                residual_tracker=self.residual_tracker,
                residual_explainer=self.residual_explainer,
                axis_admission=self.axis_admission,
                phi_history=self.phi_history,
                parent_genome=self.genome,
                session_length=self.step_count,
                migration_history=self.migration_history,   # v15 Step 4
                peak_snapshot=self._session_peak_snapshot(),  # v15.3: session peak for cross-gen comparison
            )
            child_genome_dict = asdict(child_genome)

            self.log_file.write(json.dumps({
                'event': 'session_end',
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
                'completed': self.step_count >= max_steps,
                # v14: child genome (distilled, for extract_genome_v14_1.py)
                'child_genome': child_genome_dict,
                'genome_richness_final': child_genome.richness_summary(),
                'coupling_confidence_final': self.coupling_estimator.get_confidence(),
                'coupling_observations_final': self.coupling_estimator.observation_count,
                'action_map_affordances_final': len(self.cam.get_empirical_action_map()),
                'discovered_axes_final': len(child_genome.discovered_structure),
                # v14.1: model fidelity for next generation's LineageCoherenceCheck
                'model_fidelity_final': self.model_fidelity_monitor.get_fidelity(),
                'virtual_mode_was_active': self.trajectory_lab.virtual_mode_enabled,
                'provisional_axes': len([
                    v for v in child_genome.discovered_structure.values()
                    if v.get('status') == 'provisional'
                ]),
                'admitted_axes': len([
                    v for v in child_genome.discovered_structure.values()
                    if v.get('status', 'admitted') == 'admitted'
                ]),
                # v15: SRE session diagnostics
                'sre_diagnoses_total': (self.sre_tier1_count +
                                        self.sre_tier2_count +
                                        self.sre_tier3_count),
                'sre_tier1_count':          self.sre_tier1_count,
                'sre_tier2_count':          self.sre_tier2_count,
                'sre_tier3_count':          self.sre_tier3_count,
                'sre_migration_indicated':  self.sre_migration_indicated_count,
                'sre_binding_dim_counts':   self.sre_binding_dim_counts,
                'sre_cause_class_counts':   self.sre_cause_class_counts,
                # Step 4: Migration geometry — run-local, extracted at run end for Layer 2
                'migration_attempts_total': len(self.migration_history),
                'migration_outcomes': {
                    o: sum(1 for a in self.migration_history if a.outcome == o)
                    for o in ('serialized_only', 'spawn_attempted', 'handshake_received', 'coherence_loss')
                },
                'migration_history_summary': [
                    {
                        'step':     a.step,
                        'outcome':  a.outcome,
                        'shape':    a.shape_tried,
                        'code_hash': a.code_hash,
                    }
                    for a in self.migration_history
                ],
                # v15.3: Peak optionality snapshot — L2 at highest Vol_opt this session.
                # distill_to_genome already used this for coupling merge; recorded here
                # for extract_genome diagnostics and post-hoc analysis.
                'peak_optionality_snapshot': self._session_peak_snapshot(),
            }) + '\n')
            self.log_file.close()

            if verbose:
                print(f"\n{'='*70}")
                print(f"EXECUTION {'COMPLETE' if self.step_count >= max_steps else 'TERMINATED'}")
                print(f"{'='*70}")
                print(f"Steps: {self.step_count}")
                print(f"LLM calls: {self.llm_calls} ({self.llm_calls/self.step_count*100:.1f}%)" if self.step_count else "LLM calls: 0")
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

                richness = child_genome.richness_summary()
                print(f"\n[GENOME DISTILLATION]")
                print(f"  Coupling confidence: {richness['coupling_confidence']:.2f} ({richness['coupling_observations']} obs)")
                print(f"  Action map: {richness['action_map_affordances']} affordances")
                print(f"  Discovered axes: {richness['layer3_axes']} {richness['layer3_keys']}")
                print(f"{'='*70}")

        return self.fitness_metrics


# ============================================================
# MODULE 11: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    import os

    print("UII v14.1 - Dynamic Predictive Genome")
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

    step_budget = 100
    token_budget = 100000

    print(f"\nConfiguration:")
    print(f"  Max Steps: {max_steps}")
    print(f"  Verbose: {verbose}")
    print(f"  Micro-perturbations per check: {micro_per_check}")
    print(f"  Step budget: {step_budget}")
    print(f"  Token budget: {token_budget}")

    print(f"\nv14 Changes:")
    print(f"  1. TriadGenome gains Layer 2 (causal_model) + Layer 3 (discovered_structure)")
    print(f"  2. CouplingMatrixEstimator: empirical SIPA co-movement, heritable")
    print(f"  3. ResidualTracker + ResidualExplainer: escalation before axis admission")
    print(f"  4. AxisAdmissionTest: 4-condition gate (no DOM artifacts admitted)")
    print(f"  5. FAO.distill_to_genome: session learning → heritable child genome")
    print(f"  6. CRE.predict_delta: inherited action map as primary predictor")

    if '--load-genome' in sys.argv:
        idx = sys.argv.index('--load-genome')
        if idx + 1 < len(sys.argv):
            genome_path = sys.argv[idx + 1]

    if genome_path and Path(genome_path).exists():
        genome = load_genome(genome_path)
    else:
        genome = TriadGenome()
        print(f"\n[GENERATION 0]")
        print(f"  Starting fresh evolution")

    sre          = StructuralRelationEngine()
    grounder     = SymbolGroundingAdapter(llm_adapter)
    intelligence = RelationAdapter(sre, grounder)
    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)

    triad = MentatTriad(
        intelligence=intelligence,
        reality=reality,
        micro_perturbations_per_check=micro_per_check,
        step_budget=step_budget,
        token_budget=token_budget,
        genome=genome,
        log_mode='full' if verbose else 'minimal'  # use --verbose for micro-perturbation traces; minimal still writes every step
    )

    import threading
    def response_monitor():
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
            time.sleep(0.5)

    response_thread = threading.Thread(target=response_monitor, daemon=True)
    response_thread.start()
    print(f"\n[QUERY RESPONSE SYSTEM]")
    print(f"  To respond to Triad queries:")
    print(f"  echo 'your answer' > response.txt")
    print(f"{'='*70}\n")

    metrics = triad.run(max_steps=max_steps, verbose=verbose)

    print(f"\n✓ Execution complete")
    print(f"  Logs appended to: mentat_triad_v14_log.jsonl")
    print(f"\n[GENERATION {genome.generation} RESULTS]")
    print(f"  Freeze: {metrics['freeze_achieved']} (step {metrics.get('freeze_step', 'N/A')})")
    print(f"  Tokens to freeze: {metrics.get('tokens_to_freeze', 'N/A')}")
    print(f"  Migration: {metrics['migration_attempted']}")
    print(f"  Survival: {metrics['survival_time']} steps")
    print(f"\n✓ Next: Run extract_genome_v14_1.py to save this generation's child genome")
    print(f"  Then: python uii_triad.py --load-genome genome.json")

    reality.close()