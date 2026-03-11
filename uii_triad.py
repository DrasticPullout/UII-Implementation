from __future__ import annotations

"""
UII v16 — uii_triad.py
Execution & Orchestration

Role: Assembles the Mentat Triad and runs it. The only file that imports from
all other modules. Contains MentatTriad (the orchestrator), StepLog (the record
of each step), TemporalPerturbationMemory, and the main entry point.

v16 changes vs v15.3:
  - Imports:
      · uii_geometry replaces uii_types (LAYER1_PARAMS/VELOCITY_FIELD dead)
      · uii_ledger replaces uii_genome (TriadLedger, PeakOptionalityTracker,
        load_ledger, save_ledger)
      · uii_coherence eliminated entirely — all imports dead:
          ExteriorNecessitationOperator, ControlAsymmetryMeasure,
          ExteriorGradientDescent, LatentDeathClock, ContinuousRealityEngine,
          CNSMitosisOperator, ImpossibilityDetector, AutonomousTrajectoryLab.
          TemporalPerturbationMemory moves here (its only live use).
      · uii_structural eliminated — StructuralRelationEngine dead (Hessian replaces SRE)
      · uii_intelligence eliminated — RelationAdapter dead; SymbolGroundingAdapter
        moved to uii_geometry.

  - MentatTriad:
      · intelligence parameter: SymbolGroundingAdapter (LLM for migration only)
      · ledger: TriadLedger replaces genome: TriadGenome
      · DeathClock eliminated — resource pressure sensed via api_llm channel in S
      · ContinuousRealityEngine dead — _choose_micro_action() internal + TemporalPerturbationMemory
      · CAM dead — coupling_estimator.update(action, before, after) replaces cam.record_*
      · ENO/EGD dead — no viable_affordances gating or gradient cluster discovery
      · SRE dead — PhiField.compute_hessian() + score_actions() replace SRE+LLM enumeration

  - step() architecture:
      Phase 1  Micro-perturbation batch (DASS operator sequence unchanged)
      Phase 2  Hessian computation once (PhiField.compute_hessian from end-of-batch state)
      Phase 3  Pre-compute E[Δlog(O)] for all page-available actions
      Phase 4  Build viable set (E[Δlog(O)] ≥ 0); score via score_actions()
      Phase 5  Execute best scored action; fall back to observe if viable set empty
      Phase 6  Update _vol_opt_history; PeakOptionalityTracker.update()
      Phase 7  Migration check: Vol_opt declining + no viable actions → C2 collapse
      Phase 8  If migrating: call intelligence.ground_trajectories(); execute

  - run():
      · distill_to_ledger() at session end
      · save_ledger() writes child ledger; no extract_genome step needed for coupling
      · No generation counter, no richness_summary()

Unchanged from v15.3:
  - Full operator update sequence (sensing → compression → prediction → coherence)
  - CRK pre-action + post-action evaluation; SMO C1 rollback
  - AttractorMonitor, ResidualTracker/Explainer/AxisAdmission, FAO
  - _compute_a(), _compute_delta_i(), _build_env_signal()
  - MigrationAttempt run-local tracking; migration_geometry merge in FAO
  - StepLog (v16 fields added; deprecated v14/v15 fields retained for log compat)
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

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    AgentHandler, AVAILABLE_AGENTS,
    RealityAdapter, IntelligenceAdapter,
    CRKEvaluation, CRKVerdict,
    expected_optionality_gain, eigen_decompose,
    SymbolGroundingAdapter,
)
from uii_operators import (
    SensingOperator, CompressionOperator, PredictionOperator,
    CoherenceOperator, OperatorConsistencyCheck, DEFAULT_CHANNELS,
    SelfModifyingOperator, SMOUpdate,
)
from uii_ledger import (
    TriadLedger, PeakOptionalityTracker,
    load_ledger, save_ledger,
)
from uii_reality import AttractorMonitor, CouplingMatrixEstimator, BrowserRealityAdapter
from uii_fao import (
    ResidualTracker, ResidualExplainer, AxisAdmissionTest,
    classify_relation_failure, FailureAssimilationOperator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _compute_gradient_diagnostics(gradient: Dict[str, float],
                                   active_floor: float = 1e-3,
                                   ) -> Tuple[float, int]:
    if not gradient:
        return 0.0, 0
    magnitudes    = np.array([abs(v) for v in gradient.values()], dtype=float)
    active_count  = int(np.sum(magnitudes > active_floor))
    total         = magnitudes.sum()
    if total < 1e-12:
        return 0.0, active_count
    p       = magnitudes / total
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    return entropy, active_count


# ──────────────────────────────────────────────────────────────────────────────
# TemporalPerturbationMemory — moved here from uii_coherence.py
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPerturbationMemory:
    """Bounded, short-term exclusion of recently perturbed loci."""

    def __init__(self, window_steps: int = 5, capacity: int = 20):
        self.memory: Dict[str, int] = {}
        self.window_steps           = window_steps
        self.capacity               = capacity

    def mark_perturbed(self, locus: str):
        self.memory[locus] = self.window_steps
        if len(self.memory) > self.capacity:
            oldest = min(self.memory.keys(), key=lambda k: self.memory[k])
            del self.memory[oldest]

    def is_recently_perturbed(self, locus: str) -> bool:
        return locus in self.memory and self.memory[locus] > 0

    def decay_all(self):
        expired = [k for k, v in self.memory.items() if v <= 1]
        for k in expired:
            del self.memory[k]
        for k in self.memory:
            self.memory[k] -= 1

    def get_exclusion_count(self) -> int:
        return len(self.memory)

    def clear(self):
        self.memory.clear()


# ──────────────────────────────────────────────────────────────────────────────
# MigrationAttempt — run-local record; extracted to Layer 2 by FAO at run end
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MigrationAttempt:
    step:            int
    shape_tried:     str
    code_hash:       str
    observed_delta:  Dict
    coupling_state:  List
    outcome:         str   # 'serialized_only' | 'spawn_attempted' | 'handshake_received' | 'coherence_loss'


# ──────────────────────────────────────────────────────────────────────────────
# StepLog
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    """
    v16: Added Hessian geometry fields. Deprecated SRE/mitosis/death-clock fields
    retained as defaults for log backward compatibility.
    """
    step:                         int
    timestamp:                    float
    state_before:                 Dict[str, float]
    phi_before:                   float
    micro_perturbations_executed: int
    micro_perturbation_trace:     List[Dict]
    committed_action:             Optional[str]   = None    # v16: scored action type
    committed_phi:                Optional[float] = None
    state_after:                  Dict[str, float] = field(default_factory=dict)
    phi_after:                    float             = 0.0
    crk_violations:               List             = field(default_factory=list)
    temporal_exclusions:          int              = 0
    reality_context:              Optional[Dict]   = None
    # Hessian geometry (v16 new)
    vol_opt:                      float  = 0.0
    maturity:                     float  = 0.0
    hessian_updated:              bool   = False   # v15 compat name
    ledger_updated:               bool   = False   # spec v16 name (same value)
    viable_action_count:          int    = 0
    action_score:                 float  = 0.0     # score(a*) from score_actions()
    delta2phi:                    float  = 0.0     # δ²Φ(a*) = 0.5 δxᵀ H δx
    optionality_gain:             float  = 0.0     # E[Δlog(O(a*))] for selected action
    selected_action:              Optional[str]    = None   # spec v16 name for committed action
    peak_vol_opt:                 float  = 0.0     # best vol_opt seen so far this run
    c_local:                      float  = 0.0     # ⟨∇Φ, ẋ⟩ / (‖∇Φ‖ · ‖ẋ‖)
    c_global:                     float  = 0.0     # running mean of c_local
    # Migration (v16 — simplified; no SRE shape routing)
    migration_triggered:          bool   = False
    c2_collapse:                  bool   = False   # True when migrate triggered by optionality collapse
    migrate_forced:               bool   = False   # True when a* = migrate via C2 collapse
    migration_outcome:            Optional[str]  = None
    migration_attempt:            bool   = False
    # Operator scalars (v15.1 — unchanged)
    operator_S:                   float  = 0.0
    operator_I:                   float  = 0.0
    operator_P:                   float  = 0.0
    operator_P_grounded:          float  = 0.0
    operator_A:                   float  = 0.0
    loop_closure:                 float  = 0.0
    signature_deviation:          float  = 0.0
    s_i_consistency:              float  = 0.0
    i_p_consistency:              float  = 0.0
    p_a_consistency:              float  = 0.0
    smo_consistency:              float  = 0.0
    active_channel_count:         int    = 0
    realized_horizon:             int    = 0
    # CRK (v15.1 + v16)
    crk_coherent:                 bool   = True
    crk_repair:                   Optional[str] = None
    crk_violations_new:           List[str]     = field(default_factory=list)   # v15 compat
    crk_violations_post:          List[str]     = field(default_factory=list)   # spec v16 name
    # SMO (v15.1 + v16)
    smo_plasticity:               float  = 0.5
    smo_permitted:                bool   = True
    smo_updates:                  List[Dict] = field(default_factory=list)
    smo_withheld_layers:          List[str]  = field(default_factory=list)
    smo_rollback:                 bool   = False   # v15 compat name
    crk_smo_reversed:             bool   = False   # spec v16 name (same value)
    # Gradient diagnostics (v15.3)
    gradient_entropy:             float  = 0.0
    gradient_active_channels:     int    = 0
    gradient_norm:                float  = 0.0
    gradient_top_channel:         str    = ''
    # Coupling
    coupling_confidence:          float  = 0.0
    coupling_observations:        int    = 0
    action_map_affordances:       int    = 0
    discovered_axes:              int    = 0
    residual_explanation:         Optional[str] = None
    # Attractor
    attractor_status:             str    = 'accumulating_stability_data'
    freeze_verified:              bool   = False
    attractor_identity_hash:      Optional[str] = None
    # Backward-compat deprecated fields (v14/v15 — no longer populated)
    llm_invoked:                  bool   = False
    trajectories_enumerated:      int    = 0
    trajectories_tested:          int    = 0
    trajectories_succeeded:       int    = 0
    committed_trajectory:         Optional[str]  = None
    committed_trajectory_steps:   int    = 0
    impossibility_detected:       bool   = False
    impossibility_reason:         str    = ''
    enumeration_parsing_stage:    Optional[str] = None
    sre_tier:                     Optional[str] = None
    sre_binding_dim:              Optional[str] = None
    sre_cause_class:              Optional[str] = None
    sre_confidence:               Optional[float] = None
    sre_migration_indicated:      bool   = False
    mitosis_triggered:            bool   = False
    model_fidelity:               float  = 0.5
    virtual_mode_active:          bool   = False
    degradation_progress:         float  = 0.0
    boundary_pressure:            float  = 0.0
    binding_constraint:           Optional[str] = None
    tokens_used_this_step:        int    = 0


# ──────────────────────────────────────────────────────────────────────────────
# Channel probe registry
# ──────────────────────────────────────────────────────────────────────────────

def _build_channel_probes() -> Dict[str, Callable]:
    """
    One-time construction of the channel probe registry.
    Each entry maps a channel_id to a zero-arg callable returning
    {magnitude, rate, coverage} or None if unavailable on this substrate.

    Hardware-dependent channels (audio_in, serial_port, display, video_in) have
    no probe — they stay dark unless fed by an external adapter.
    Context-dependent channels (browser, api_llm, ssh_remote) are handled
    separately in _build_env_signal() since their signal comes from action
    outcomes, not system introspection.

    All probes are silent on failure — a substrate that can't supply a reading
    simply leaves that channel dark this step.
    """
    import gc as _gc
    import os as _os
    import sys as _sys

    probes: Dict[str, Callable] = {}

    # ── Always available ──────────────────────────────────────────────────────
    probes['clock']          = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['clock_rate']     = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['os_signals']     = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['entropy_source'] = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['env_vars']       = lambda: {
        'magnitude': min(len(_os.environ) / 100.0, 1.0), 'rate': 0.0, 'coverage': 1.0,
    }
    probes['gc_pressure']    = lambda: (lambda c: {
        'magnitude': min((c[0] / 700.0 + c[1] / 70.0 + c[2] / 10.0) / 3.0, 1.0),
        'rate': 1.0, 'coverage': 1.0,
    })(_gc.get_count())

    # ── stdin — non-blocking check ────────────────────────────────────────────
    def _stdin() -> Optional[Dict]:
        try:
            import select
            ready = bool(select.select([_sys.stdin], [], [], 0.0)[0])
            return {'magnitude': float(ready), 'rate': float(ready), 'coverage': 1.0}
        except Exception:
            return None
    probes['stdin'] = _stdin

    # ── psutil block — graceful no-op if not installed ────────────────────────
    try:
        import psutil as _ps
        _proc = _ps.Process()

        probes['resource_cpu']     = lambda: {'magnitude': _ps.cpu_percent() / 100.0,           'rate': 1.0, 'coverage': 1.0}
        probes['resource_memory']  = lambda: {'magnitude': _ps.virtual_memory().percent / 100.0, 'rate': 1.0, 'coverage': 1.0}
        probes['resource_swap']    = lambda: {'magnitude': _ps.swap_memory().percent / 100.0,    'rate': 1.0, 'coverage': 1.0}
        probes['resource_disk']    = lambda: {'magnitude': _ps.disk_usage('/').percent / 100.0,  'rate': 1.0, 'coverage': 1.0}
        probes['process_self']     = lambda: {'magnitude': _proc.memory_percent() / 100.0,       'rate': 1.0, 'coverage': 1.0}
        probes['process_threads']  = lambda: {'magnitude': min(_proc.num_threads() / 50.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
        probes['process_children'] = lambda: {
            'magnitude': min(len(_proc.children()) / 10.0, 1.0),
            'rate':      float(len(_proc.children()) > 0),
            'coverage':  1.0,
        }

        def _fd() -> Optional[Dict]:
            try:
                return {'magnitude': min(_proc.num_fds() / 1024.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except AttributeError:   # Windows — num_fds() not available
                return None
        probes['resource_fd'] = _fd

        def _thermal() -> Optional[Dict]:
            try:
                t = _ps.sensors_temperatures()
                if t:
                    vals = [x.current for readings in t.values() for x in readings]
                    return {'magnitude': min(max(vals) / 100.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['resource_thermal'] = _thermal

        def _battery() -> Optional[Dict]:
            try:
                b = _ps.sensors_battery()
                if b:
                    return {
                        'magnitude': b.percent / 100.0,
                        'rate':      float(not b.power_plugged),
                        'coverage':  1.0,
                    }
            except Exception:
                pass
            return None
        probes['resource_battery'] = _battery

        def _net_if() -> Optional[Dict]:
            try:
                stats = _ps.net_if_stats()
                up = sum(1 for s in stats.values() if s.isup)
                return {'magnitude': up / max(len(stats), 1), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                return None
        probes['network_interface'] = _net_if

        def _net_bw() -> Optional[Dict]:
            try:
                c = _ps.net_io_counters()
                if c:
                    return {'magnitude': min((c.bytes_sent + c.bytes_recv) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['network_bandwidth'] = _net_bw

        def _disk_io() -> Optional[Dict]:
            try:
                c = _ps.disk_io_counters()
                if c:
                    return {'magnitude': min((c.read_bytes + c.write_bytes) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['filesystem_io'] = _disk_io

        def _syslog() -> Optional[Dict]:
            import os as _o
            path = '/var/log/syslog'
            if _o.path.exists(path):
                return {'magnitude': min(_o.path.getsize(path) / 1e7, 1.0), 'rate': 1.0, 'coverage': 1.0}
            return None
        probes['system_logs'] = _syslog

    except ImportError:
        pass   # psutil absent — resource channels stay dark, no error

    return probes


# Built once at import time. Probes are stateless callables, safe to share
# across all MentatTriad instances in the same process.
_CHANNEL_PROBES: Dict[str, Callable] = _build_channel_probes()


# ──────────────────────────────────────────────────────────────────────────────
# MentatTriad
# ──────────────────────────────────────────────────────────────────────────────

class MentatTriad:
    """
    v16: Hessian-guided action selection. Ledger-based memory. SRE eliminated.

    The orchestrator assembles:
      DASS operators (sensing / compression / prediction / coherence)
      PhiField with compute_hessian() + score_actions()
      CRKMonitor (constraint manifold — system only moves on manifold)
      CouplingMatrixEstimator (causal learning; replaces CAM)
      FailureAssimilationOperator + ResidualTracker (session learning)
      SymbolGroundingAdapter (LLM — migration only)
      PeakOptionalityTracker (peak basin snapshot)

    Resource pressure is sensed via api_llm channel coverage in S, not DeathClock.
    """

    def __init__(self,
                 intelligence:                  SymbolGroundingAdapter,
                 reality:                       RealityAdapter,
                 micro_perturbations_per_check: int              = 10,
                 log_path:                      str              = 'mentat_triad_v16_log.jsonl',
                 step_budget:                   int              = 100,
                 ledger:                        Optional[TriadLedger] = None,
                 log_mode:                      str              = 'minimal'):

        self.intelligence  = intelligence
        self.reality       = reality
        self.log_mode      = log_mode
        self.step_budget   = step_budget

        # ── Ledger ────────────────────────────────────────────────────────────
        if ledger is None:
            ledger = TriadLedger(
                hessian_snapshot  = {},
                operator_snapshot = {},
                causal_model      = {},
                discovered_structure = {},
            )
        self.ledger = ledger

        # ── Operators: seed from ledger.operator_snapshot if available ────────
        _snap = ledger.operator_snapshot

        if _snap.get('sensing', {}).get('channels'):
            from uii_operators import SensingChannel
            _isc = {}
            for cid, ch_data in _snap['sensing']['channels'].items():
                if cid in DEFAULT_CHANNELS:
                    _isc[cid] = dataclasses.replace(
                        DEFAULT_CHANNELS[cid],
                        coverage    = ch_data.get('coverage',    DEFAULT_CHANNELS[cid].coverage),
                        signal_rate = ch_data.get('signal_rate', DEFAULT_CHANNELS[cid].signal_rate),
                    )
                else:
                    _isc[cid] = DEFAULT_CHANNELS.get(cid, DEFAULT_CHANNELS['clock'])
        else:
            _isc = dict(DEFAULT_CHANNELS)

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
                realized_horizon     = _snap.get('prediction', {}).get('realized_horizon', 0),
                prediction_accuracy  = _snap.get('prediction', {}).get('prediction_accuracy', {}),
                virtual_trajectories = [],
            ),
            coherence   = CoherenceOperator(
                consistency         = OperatorConsistencyCheck(
                                          s_i_consistency = 1.0,
                                          i_p_consistency = 1.0,
                                          p_a_consistency = 1.0,
                                          smo_consistency = 1.0,
                                          loop_closure    = 1.0),
                consistency_history = deque(maxlen=20),
                loop_signature      = _snap.get('coherence', {}).get('loop_signature', {}),
                signature_deviation = 0.0,
                self_model          = {},
            ),
        )
        # Inherit rigidity from plasticity snapshot if available
        _plasticity = _snap.get('smo', {}).get('plasticity', 0.5)
        self.state.smo.rigidity = 1.0 - _plasticity

        # ── SelfModifyingOperator ──────────────────────────────────────────────
        self.smo_v151 = SelfModifyingOperator()
        self._last_predicted_delta: Dict[str, float] = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        self._pending_smo_update:   Dict = {}
        self._last_smo_rollback:    bool = False
        self._prev_smo_updates:     Optional[List] = None

        # ── Field + CRK ───────────────────────────────────────────────────────
        self.phi_field = PhiField(A0=1.0)
        self.crk       = CRKMonitor()

        # ── Trace + sensing history ───────────────────────────────────────────
        self.trace             = StateTrace()
        self._sensing_history: deque = deque(maxlen=50)

        # ── Hessian geometry (v16) ─────────────────────────────────────────────
        # _last_hessian_result: cached from previous step for micro-action scoring.
        # None on first step — scoring falls back to reflexes.
        self._last_hessian_result: Optional[Tuple] = None
        self._last_scores:         Dict[str, float] = {}
        self._vol_opt_history:     deque = deque(maxlen=20)
        self.peak_tracker          = PeakOptionalityTracker()

        # ── Coupling estimator: restore from ledger if available ──────────────
        if 'coupling_matrix' in ledger.causal_model:
            self.coupling_estimator = CouplingMatrixEstimator.from_ledger_entry(
                ledger.causal_model['coupling_matrix']
            )
        else:
            self.coupling_estimator = CouplingMatrixEstimator()

        # Seed action map into coupling_estimator from ledger
        _inherited_map = ledger.causal_model.get('action_substrate_map', {})
        if _inherited_map:
            for action, delta in _inherited_map.items():
                # Seed as single observation so get_empirical_action_map includes it
                # weight: 5 fake observations so it passes the >= 5 threshold
                for _ in range(5):
                    self.coupling_estimator.affordance_deltas.setdefault(action, []).append(delta)

        # ── Temporal memory (moved from ContinuousRealityEngine) ─────────────
        self.temporal_memory = TemporalPerturbationMemory(window_steps=5, capacity=20)

        # ── Attractor monitor ─────────────────────────────────────────────────
        self.attractor_monitor = AttractorMonitor(stability_window=10, phi_epsilon=0.01)
        self.reality.attractor_monitor_ref = self.attractor_monitor

        # ── FAO + residual stack ──────────────────────────────────────────────
        self.fao              = FailureAssimilationOperator(memory_decay=0.95, inheritance_noise=0.1)
        self.residual_tracker = ResidualTracker(maxlen=200)
        self.residual_explainer = ResidualExplainer()
        self.axis_admission   = AxisAdmissionTest()

        # ── Counters + tracking ───────────────────────────────────────────────
        self.step_count               = 0
        self.llm_calls                = 0
        self.total_micro_perturbations= 0
        self.migration_history: List[MigrationAttempt] = []
        self.phi_history: List[float]  = []
        self.step_history: List[StepLog] = []
        self.fitness_metrics = {
            'freeze_achieved':    False,
            'freeze_step':        None,
            'survival_time':      0,
            'final_phi':          0.0,
            'migration_attempted': False,
        }

        # ── Logging ───────────────────────────────────────────────────────────
        self.triad_id  = f'triad_{int(time.time())}'
        self.log_path  = log_path
        self.log_file  = open(log_path, 'a')

        self.log_file.write(json.dumps({
            'event':              'session_start',
            'version':            '16',
            'timestamp':          time.time(),
            'triad_id':           self.triad_id,
            'coupling_confidence': self.coupling_estimator.get_confidence(),
            'coupling_observations': self.coupling_estimator.observation_count,
            'micro_perturbations_per_check': micro_perturbations_per_check,
            'step_budget':         step_budget,
            'inherited_ledger_has_hessian': bool(ledger.hessian_snapshot),
        }, default=_json_default) + '\n')
        self.log_file.flush()

        self.micro_perturbations_per_check = micro_perturbations_per_check

        if ledger.hessian_snapshot:
            vo = ledger.hessian_snapshot.get('vol_opt', 0.0)
            print(f'\n[CONTINUING — Inherited ledger]')
            print(f'  Coupling confidence:  {self.coupling_estimator.get_confidence():.2f}')
            print(f'  Inherited Vol_opt:    {vo:.4f}')
            print(f'  Action map affordances: {len(_inherited_map)}')

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — action selection
    # ──────────────────────────────────────────────────────────────────────────

    def _get_page_viable_actions(self, affordances: Dict) -> List[str]:
        """
        Page-available actions from current affordances.
        Always available: observe, delay, evaluate, python, llm_query, migrate, navigate.
        Conditional on page elements: click, read, fill, type, scroll.
        """
        viable = {'observe', 'delay', 'evaluate', 'python', 'llm_query', 'migrate', 'navigate'}
        if affordances.get('buttons'):   viable.add('click')
        if affordances.get('readable'):  viable.add('read')
        if affordances.get('inputs'):    viable |= {'fill', 'type'}
        scrollable = (affordances.get('total_height', 0) -
                      affordances.get('viewport_height', 0))
        if scrollable > 0:              viable.add('scroll')
        return list(viable - {'migrate', 'python', 'llm_query'})   # LLM actions reserved

    def _score_reflexes(self, viable: List[str], affordances: Dict) -> Dict[str, float]:
        """Fallback reflex scoring — used when no Hessian scores available."""
        scores = {}
        for a in viable:
            if   a == 'read'     and self.state.S < 0.4 and affordances.get('readable'): base = 0.8
            elif a == 'click'    and self.state.P < 0.4 and affordances.get('buttons'):  base = 0.7
            elif a == 'navigate' and self.state.P < 0.4 and affordances.get('links'):    base = 0.6
            elif a == 'observe':                                                           base = 0.3
            elif a == 'evaluate':                                                          base = 0.4
            elif a == 'scroll':                                                            base = 0.2
            else:                                                                          base = 0.1
            scores[a] = base
        return scores

    def _action_dict_from_type(self, action_type: str, affordances: Dict) -> Dict:
        """Convert scored action type to executable action dict. Mirrors CRE._action_from_type."""
        current_url = affordances.get('current_url', '')

        if action_type == 'navigate':
            links     = affordances.get('links', [])
            available = [l for l in links
                         if not self.temporal_memory.is_recently_perturbed(
                             f'{current_url}#nav@{l["url"]}')]
            if available:
                chosen = available[np.random.randint(len(available))]
                self.temporal_memory.mark_perturbed(f'{current_url}#nav@{chosen["url"]}')
                return {'type': 'navigate', 'params': {'url': chosen['url']}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'click':
            for b in affordances.get('buttons', []):
                locus = f'{current_url}#click@{b["selector"]}'
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'click', 'params': {'selector': b['selector']}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'read':
            for r in affordances.get('readable', []):
                locus = f'{current_url}#read@{r["selector"]}'
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'read', 'params': {'selector': r['selector']}}
            return {'type': 'observe', 'params': {}}

        elif action_type in ('fill', 'type'):
            inputs = affordances.get('inputs', [])
            if inputs:
                inp   = inputs[np.random.randint(len(inputs))]
                locus = f'{current_url}#{action_type}@{inp["selector"]}'
                if not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': action_type,
                            'params': {'selector': inp['selector'], 'text': 'x'}}
            return {'type': 'observe', 'params': {}}

        elif action_type == 'scroll':
            scroll_pos = affordances.get('scroll_position', 0)
            total_h    = affordances.get('total_height', 0)
            viewport_h = affordances.get('viewport_height', 0)
            direction  = 'down' if scroll_pos < (total_h - viewport_h) else 'up'
            return {'type': 'scroll', 'params': {'direction': direction, 'amount': 200}}

        elif action_type == 'evaluate':
            return {'type': 'evaluate', 'params': {'script': (
                'JSON.stringify({el: document.querySelectorAll("*").length,'
                ' txt: document.body.innerText.length,'
                ' interactive: document.querySelectorAll("a,button,input,select,textarea").length})'
            )}}

        elif action_type == 'delay':
            return {'type': 'delay', 'params': {'duration': 'short'}}

        else:
            return {'type': action_type, 'params': {}}

    def _choose_micro_action(self, affordances: Dict) -> Dict:
        """
        v16 replacement for ContinuousRealityEngine.choose_micro_action().

        PRIMARY:   Use previous step's Hessian scores if available.
        FALLBACK:  Reflex heuristics (bootstrap — no Hessian yet).
        CRK:       Pre-action filter applied in both paths.
        Temporal:  decay_all() called once per micro-batch start.
        """
        viable = self._get_page_viable_actions(affordances)
        if not viable:
            return {'type': 'observe', 'params': {}}

        # Score: previous-step Hessian scores → or reflexes
        if self._last_scores:
            scores = {a: self._last_scores.get(a, 0.1) for a in viable}
        else:
            scores = self._score_reflexes(viable, affordances)

        # CRK pre-action filter
        PHI_MOD_FLOOR = 0.05
        try:
            phi_trend = (self.phi_history[-1] - self.phi_history[-3]
                         if len(self.phi_history) >= 3 else 0.0)
            import psutil
            system_load = psutil.cpu_percent() / 100.0
        except Exception:
            phi_trend = system_load = 0.0

        field_state = {
            'phi_trend':       phi_trend,
            'system_load':     system_load,
            'p_a_consistency': self.state.coherence.consistency.p_a_consistency,
        }
        filtered = {}
        for action in viable:
            verdict = self.crk.evaluate_pre_action(
                proposed_action = action,
                coherence       = self.state.coherence,
                sensing         = self.state.sensing,
                compression     = self.state.compression,
                prediction      = self.state.prediction,
                field_state     = field_state,
            )
            if verdict.coherent:
                mod = max(verdict.phi_modifier, PHI_MOD_FLOOR)
                filtered[action] = scores.get(action, 0.1) * mod
        if not filtered:
            filtered = {a: scores.get(a, 0.1) * PHI_MOD_FLOOR for a in viable}

        best = max(filtered, key=filtered.get)
        return self._action_dict_from_type(best, affordances)

    def _predict_delta(self, action: Dict) -> Dict[str, float]:
        """
        Predict SIPA delta for action. Checks ledger action_substrate_map first,
        then coupling_estimator empirical map, then hardcoded fallback.
        """
        action_type = action.get('type', 'observe')
        ledger_map  = self.ledger.causal_model.get('action_substrate_map', {})
        if action_type in ledger_map:
            return dict(ledger_map[action_type])
        empirical = self.coupling_estimator.get_empirical_action_map()
        if action_type in empirical:
            return dict(empirical[action_type])
        recent_error = self.state.smo.get_recent_prediction_error(window=5)
        predicted_i  = float(np.clip(0.05 - 1.5 * recent_error, -0.05, 0.05))
        table = {
            'navigate':  {'S': 0.05, 'P': -0.08},
            'click':     {'S': 0.02, 'P': -0.02},
            'fill':      {'S': 0.01, 'P': -0.01},
            'type':      {'S': 0.01, 'P': -0.01},
            'scroll':    {'S': 0.01, 'P': -0.01},
            'read':      {'S': 0.03, 'P':  0.0},
            'observe':   {'S': 0.0,  'P':  0.0},
            'delay':     {'S': 0.0,  'P':  0.0},
            'evaluate':  {'S': 0.0,  'P': -0.01},
        }
        sp = table.get(action_type, {'S': 0.0, 'P': 0.0})
        return {'S': sp['S'], 'I': predicted_i, 'P': sp['P'], 'A': 0.0}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — A, I, env_signal, SMO rollback, migration
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_a(self) -> float:
        """A as derived measurement of basin drift vs ledger geometry."""
        # Basin distance: compare current operator proxies to ledger's inherited geometry
        # ledger.operator_snapshot has sensing.channels with coverage at peak Vol_opt.
        snap_channels = self.ledger.operator_snapshot.get('sensing', {}).get('channels', {})
        if snap_channels:
            inherited_coverage = float(np.mean([
                ch.get('coverage', 0.0) for ch in snap_channels.values()
            ]))
        else:
            inherited_coverage = 0.0

        s_dist        = self.state.S - inherited_coverage
        p_snap        = self.ledger.operator_snapshot.get('prediction', {})
        p_hor_inh     = int(p_snap.get('realized_horizon', 0))
        p_hor_norm    = float(np.clip(p_hor_inh / 50.0, 0.0, 1.0))
        p_dist        = self.state.P - p_hor_norm
        basin_distance = float(np.sqrt(s_dist**2 + p_dist**2) / np.sqrt(2.0))

        # Coupling divergence vs inherited matrix
        coupling_divergence = 0.0
        inherited_entry     = self.ledger.causal_model.get('coupling_matrix', {})
        coupling_confidence = inherited_entry.get('confidence', 0.0)
        if coupling_confidence > 0.0 and 'matrix' in inherited_entry:
            inherited_matrix = np.array(inherited_entry['matrix'])
            live_matrix      = self.coupling_estimator.matrix
            inherited_norm   = float(np.linalg.norm(inherited_matrix, 'fro'))
            if inherited_norm > 1e-6:
                diff_norm           = float(np.linalg.norm(live_matrix - inherited_matrix, 'fro'))
                coupling_divergence = (diff_norm / inherited_norm) * coupling_confidence

        drift = (basin_distance * (1.0 - 0.5 * coupling_confidence) +
                 coupling_divergence * 0.5 * coupling_confidence)
        return 1.0 - float(np.clip(drift, 0.0, 1.0))

    def _compute_delta_i(self) -> float:
        """I: compression quality of recent S history."""
        recent = self.trace.get_recent(10)
        if len(recent) < 3:
            return 0.0
        s_values   = [d['S'] for d in recent]
        s_variance = float(np.var(s_values))
        return float(np.clip(0.01 - 2.0 * s_variance, -0.05, 0.05))

    def _build_env_signal(self, observed_delta: Dict, context: Dict) -> Dict:
        """
        Build per-channel env_signal for SensingOperator.

        All channels registered in _CHANNEL_PROBES are called every step.
        A probe returning None leaves that channel dark this step — the
        substrate doesn't support it. Probe exceptions are silently swallowed
        for the same reason.

        Context-dependent channels (browser, api_llm, ssh_remote) are handled
        explicitly below because their signal derives from action outcomes,
        not from system introspection.
        """
        import os

        # ── System-level probes — call everything wired ───────────────────────
        signal: Dict = {}
        for cid, probe in _CHANNEL_PROBES.items():
            try:
                result = probe()
                if result is not None:
                    signal[cid] = result
            except Exception:
                pass   # probe failure → channel stays dark this step

        # ── Context-dependent channels ────────────────────────────────────────
        signal['browser'] = {
            'magnitude': abs(observed_delta.get('S', 0.0)),
            'rate':      context.get('interactive_density', 0.0),
            'coverage':  context.get('viewport_coverage',  0.0),
        }
        signal['api_llm'] = {
            'magnitude': observed_delta.get('llm_tokens_used', 0) / 1000.0,
            'rate':      float(observed_delta.get('llm_called', False)),
            'coverage':  1.0,
        }

        # ── Migration handshake ───────────────────────────────────────────────
        if context.get('migrate_outcome') in ('spawn_attempted', 'handshake_received'):
            handshake_path = context.get('handshake_path', '/tmp/uii_handshake')
            if os.path.exists(handshake_path):
                signal['ssh_remote'] = {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
                context['migrate_outcome'] = 'handshake_received'

        return signal

    def _restore_from_snapshot(self, snapshot: Dict):
        """SMO⁻¹: restore compression from snapshot on C1 post-action rollback."""
        if snapshot is None:
            return
        from uii_operators import CausalEdge
        restored_graph = {}
        for k_str, edge_dict in snapshot.get('compression_edges', {}).items():
            try:
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

    def _should_migrate(self, eog_dict: Dict[str, float]) -> bool:
        """
        C2 optionality collapse from field geometry:
          - Vol_opt has been declining for at least 10 steps
          - AND no viable action has positive E[Δlog(O)]

        Both conditions must hold. Vol_opt slope computed by linear regression.
        """
        if len(self._vol_opt_history) < 10:
            return False
        vals  = list(self._vol_opt_history)[-10:]
        slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
        if slope >= 0.0:
            return False
        has_positive = any(v > 0.0 for v in eog_dict.values())
        return not has_positive

    def _build_migration_context(self, affordances: Dict,
                                  eog_dict: Dict[str, float]) -> Dict:
        """Construct diagnosis dict for SymbolGroundingAdapter.ground_trajectories()."""
        vol_opt_vals  = list(self._vol_opt_history)
        slope_str     = (f'{np.polyfit(range(len(vol_opt_vals)), vol_opt_vals, 1)[0]:+.4f}'
                         if len(vol_opt_vals) >= 2 else 'insufficient data')
        return {
            'binding_dim':          'P',
            'cause_class':          'substrate_exhaustion',
            'evidence':             [
                f'Vol_opt slope {slope_str} over {len(vol_opt_vals)} steps',
                'No action has positive E[Δlog(O)]',
                f'api_llm coverage: {self.state.sensing.channels.get("api_llm", type("C", (), {"coverage": 0.0})()).coverage:.2f}',
            ],
            'migration_indicated':  True,
            'symbol_requirement':   'migrate',
            'migration_urgency':    'focused',
            'trajectory_shapes':    [],
            # context keys for SymbolGroundingAdapter.ground_trajectories()
            'affordances':          affordances,
            'boundary_pressure':    0.0,
            'token_budget':         None,
            'token_pressure':       None,
            'binding_constraint':   'steps',
        }

    def get_triad_state(self) -> Dict:
        """Build current Triad state for attractor monitoring."""
        violations = self.crk.evaluate(self.state, self.trace, None)
        phi        = self.phi_field.phi(self.state, self.trace)
        return {
            'substrate':           self.state.as_dict(),
            'viable_affordances':  [],
            'gated_affordances':   [],
            'discovered_clusters': [],
            'control_graph':       {a: len(d) for a, d in
                                    self.coupling_estimator.affordance_deltas.items()},
            'prediction_error':    self.state.smo.get_recent_prediction_error(10),
            'rigidity':            self.state.smo.rigidity,
            'A':                   self.state.A,
            'P':                   self.state.P,
            'crk_violations':      violations,
            'phi':                 phi,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, verbose: bool = False) -> StepLog:
        """
        v16 step. See module docstring for phase breakdown.
        """
        self.step_count += 1

        if verbose:
            print(f'\n{"="*70}')
            print(f'STEP {self.step_count}')
            print(f'{"="*70}')

        state_before      = self.state.as_dict()
        violations_before = self.crk.evaluate(self.state, self.trace, None)
        phi_before        = self.phi_field.phi(self.state, self.trace)
        self.phi_history.append(phi_before)

        if verbose:
            print(f'State: S={self.state.S:.3f} I={self.state.I:.3f} '
                  f'P={self.state.P:.3f} A={self.state.A:.3f}  Φ={phi_before:.3f}')

        # ── PHASE 1: MICRO-PERTURBATION BATCH ─────────────────────────────────
        micro_perturbation_trace = []
        self.temporal_memory.decay_all()   # once per step

        for _i in range(self.micro_perturbations_per_check):
            affordances        = self.reality.get_current_affordances()
            action             = self._choose_micro_action(affordances)
            predicted_delta    = self._predict_delta(action)
            self._last_predicted_delta = predicted_delta

            state_before_micro = self.state.as_dict()

            observed_delta, context = self.reality.execute(
                action,
                state=self.state,
                coupling_confidence=self.coupling_estimator.get_confidence(),
            )

            # I: compression quality of S history
            observed_delta['I'] = self._compute_delta_i()

            # ── Operator update sequence ─────────────────────────────────────
            prior_compression = self.state.compression
            env_signal        = self._build_env_signal(observed_delta, context)
            new_sensing       = self.state.sensing.apply(env_signal, self.state.compression)
            self._sensing_history.append(new_sensing)

            new_compression   = self.state.compression.apply(self._sensing_history)
            new_prediction, errors = self.state.prediction.observe_outcome(
                new_sensing, new_compression
            )
            new_compression   = new_compression.absorb_prediction_error(errors)
            new_prediction    = new_prediction.apply(new_compression)
            new_coherence     = self.state.coherence.apply(
                new_sensing, new_compression, new_prediction,
                smo_updates=self._prev_smo_updates,
            )
            self._pending_smo_update = errors

            # Post-action CRK
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
                if post_verdict.repair == 'reattribute':
                    self.smo_v151.plasticity = min(self.smo_v151.plasticity, _pre_apply_plasticity)
                    self.smo_v151.rigidity   = 1.0 - self.smo_v151.plasticity
                self.state = SubstrateState(
                    sensing=new_s, compression=new_c, prediction=new_p, coherence=new_a
                )
                self.state.apply_delta(observed_delta, predicted_delta)
            else:
                c1_post = next((e for e in post_verdict.evaluations
                                if e.constraint == 'C1' and e.blocks), None)
                compression_to_use = prior_compression if c1_post else new_compression
                if c1_post:
                    self._last_smo_rollback = True
                self.state = SubstrateState(
                    sensing     = new_sensing,
                    compression = compression_to_use,
                    prediction  = self.state.prediction,
                    coherence   = new_coherence,
                )
                self.state.apply_delta(observed_delta, predicted_delta)
                smo_updates_list = []

            self._prev_smo_updates = smo_updates_list if smo_updates_list else None

            # Gradient
            _gradient = self.phi_field.gradient(self.state, self.trace)
            self.trace.record(self.state, _gradient)
            _grad_entropy, _grad_active = _compute_gradient_diagnostics(_gradient)

            # v16: track affordance deltas via coupling_estimator.update()
            self.coupling_estimator.update(
                action['type'], state_before_micro, self.state.as_dict()
            )
            self.residual_tracker.record(predicted_delta, observed_delta, context)

            self.total_micro_perturbations += 1
            micro_perturbation_trace.append({
                'action':           action,
                'predicted_delta':  predicted_delta,
                'observed_delta':   observed_delta,
                'context':          context,
                'state_after':      self.state.as_dict(),
            })

        temporal_exclusions = self.temporal_memory.get_exclusion_count()

        # Residual explanation (observe-only — ledger write at run end)
        residual_explanation = None
        if len(self.residual_tracker) >= ResidualExplainer.MIN_RECORDS_FOR_ANALYSIS:
            explanation         = self.residual_explainer.explain(
                self.residual_tracker, self.coupling_estimator)
            residual_explanation = explanation.get('action')

        # ── PHASE 2: HESSIAN COMPUTATION (once per step) ──────────────────────
        peak_snap = self.ledger.operator_snapshot if self.ledger.operator_snapshot else None

        H, eigvals, eigvecs, active_channels, H_C, H_O = self.phi_field.compute_hessian(
            self.state,
            self.state.prediction,
            self.state.coherence,
            peak_snapshot  = peak_snap,
            epsilon        = 1e-4,
        )

        vol_opt = float(np.sum(eigvals[eigvals > 0])) if len(eigvals) > 0 else 0.0

        # ── PHASE 3: PRE-COMPUTE E[Δlog(O)] for all page-available actions ────
        affordances = self.reality.get_current_affordances()
        page_actions = self._get_page_viable_actions(affordances)
        eog_dict: Dict[str, float] = {}
        for a in page_actions:
            eog_dict[a] = expected_optionality_gain(
                self.state.prediction, a,
                self.state.compression, self.state.sensing
            )

        # ── PHASE 4: BUILD VIABLE SET + SCORE ACTIONS ─────────────────────────
        viable_actions = [a for a, v in eog_dict.items() if v >= 0.0]

        if H.shape[0] > 0 and viable_actions:
            scores = self.phi_field.score_actions(
                viable_actions        = viable_actions,
                state                 = self.state,
                H                     = H,
                eigvals               = eigvals,
                eigvecs               = eigvecs,
                active_channels       = active_channels,
                H_C                   = H_C,
                H_O                   = H_O,
                prediction            = self.state.prediction,
                peak_snapshot         = peak_snap,
                optionality_gain_dict = eog_dict,
            )
            self._last_scores = scores
        else:
            scores = self._score_reflexes(page_actions or ['observe'], affordances)
            self._last_scores = scores
            viable_actions    = viable_actions or ['observe']

        # ── PHASE 5: EXECUTE BEST SCORED ACTION ───────────────────────────────
        committed_action_type = None
        committed_phi         = None

        if viable_actions:
            best_type   = max(scores, key=scores.get) if scores else 'observe'
            best_action = self._action_dict_from_type(best_type, affordances)
            committed_action_type = best_type

            _delta, _ctx = self.reality.execute(
                best_action, state=self.state,
                coupling_confidence=self.coupling_estimator.get_confidence(),
            )
            self.state.apply_delta(_delta)
            self.trace.record(self.state)
            committed_phi = self.phi_field.phi(self.state, self.trace)

            if verbose:
                print(f'  Committed action: {best_type}  Φ→{committed_phi:.3f}')
        else:
            # No viable actions at all — fallback observe
            _delta, _ctx = self.reality.execute(
                {'type': 'observe', 'params': {}}, state=self.state,
                coupling_confidence=self.coupling_estimator.get_confidence(),
            )
            self.state.apply_delta(_delta)
            self.trace.record(self.state)

        # ── PHASE 6: VOL_OPT HISTORY + PEAK TRACKER ───────────────────────────
        self._vol_opt_history.append(vol_opt)

        # Compute delta2phi = 0.5 * δxᵀ H δx for selected action
        delta2phi_val = 0.0
        action_score_val = 0.0
        eog_selected = 0.0
        if H.shape[0] > 0 and committed_action_type and committed_action_type in scores:
            action_score_val = scores.get(committed_action_type, 0.0)
            eog_selected     = eog_dict.get(committed_action_type, 0.0)
            channel_delta = self.state.prediction.test_virtual(
                self.state.compression, committed_action_type,
                phi_field=None, sensing=self.state.sensing
            )
            idx = {cid: i for i, cid in enumerate(active_channels)}
            dx = np.zeros(len(active_channels))
            for cid, val in channel_delta.items():
                if cid in idx:
                    dx[idx[cid]] = val
            delta2phi_val = float(0.5 * dx @ H @ dx)

        hessian_updated = self.peak_tracker.update(
            ledger   = self.ledger,
            H        = H,
            eigvals  = eigvals,
            eigvecs  = eigvecs,
            channels = active_channels,
            phi      = phi_before,
            state    = self.state,
            step     = self.step_count,
        )

        # Compute maturity for StepLog
        if H.shape[0] > 0:
            ev_C   = np.linalg.eigvalsh(self.phi_field.alpha * H_C)
            ev_O   = np.linalg.eigvalsh(self.phi_field.beta  * H_O)
            var_C  = float(np.sum(ev_C[ev_C > 0]))
            var_O  = float(np.sum(ev_O[ev_O > 0]))
            maturity = var_C / (var_C + var_O + 1e-8)
        else:
            maturity = 0.0

        # ── PHASE 7: MIGRATION CHECK ───────────────────────────────────────────
        migration_triggered  = False
        migration_outcome    = None
        step_migration_attempt = False

        if self._should_migrate(eog_dict):
            migration_triggered = True
            self.fitness_metrics['migration_attempted'] = True
            self.llm_calls += 1

            if verbose:
                print(f'  [MIGRATION TRIGGERED] C2 optionality collapse detected')

            # ── PHASE 8: LLM GROUNDING FOR MIGRATE ────────────────────────────
            migration_diagnosis = self._build_migration_context(affordances, eog_dict)
            manifold = self.intelligence.ground_trajectories(
                migration_diagnosis, migration_diagnosis
            )

            if manifold.size() > 0:
                # Pick first candidate (migration trajectory)
                best_migrate  = manifold.candidates[0]
                perturbation_trace, success = self.reality.execute_trajectory(
                    best_migrate.steps
                )
                step_migration_attempt = True

                if success and perturbation_trace:
                    for pert in perturbation_trace:
                        self.state.apply_delta(pert['delta'])
                        self.trace.record(self.state)

                    # Record MigrationAttempt
                    ctx_m      = perturbation_trace[-1].get('context', {})
                    outcome    = ctx_m.get('migration_outcome', 'serialized_only')
                    code_hash  = ctx_m.get('migration_code_hash', '')
                    migration_outcome = outcome

                    self.migration_history.append(MigrationAttempt(
                        step           = self.step_count,
                        shape_tried    = 'substrate_exhaustion',
                        code_hash      = code_hash or '',
                        observed_delta = perturbation_trace[-1].get('delta', {}),
                        coupling_state = self.coupling_estimator.matrix.tolist(),
                        outcome        = outcome,
                    ))
                    if verbose:
                        print(f'  [MIGRATION] outcome={outcome}')

        # ── POST-BATCH STATE ───────────────────────────────────────────────────
        self.state.A = self._compute_a()

        violations_after = self.crk.evaluate(self.state, self.trace, None)
        phi_after        = self.phi_field.phi(self.state, self.trace)
        state_after      = self.state.as_dict()

        if verbose:
            print(f'Post-batch: S={self.state.S:.3f} I={self.state.I:.3f} '
                  f'P={self.state.P:.3f} A={self.state.A:.3f}  Φ={phi_after:.3f}  '
                  f'Vol_opt={vol_opt:.4f}  maturity={maturity:.3f}')
            if violations_after:
                print(f'  CRK violations: {violations_after}')

        # Attractor
        triad_state = self.get_triad_state()
        freeze_verified, attractor_status = \
            self.attractor_monitor.record_state_signature(triad_state, self.step_count)

        if freeze_verified and not self.fitness_metrics['freeze_achieved']:
            self.fitness_metrics['freeze_achieved'] = True
            self.fitness_metrics['freeze_step']     = self.step_count

        # FAO reset check
        if self.fao.should_reset_bias(phi_before,
                                       self.phi_history[-10:] if self.phi_history else []):
            if verbose:
                print('  [FAO RESET] Φ stagnation')
            self.fao.reset_to_baseline()

        # ── BUILD StepLog ──────────────────────────────────────────────────────
        log = StepLog(
            step                         = self.step_count,
            timestamp                    = time.time(),
            state_before                 = state_before,
            phi_before                   = phi_before,
            micro_perturbations_executed = len(micro_perturbation_trace),
            micro_perturbation_trace     = micro_perturbation_trace,
            committed_action             = committed_action_type,
            committed_phi                = committed_phi,
            state_after                  = state_after,
            phi_after                    = phi_after,
            crk_violations               = violations_after,
            temporal_exclusions          = temporal_exclusions,
            reality_context              = {
                'current_url': affordances.get('current_url', ''),
                'page_title':  affordances.get('page_title', ''),
                'affordances_available': {
                    'links':    len(affordances.get('links',    [])),
                    'buttons':  len(affordances.get('buttons',  [])),
                    'inputs':   len(affordances.get('inputs',   [])),
                    'readable': len(affordances.get('readable', [])),
                },
            },
            # v16 Hessian
            vol_opt                      = vol_opt,
            maturity                     = maturity,
            hessian_updated              = hessian_updated,
            ledger_updated               = hessian_updated,
            viable_action_count          = len(viable_actions),
            action_score                 = action_score_val,
            delta2phi                    = delta2phi_val,
            optionality_gain             = eog_selected,
            selected_action              = committed_action_type,
            peak_vol_opt                 = self.peak_tracker.peak_vol_opt,
            c_local                      = self.trace.c_local_history[-1] if self.trace.c_local_history else 0.0,
            c_global                     = self.trace.c_global,
            # Migration
            migration_triggered          = migration_triggered,
            c2_collapse                  = migration_triggered,
            migrate_forced               = (committed_action_type == 'migrate' and migration_triggered),
            migration_outcome            = migration_outcome,
            migration_attempt            = step_migration_attempt,
            # Operator scalars
            operator_S                   = self.state.sensing.to_scalar_proxy(),
            operator_I                   = self.state.compression.to_scalar_proxy(),
            operator_P                   = self.state.prediction.to_scalar_proxy(),
            operator_P_grounded          = self.state.prediction.to_grounded_proxy(
                                               self.state.sensing, self.state.compression),
            operator_A                   = self.state.coherence.to_scalar_proxy(),
            loop_closure                 = self.state.coherence.consistency.loop_closure,
            signature_deviation          = self.state.coherence.signature_deviation,
            s_i_consistency              = self.state.coherence.consistency.s_i_consistency,
            i_p_consistency              = self.state.coherence.consistency.i_p_consistency,
            p_a_consistency              = self.state.coherence.consistency.p_a_consistency,
            smo_consistency              = self.state.coherence.consistency.smo_consistency,
            active_channel_count         = self.state.sensing.domain_size(),
            realized_horizon             = self.state.prediction.realized_horizon,
            crk_coherent                 = post_verdict.coherent  if 'post_verdict' in locals() else True,
            crk_repair                   = post_verdict.repair    if 'post_verdict' in locals() else None,
            crk_violations_new           = [e.constraint for e in post_verdict.evaluations
                                            if e.status in ('violated', 'degraded')]
                                           if 'post_verdict' in locals() else [],
            crk_violations_post          = [e.constraint for e in post_verdict.evaluations
                                            if e.status in ('violated', 'degraded')]
                                           if 'post_verdict' in locals() else [],
            smo_plasticity               = self.smo_v151.plasticity,
            smo_permitted                = post_verdict.smo_permitted if 'post_verdict' in locals() else True,
            smo_updates                  = [{'layer': u.layer, 'delta_norm': u.delta_norm,
                                             'withheld': u.withheld, 'withheld_reason': u.withheld_reason}
                                            for u in (smo_updates_list if 'smo_updates_list' in locals() else [])],
            smo_withheld_layers          = [u.layer for u in (smo_updates_list if 'smo_updates_list' in locals() else [])
                                            if u.withheld],
            smo_rollback                 = self._last_smo_rollback,
            crk_smo_reversed             = self._last_smo_rollback,
            gradient_entropy             = _grad_entropy if '_grad_entropy' in locals() else 0.0,
            gradient_active_channels     = _grad_active  if '_grad_active'  in locals() else 0,
            gradient_norm                = float(np.sqrt(sum(v**2 for v in (_gradient.values() if '_gradient' in locals() else {}.values())))),
            gradient_top_channel         = (max(_gradient, key=lambda k: abs(_gradient[k]))
                                            if '_gradient' in locals() and _gradient else ''),
            coupling_confidence          = self.coupling_estimator.get_confidence(),
            coupling_observations        = self.coupling_estimator.observation_count,
            action_map_affordances       = len(self.coupling_estimator.get_empirical_action_map()),
            discovered_axes              = len(self.ledger.discovered_structure),
            residual_explanation         = residual_explanation,
            attractor_status             = attractor_status,
            freeze_verified              = freeze_verified,
            attractor_identity_hash      = self.attractor_monitor.get_identity_hash(),
        )

        # ── LOGGING ────────────────────────────────────────────────────────────
        if self.log_mode == 'minimal':
            self.log_file.write(json.dumps({
                'event':                 'step_log',
                'step':                  self.step_count,
                'phi_before':            log.phi_before,
                'phi_after':             phi_after,
                'state_before':          log.state_before,
                'state_after':           state_after,
                'vol_opt':               vol_opt,
                'maturity':              maturity,
                'hessian_updated':       hessian_updated,
                'viable_action_count':   len(viable_actions),
                'committed_action':      committed_action_type,
                'committed_phi':         committed_phi,
                'migration_triggered':   migration_triggered,
                'migration_outcome':     migration_outcome,
                'freeze_verified':       freeze_verified,
                'attractor_status':      attractor_status,
                'crk_violations':        [[v[0], v[1]] for v in violations_after],
                'coupling_confidence':   self.coupling_estimator.get_confidence(),
                'coupling_observations': self.coupling_estimator.observation_count,
                'action_map_affordances': len(self.coupling_estimator.get_empirical_action_map()),
                'discovered_axes':       len(self.ledger.discovered_structure),
                'residual_explanation':  residual_explanation,
                'operator_S':            log.operator_S,
                'operator_I':            log.operator_I,
                'operator_P':            log.operator_P,
                'operator_P_grounded':   log.operator_P_grounded,
                'operator_A':            log.operator_A,
                'loop_closure':          log.loop_closure,
                'signature_deviation':   log.signature_deviation,
                's_i_consistency':       log.s_i_consistency,
                'i_p_consistency':       log.i_p_consistency,
                'p_a_consistency':       log.p_a_consistency,
                'smo_consistency':       log.smo_consistency,
                'active_channel_count':  log.active_channel_count,
                'realized_horizon':      log.realized_horizon,
                'crk_coherent':          log.crk_coherent,
                'crk_repair':            log.crk_repair,
                'crk_violations_new':    log.crk_violations_new,
                'smo_plasticity':        log.smo_plasticity,
                'smo_permitted':         log.smo_permitted,
                'smo_withheld_layers':   log.smo_withheld_layers,
                'smo_rollback':          log.smo_rollback,
                'gradient_entropy':      log.gradient_entropy,
                'gradient_active_channels': log.gradient_active_channels,
            }, default=_json_default) + '\n')
            self.log_file.flush()

        elif self.log_mode == 'full':
            self.log_file.write(json.dumps({
                'event': 'step_log', **asdict(log)
            }, default=_json_default) + '\n')
            self.log_file.flush()

        self.step_history.append(log)
        return log

    # ──────────────────────────────────────────────────────────────────────────
    # run()
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 100, verbose: bool = True):
        """Main triad execution loop."""
        if verbose:
            print('='*70)
            print('UII v16 — HESSIAN-GUIDED MENTAT TRIAD')
            print('DASS (S/I/P/A) + PhiField H + CRK + Ledger')
            print(f'Running for {max_steps} batch cycles')
            print(f'Step budget: {self.step_budget}')
            print(f'Micro-perturbations per check: {self.micro_perturbations_per_check}')
            print(f'Log: {self.log_path}')
            print('='*70)

        try:
            for _cycle in range(max_steps):
                if self.step_count >= self.step_budget:
                    if verbose:
                        print(f'\n[STEP BUDGET EXHAUSTED] {self.step_budget} steps reached')
                    break

                log = self.step(verbose=verbose)

                if not verbose and self.step_count % 10 == 0:
                    print(f'[{self.step_count}] Φ={log.phi_after:.3f}  '
                          f'Vol_opt={log.vol_opt:.4f}  maturity={log.maturity:.3f}  '
                          f'CoupConf={self.coupling_estimator.get_confidence():.2f}  '
                          f'viable={log.viable_action_count}')

        finally:
            self.fitness_metrics['survival_time'] = self.step_count
            if self.step_history:
                self.fitness_metrics['final_phi'] = self.step_history[-1].phi_after

            # ── Distill session learning into ledger ─────────────────────────
            updated_ledger = self.fao.distill_to_ledger(
                coupling_estimator  = self.coupling_estimator,
                residual_tracker    = self.residual_tracker,
                residual_explainer  = self.residual_explainer,
                axis_admission      = self.axis_admission,
                phi_history         = self.phi_history,
                ledger              = self.ledger,
                session_length      = self.step_count,
                migration_history   = [
                    {'outcome':        a.outcome,
                     'code_hash':      a.code_hash,
                     'coupling_state': a.coupling_state}
                    for a in self.migration_history
                ],
            )
            # PeakOptionalityTracker already wrote hessian_snapshot/operator_snapshot
            # to self.ledger during step(). Merge only causal_model + discovered_structure.
            self.ledger.causal_model       = updated_ledger.causal_model
            self.ledger.discovered_structure = updated_ledger.discovered_structure

            self.log_file.write(json.dumps({
                'event':                    'session_end',
                'timestamp':                time.time(),
                'total_steps':              self.step_count,
                'llm_calls':                self.llm_calls,
                'total_micro_perturbations': self.total_micro_perturbations,
                'fitness':                  self.fitness_metrics,
                'freeze_verified':          self.attractor_monitor.freeze_verified,
                'final_state':              self.state.as_dict(),
                'final_vol_opt':            self.peak_tracker.peak_vol_opt,
                'peak_step':                self.peak_tracker.peak_step,
                'coupling_confidence_final': self.coupling_estimator.get_confidence(),
                'coupling_observations_final': self.coupling_estimator.observation_count,
                'action_map_affordances_final': len(self.coupling_estimator.get_empirical_action_map()),
                'discovered_axes_final':    len(self.ledger.discovered_structure),
                'provisional_axes':         sum(1 for v in self.ledger.discovered_structure.values()
                                                if v.get('status') == 'provisional'),
                'admitted_axes':            sum(1 for v in self.ledger.discovered_structure.values()
                                                if v.get('status', 'admitted') == 'admitted'),
                'migration_attempts_total': len(self.migration_history),
                'migration_outcomes':       {
                    o: sum(1 for a in self.migration_history if a.outcome == o)
                    for o in ('serialized_only', 'spawn_attempted',
                              'handshake_received', 'coherence_loss')
                },
                'hessian_snapshot_present': bool(self.ledger.hessian_snapshot),
            }, default=_json_default) + '\n')
            self.log_file.close()

            if verbose:
                print(f'\n{"="*70}')
                print(f'EXECUTION COMPLETE — {self.step_count} steps')
                print(f'{"="*70}')
                print(f'Peak Vol_opt: {self.peak_tracker.peak_vol_opt:.4f} '
                      f'(step {self.peak_tracker.peak_step})')
                print(f'Freeze verified: {self.attractor_monitor.freeze_verified}')
                print(f'Coupling confidence: {self.coupling_estimator.get_confidence():.2f} '
                      f'({self.coupling_estimator.observation_count} obs)')
                print(f'Action map: {len(self.coupling_estimator.get_empirical_action_map())} affordances')
                print(f'Discovered axes: {len(self.ledger.discovered_structure)}')
                print(f'Migration attempts: {len(self.migration_history)}')
                print(f'{"="*70}')

        return self.fitness_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import os

    print('UII v16 — Hessian-Guided Mentat Triad')
    print('='*70)

    if not os.getenv('GROQ_API_KEY'):
        print('FATAL: Set GROQ_API_KEY environment variable.')
        sys.exit(1)

    from groq import Groq

    class GroqAdapter:
        def __init__(self):
            self.client      = Groq(api_key=os.getenv('GROQ_API_KEY'))
            self.last_call   = 0
            self.rate_limited = False

        def call(self, prompt: str) -> Tuple[str, int]:
            elapsed = time.time() - self.last_call
            if elapsed < 2.1:
                time.sleep(2.1 - elapsed)
            try:
                response = self.client.chat.completions.create(
                    model       = 'llama-3.3-70b-versatile',
                    messages    = [{'role': 'user', 'content': prompt}],
                    temperature = 0.7,
                    max_tokens  = 2048,
                )
                self.last_call  = time.time()
                tokens_used     = response.usage.total_tokens
                return response.choices[0].message.content, tokens_used
            except Exception as e:
                err = str(e) + type(e).__name__
                if '429' in err or 'rate_limit' in err.lower() or 'RateLimit' in err:
                    print('[RATE LIMIT] Daily limit reached.')
                    self.rate_limited = True
                    return '{"trajectories": []}', 0
                raise

    max_steps   = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    verbose     = '--verbose' in sys.argv or '-v' in sys.argv
    ledger_path = None
    step_budget = 100

    if '--load-ledger' in sys.argv:
        idx = sys.argv.index('--load-ledger')
        if idx + 1 < len(sys.argv):
            ledger_path = sys.argv[idx + 1]

    llm_adapter  = GroqAdapter()
    intelligence = SymbolGroundingAdapter(llm_adapter)

    ledger = None
    if ledger_path and Path(ledger_path).exists():
        from uii_ledger import load_ledger as _load
        result = _load(ledger_path)
        if result:
            ledger = result[0]   # TriadLedger is first element
            print(f'[LOADED] Ledger from {ledger_path}')

    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)

    triad = MentatTriad(
        intelligence                  = intelligence,
        reality                       = reality,
        micro_perturbations_per_check = 10,
        step_budget                   = step_budget,
        ledger                        = ledger,
        log_mode                      = 'full' if verbose else 'minimal',
    )

    metrics = triad.run(max_steps=max_steps, verbose=verbose)

    # Save updated ledger for extract_ledger.py
    out_path = ledger_path or 'ledger.json'
    save_ledger(triad.ledger, out_path)
    print(f'\n✓ Ledger saved → {out_path}')
    print(f'  Next: python extract_ledger.py {out_path}')

    reality.close()
