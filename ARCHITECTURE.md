# Architecture

## Module Dependency Order

```
uii_operators.py        ← no UII dependencies (numpy only)
uii_types.py            ← uii_operators
uii_genome.py           ← uii_types
uii_reality.py          ← uii_types
uii_structural.py       ← uii_types, uii_operators
uii_intelligence.py     ← uii_types
uii_coherence.py        ← uii_types, uii_operators
uii_fao.py              ← uii_types, uii_genome, uii_reality
uii_triad.py            ← everything above
extract_genome_v14_1.py ← standalone CLI utility
```

`uii_triad.py` is the only file that imports across all modules. Nothing imports from it.

---

## Module Roles

### `uii_operators.py` — DASS Operator Layer
Implements the four substrate operators: `SensingOperator`, `CompressionOperator`, `PredictionOperator`, `CoherenceOperator`, and `SelfModifyingOperator`. Each exposes `to_scalar_proxy()` for Φ field compatibility. The SMO governs per-layer update rates with plasticity/rigidity logic. No UII dependencies — this is the base layer everything else builds on.

### `uii_types.py` — Shared Foundation Types
All shared data structures, abstract interfaces, and constants. `SubstrateState`, `StateTrace`, `PhiField`, `CRKMonitor`, `TrajectoryCandidate`, `TrajectoryManifold`, `RealityAdapter` (ABC), `IntelligenceAdapter` (ABC). Nothing in the codebase depends on a module that depends on this one — it is the single import root.

### `uii_genome.py` — Memory & Continuity
The fourth leg of the Triad. The genome is not a config file — it carries heritable causal structure across generations. Four layers: operator geometry + velocity fields (L1), learned causal model (L2), discovered emergent axes (L3), lineage history (L4). `GeneticVelocityEstimator` makes L1 predictive via least-squares slope. Only predictively valid structure survives compression into the child genome.

### `uii_reality.py` — Perturbation Harnessing (Reality Leg)
Authoritative, non-optimizing source of perturbations. Executes actions, returns measured deltas. Contains `BrowserRealityAdapter` (Playwright), `AttractorMonitor` (basin stability detection for mitosis gating), and `CouplingMatrixEstimator` (empirical S/I/P/A co-movement — the learned causal model that feeds into Layer 2 of the genome).

### `uii_structural.py` — Structural Relation Engine (Relation Leg, Stage 1)
Runs at every impossibility event at zero token cost. Reads from `CouplingMatrixEstimator` and `CAM` outputs — it applies σ(x)'s outputs, it does not perform σ(x). Backward pass: diagnose why the system is stuck. Forward pass: determine what geometry resolves it. Routes to one of three tiers: `TIER_1_CNS_WEIGHT` (geometric fix, bias CNS directly), `TIER_2_LAB` (test candidate shapes in the trajectory lab), `TIER_3_LLM` (symbol grounding required).

### `uii_intelligence.py` — Symbol Grounding (Relation Leg, Stage 2)
Called only when SRE sets `resolution_tier = TIER_3_LLM`. The LLM role is narrow: fill in concrete tokens (URLs, Python code, CSS selectors) for trajectories the SRE has already geometrically determined. Receives a compact geometry package, not a raw impossibility dump. `RelationAdapter` wires SRE + `SymbolGroundingAdapter` behind the `IntelligenceAdapter` interface.

### `uii_coherence.py` — Coherence Management (CNS Leg)
The CNS leg of the Mentat Triad. Maintains internal coherence through continuous micro-perturbations and gradient-following (∇Φ), not scalar thresholds. Key components: `ContinuousRealityEngine` (micro-action selection), `ControlAsymmetryMeasure` (empirical action→substrate map), `ExteriorNecessitationOperator` (gating under boundary pressure), `AutonomousTrajectoryLab` (trajectory testing with virtual mode), `CNSMitosisOperator` (geometric replication attempts), `ImpossibilityDetector` (structural impossibility from Reality signals only).

### `uii_fao.py` — Failure Assimilation Operator
The legal mechanism by which Relation failures inform CNS search geometry. Operates on the mutation distribution — never on action selection. Input: observable signals only (Φ delta, CRK violation types, trajectory outcomes). Output: mutation distribution shape. Contains the residual learning stack: `ResidualTracker`, `ResidualExplainer`, `AxisAdmissionTest` (4-condition gate), and `distill_to_genome()` which writes the compressed session into the child genome.

### `uii_triad.py` — Execution & Orchestration
Assembles the Mentat Triad and runs it. The only file that imports from all modules. `MentatTriad` is the orchestrator: CNS (Coherence) + Relation (SRE + Symbol Grounding) + Reality (Perturbation), held together by Memory (Genome). `StepLog` records the full state of each step. Exposes `on_step_complete` callback hook for real-time monitoring. Contains the main entry point and CLI.

### `extract_genome_v14_1.py` — Genome Extraction Utility
Standalone CLI. Reads the last `session_end` record from a v14.1 log file and writes `genome.json` for the next generation. Computes velocity fields via `GeneticVelocityEstimator`, appends lineage snapshots (max 5), and prunes lineage to Summary Vector at G ≥ 1000. Backward compatible with v14 logs.

---

## Triad Structure

```
        ┌─────────────────────────────────────────┐
        │              MentatTriad                │
        │                                         │
        │   CNS          Relation        Reality  │
        │ (coherence)  (structural +   (reality)  │
        │              intelligence)              │
        │      └───────────── FAO ───────────┘   │
        │                    │                    │
        │               Genome (Memory)           │
        └─────────────────────────────────────────┘
```

The Relation leg is two-stage: SRE runs first (zero cost), LLM grounding only when SRE routes to `TIER_3_LLM`. FAO is the bridge — failure signals from Relation reshape the mutation distribution that CNS searches over. The genome carries the compressed residue of each session into the next generation.

---

## Per-Step Execution Flow

1. Choose micro-action: gradient alignment (primary) or reflex heuristic
2. Pre-action CRK evaluation → `phi_modifier`
3. `Reality.execute()` → observes ΔS, ΔP (I=0, A=0 from Reality)
4. DASS: S → I (with P feedback) → P → A → SMO
5. Post-action CRK → SMO gating
6. If allowed: `SelfModifyingOperator.apply()` → updates operators
7. Compute gradient = `Φ.gradient()` → `trace.record(C_local)`
8. `CAM.record_action_outcome()` → gradient-aligned EMA per affordance

One step = 10 micro-perturbations + impossibility check + Tier routing.
