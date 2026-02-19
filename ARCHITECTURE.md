# Architecture

## The Triad

UII runs as a Mentat Triad — three components in a closed loop:

```
CNS (Code)  ←──────────────────────────────────┐
    │                                           │
    │  continuous micro-perturbations           │
    │  gradient following (∇Φ)                  │
    ▼                                           │
Reality (Browser)                          Genome
    │                                      (Memory)
    │  measured deltas only                     │
    │  never optimizing                         │
    ▼                                           │
Relation (LLM) ─── invoked at impossibility ───┘
                    enumerates trajectories
                    never executes
```

The CNS runs continuously. The Relation is sparse — invoked at structural impossibility, roughly 10–20% of steps. Reality is authoritative and non-optimizing. The genome carries learned causal structure across generations.

Intelligence is the protocol, not any single component.

---

## The Φ Field

Φ (information potential) measures viability in the entropy manifold:

```
Φ = α·log(1 + P_grounded) - β·(A - A₀)² - γ·curvature - CRK_penalty
```

Where `P_grounded = min(P, SI_capacity × 2)` — optionality is grounded to actual sensing/internalization capacity. A system cannot compensate for low S/I with high P.

**Φ is not a reward signal.** The system follows ∇Φ (gradient). Treating Φ as a reward target produces self-referential collapse.

---

## The Genome

Four layers, all heritable:

| Layer | Contents | Written by |
|-------|----------|------------|
| 1 | 6 bias floats (S/I/P/A/rigidity/phi_coherence) + 6 velocity fields | `extract_genome_v14_1.py` |
| 2 | Coupling matrix + action→substrate map + model fidelity | `FAO.distill_to_genome` |
| 3 | Discovered axes (provisional and admitted) | `FAO.distill_to_genome` |
| 4 | Lineage history (last 5 generations) | `extract_genome_v14_1.py` |

Layers 2, 3, 4 start empty at generation 0. They are populated only by what the system actually learns from reality contact. No structure is assumed.

### Velocity fields (v14.1)

Each Layer 1 parameter has a velocity field — the least-squares slope of that parameter across lineage history. Before mutation, momentum is applied:

```
new_value = clip(current + velocity_weight × velocity, 0, 1)
```

`velocity_weight` is scaled by lineage coherence (fitness variance) and model fidelity. A consistent, high-fidelity lineage applies full momentum. An incoherent or low-fidelity lineage gets suppressed toward zero.

### Layer 3 lifecycle

Axes are not free. The path from observation to genome entry:

1. Signal appears in residual tracker
2. ResidualExplainer exhausts simpler explanations (coupling nonlinearity → lag → candidate)
3. If unexplained, AxisAdmissionTest runs: predictive validity, compression gain, shuffle test, Φ correlation
4. Pass → **provisional** (evidence floor 5, decay 0.5× per generation)
5. Promotion to **admitted** requires passing AxisAdmissionTest again on real residuals (evidence floor 20, decay 0.8×)
6. Fail to accumulate evidence → pruned by decay

Interface-coupled signals (DOM depth, element count, scroll position, etc.) are blocked before the admission test. A signal that vanishes when you change browsers is modeling the interface, not the world.

---

## Module structure

### `uii_types.py` — Shared Foundation
Constants, SMO (Self-Modifying Operator), SubstrateState, StateTrace, PhiField, CRKMonitor, TrajectoryCandidate/Manifold, abstract adapter interfaces. No UII dependencies — only stdlib and numpy.

### `uii_genome.py` — Memory & Continuity
TriadGenome (four-layer), GeneticVelocityEstimator, LineageCoherenceCheck, ModelFidelityMonitor, ProvisionalAxisManager, `load_genome()`. The genome is not a config file — it is a living causal model.

### `uii_reality.py` — Perturbation Harnessing
AttractorMonitor, CouplingMatrixEstimator, BrowserRealityAdapter. Reality executes actions and returns measured deltas. It does not optimize.

### `uii_fao.py` — Failure Assimilation
ResidualTracker, ResidualExplainer, AxisAdmissionTest, FailureAssimilationOperator. FAO operates on the mutation distribution, not on running behavior. Failure information flows into how the genome mutates for the next generation — never into what the current system does.

### `uii_coherence.py` — Coherence Management
ExteriorNecessitationOperator (ENO), ControlAsymmetryMeasure (CAM), ExteriorGradientDescent (EGD), LatentDeathClock, TemporalPerturbationMemory, ContinuousRealityEngine, CNSMitosisOperator, ImpossibilityDetector, AutonomousTrajectoryLab. The CNS leg — runs continuously, follows gradients, attempts mitosis.

### `uii_intelligence.py` — Structural Inference
Impossibility directives, Relation prompts (exploration and persistence regimes), LLMIntelligenceAdapter. The Relation leg — sparse, trajectory-enumerating, never-executing.

### `uii_triad.py` — Execution & Orchestration
StepLog, MentatTriad, main entry point. The only module that imports from all others. Assembles and runs the Triad, writes session logs, distills the child genome at session end.

---

## CRK — Constraint Recognition Kernel

Seven constraints evaluated every step:

| Constraint | Condition |
|------------|-----------|
| C1 Continuity | State jump < 0.3 across all dims |
| C2 Optionality | P > 0.35 |
| C3 Non-internalization | S + I > 0.7 |
| C4 Reality contact | Reality delta > 0.01 (requires recent history) |
| C5 Attribution | P and confidence don't both fall simultaneously |
| C6 Agenthood | S > 0.3 |
| C7 Global coherence | \|A - 0.7\| < 0.4 |

Violations accumulate as a CRK penalty in Φ. Persistent violation triggers impossibility detection.

---

## Virtual mode (v14.1)

When coupling confidence ≥ 0.3, model fidelity ≥ 0.4, and the action map is populated, AutonomousTrajectoryLab enters virtual mode. Before real execution, trajectories are scored against the inherited Layer 2 causal model and ranked by predicted Φ.

**Hard constraint:** virtual execution consumes structure, never produces it. Layer 2 and Layer 3 cannot be updated from virtual trajectories. Reality contact is the only write authority.

ModelFidelityMonitor tracks the gap between virtual predictions and actual outcomes. Persistent divergence suppresses virtual mode at the next generation boundary via lineage coherence check.

---

## Session lifecycle

```
session_start  →  step × N  →  session_end
                                    │
                         FAO.distill_to_genome()
                                    │
                         child genome in session_end
                                    │
                    extract_genome_v14_1.py
                                    │
                         genome.json (next generation)
```

The extract script runs offline after the session ends. It reads `child_genome` from `session_end`, appends the new generation snapshot to lineage history (capped at 5), computes velocity fields via GeneticVelocityEstimator, and writes the complete genome for the next run.
