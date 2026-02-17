# Architecture — UII v13.8

How the theoretical framework maps to running code. This is a module-by-module reference — if you want to understand the concepts first, start with [THEORY.md](THEORY.md).

**v13.8 key changes from v13.7:**
- Stricter S/I grounding via bottleneck formula (P can no longer be inflated when I=0)
- Fitness diagnostic includes S/I health (not factored into selection score — testing whether Φ grounding handles it alone)
- `session_end` log entry guaranteed even on API rate-limit crash
- Graceful 429 handling — clean termination, no exception

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                    MentatTriad                      │
│                                                     │
│  ┌──────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │   CNS    │   │  Relation   │   │   Reality   │  │
│  │  (Code)  │◄──│   (LLM)     │◄──│  (Browser)  │  │
│  └────┬─────┘   └─────────────┘   └──────┬──────┘  │
│       │                                   │         │
│       └───────── micro-perturbations ─────┘         │
│                                                     │
│  SubstrateState [S, I, P, A]  ←  all measurements  │
│  PhiField Φ                   ←  recoverability     │
│  CRKMonitor C1–C7             ←  invariants         │
│  AttractorMonitor             ←  freeze detection   │
│  FailureAssimilationOperator  ←  inter-gen learning │
│  CNSMitosisOperator           ←  geometric mitosis  │
│  LatentDeathClock             ←  dual budget        │
└─────────────────────────────────────────────────────┘
```

---

## Execution Flow

One `step()` runs as follows:

1. `death_clock.tick_step()` — advance step counter
2. `ENO.check_activation()` — check if affordance gating analysis is warranted
3. **Micro-perturbation batch (×10):** choose action → execute in Reality → `apply_delta()` → record trace → ENO/CAM record outcome
4. `check_agent_responses()` — integrate any non-blocking user/agent responses
5. `attractor_monitor.record_state_signature()` — freeze detection
6. `mitosis_operator.check_triggers()` — attempt geometric mitosis if triggered
7. `impossibility_detector.check_impossibility()` — six triggers checked
8. **If impossible:** ENO/EGD pattern discovery → LLM enumerate trajectories → Lab test all → commit best → `death_clock.tick_tokens(actual_usage)`
9. FAO: classify Relation failure → `assimilate_relation_failure()` → update mutation bias
10. FAO: stochastic bias reset check (if Φ stagnating)
11. Compute `phi_after`, `violations_after`
12. `death_clock.should_terminate()?`
13. Write `StepLog` per logging mode

---

## Modules

### SubstrateState
Four-dimensional information processing geometry. All measurements attributed externally — no internal confidence.

| Dim | Meaning | Optimal |
|-----|---------|---------|
| S | Sensing bandwidth | High |
| I | Integration capacity | High |
| P | Prediction horizon / optionality | Grounded to S/I |
| A | Attractor / coherence | ~0.7 |

**`apply_delta(observed, predicted)`** — each dim updated via SMO. P ceiling damping: `-0.05*(P-0.9)²` when P>0.9.

---

### SMO (Self-Modifying Operator)
Bounded, reversible substrate updates with rigidity adaptation.

- `apply()` — modulates delta by `(1 - 0.3*rigidity)`. Rigidity +0.01 on low error, -0.02 on high error, -0.001 decay per step
- `reverse()` — pops previous state from history (depth 10)
- Prediction error tracked over rolling window of 100

---

### PhiField — v13.8 Grounding
**The core change in v13.8.** P is grounded to actual S/I capacity before contributing to Φ.

```python
SI_capacity = min(S, I) * 0.5 + (S + I) / 4
grounded_P  = min(P, SI_capacity * 2.0)
Φ = α·log(1 + grounded_P) - β·(A - A₀)² - γ·curvature - α_crk·Σ(violations)
```

| S | I | SI_capacity | Max grounded_P |
|---|---|-------------|----------------|
| 1.0 | 1.0 | 1.0 | 2.0 (unconstrained) |
| 1.0 | 0.0 | 0.25 | 0.5 (hard cap) |
| 0.5 | 0.5 | 0.5 | 1.0 (unconstrained) |

The bottleneck term `min(S,I)` means neither dimension can compensate for the other's collapse. This prevents the system from claiming high optionality it cannot actually recover.

---

### CRKMonitor
Seven constraints evaluated every step. Any violation penalizes Φ.

| Constraint | Threshold |
|------------|-----------|
| C1 Continuity | `Σ\|Δdim\| < 0.3` |
| C2 Optionality | `P ≥ 0.35` |
| C3 Non-Internalization | `S + I ≥ 0.7` |
| C4 Reality | `\|feedback\| ≥ 0.01` |
| C5 Attribution | If P↓ and (S+I)↓ simultaneously |
| C6 Agenthood | `S ≥ 0.3` |
| C7 Global Coherence | `\|A - 0.7\| ≤ 0.4` |

---

### AttractorMonitor
Detects basin stability (freeze). Freeze = `|ΔΦ| < 0.01` for 10 consecutive steps AND no CRK violations. Freeze lost if either condition breaks.

Freeze triggers prompt switch: pre-freeze uses EXPLORATION prompt, post-freeze uses PERSISTENCE prompt (focused on geometric externalization and substrate migration).

---

### BrowserRealityAdapter
Playwright browser as Reality. All substrate deltas computed from actual DOM changes — never simulated.

| Delta | Source |
|-------|--------|
| S | `0.7 * surface_delta + 0.3 * coverage_delta` |
| I | Complexity variance over 3-step window |
| P | Negative volatility variance over 5-step window |
| A | `-url_change - error_appearance` |

Boundary pressure (constant 0.85) damps S, I, A response and adds noise — represents the inherent difficulty of operating in external web environment. Python affordance (`exec()`) available ungated from step 1.

---

### ContinuousRealityEngine
CNS micro-perturbation selection. Myopic — no planning.

State-driven thresholds:
- `P > 0.7` → observe (high prediction, don't disturb)
- `|A - 0.7| > 0.25` → scroll (restore coherence)
- `S < 0.4` → read / evaluate (increase sensing)
- `P < 0.4` → click / navigate / type (expand prediction)

Temporal memory (window=5 steps, capacity=20) prevents oscillation between same loci.

---

### ImpossibilityDetector
Six triggers that hand control to the LLM:

| Trigger | Condition |
|---------|-----------|
| A: Prediction Failure | prediction error > 0.15 |
| B: Coherence Collapse | std(A) over 10 states > 0.05 |
| C: Optionality Trap | (P<0.25 or P>0.85) AND var(P)<0.01 AND stagnant≥15 steps |
| D: DOM Stagnation | total signal < 0.02 for 10 consecutive batches |
| E: Rigidity Crisis | rigidity < 0.15 OR > 0.85 |
| F: Pattern Collapse | ENO+EGD: all clusters → zero control |

Bootstrap state (about:blank) triggers immediately. All triggers are Reality-based — not Φ heuristics.

---

### ENO / CAM / EGD — Pattern Discovery

Three cooperating components that discover which action combinations actually produce control:

**ENO (ExteriorNecessitationOperator)** — activates when prediction error is consistently low (<0.005 for 20 steps). Empirically identifies which affordances are being refused (refusal rate >0.6 over 30 attempts).

**CAM (ControlAsymmetryMeasure)** — builds a covariance graph of affordances. Measures whether action A followed by action B produces more substrate change than A alone. Extracts dense clusters via DFS.

**EGD (ExteriorGradientDescent)** — selects the highest-control cluster from CAM's graph. When all clusters collapse to zero control, triggers Impossibility F.

When ENO is active, EGD replaces CNS micro-action selection.

---

### LLMIntelligenceAdapter
Enumeration only — the LLM never executes or measures. Invoked only on impossibility.

Two prompt modes:
- **EXPLORATION** (pre-freeze) — find stable basin, expand affordance compositions
- **PERSISTENCE** (post-freeze) — geometric externalization, substrate migration

Per-trigger directives shape the prompt focus (bootstrap, optionality trap, rigidity crisis, etc.). Parse pipeline degrades gracefully: JSON block → JSON repair → partial regex extraction → fallback observe.

---

### AutonomousTrajectoryLab
Tests all candidate trajectories from the LLM in Reality before any commitment. Each test: deepcopy state+trace → `execute_trajectory()` → apply all deltas → collect CRK violations → compute Φ. Best selected by measured Φ, not LLM estimates.

---

### FailureAssimilationOperator (FAO)
Translates what went wrong semantically into geometric mutation bias for the next generation. Crosses the CNS/Relation boundary without changing topology.

**Semantic → Geometric mappings:**

| Failure Type | Bias Update |
|---|---|
| `state_instability` | Widen I coupling exploration |
| `optionality_collapse` | Increase P_bias sigma |
| `coherence_drift` | Emphasize A in perturbation weighting |
| `closure_violation` | Decay all couplings toward baseline 1.0 |
| `serialization_failed` | Increase rigidity_init exploration |
| `boundary_exhaustion` | Broaden all genome sigmas |

Memory decays at 0.95/step. Entropy floor `[0.01, 0.5]` prevents sigma collapse. Inheritance noise (10%) preserves lineage diversity. Stochastic reset (20% probability) on Φ decline escapes local basins.

---

### CNSMitosisOperator
Geometric mitosis — externalizes invariant structure to file so a child instance can load it.

Two triggers:
- **Opportunistic:** Φ std < 0.01 over 5 steps AND P > 0.7 AND no CRK violations
- **Boundary compression:** Φ declining AND degradation > 80%

Process: FAO-informed genome mutation → verify closure (topology hash identical AND coupling distance < 0.05) → verify optionality (child diversity ≥ parent diversity) → write `kernel_snapshot.json` with learned bias included → verify geometry persisted on reload.

What never mutates: topology, Φ definition, operator types.
What may mutate: genome parameters (FAO-weighted), coupling values (within ε=0.05).

---

### LatentDeathClock
Dual-budget mortality. Terminates on whichever exhausts first: step budget OR token budget (tracked from actual Groq API usage). Boundary pressure = 0.85 constant — represents environmental difficulty, not a countdown.

---

### TriadGenome + Fitness

**Genome (6 floats):** `S_bias, I_bias, P_bias, A_bias, rigidity_init, phi_coherence_weight`

These initialize the substrate basin and Φ field shape. Gaussian mutation (σ=0.1, FAO-biased) produces offspring.

**Fitness (v13.8):**
```
fitness = freeze_speed + token_efficiency + survival + migration_bonus
        = (100 / freeze_step) + (50000 / tokens_to_freeze) + survival_steps + 100·(migration? 1:0)
```

S/I health is **logged and reported but excluded from the fitness score**. This is a deliberate experimental choice: if v13.8 Φ grounding is correctly load-bearing, balanced S/I should emerge from the grounded field without additional selection pressure. If it doesn't, that's a finding about whether the grounding is sufficient.

---

### Logging Modes

| Mode | What's Logged |
|------|--------------|
| `minimal` | Only impossibility events, freeze events, and commits |
| `fitness` | Φ and token count every step |
| `full` | Complete StepLog every step |

Log file: `mentat_triad_v13_8_log.jsonl` (append mode — grows across generations).
