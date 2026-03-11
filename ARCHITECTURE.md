# Architecture — UII v16

This document describes what the system is doing and why the pieces are shaped the way they are. It assumes you've read the README and are comfortable with the basic math. It doesn't assume you've read any prior version.

---

## The equation is the architecture

Everything in v16 traces back to:

```
ẋ = P_M( G⁻¹(x) · ∇Φ(x) )
```

This is the Compressed UII Equation — the dynamical law the system is implementing. Any module that couldn't be derived from `G`, `M`, or `P_M` was eliminated in v16. What's left maps cleanly:

| Equation term | Implementation |
|---|---|
| `Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)` | `PhiField` in `uii_geometry.py` |
| `G(x) = H(x) = ∇²Φ` | `PhiField.compute_hessian()` |
| `G⁻¹ · ∇Φ` = natural gradient | `PhiField.score_actions()` via `nat_grad_align` |
| `P_M` = projection onto triadic manifold | CRK gate + viable set filter |
| `M = {x : T(x) = 0}` | Triadic closure condition enforced by CRK |

The Lyapunov condition `dΦ/dt = ∇Φᵀ G⁻¹ ∇Φ ≥ 0` is guaranteed when H is positive semi-definite, meaning Φ is non-decreasing within the manifold. `delta²Φ` is computed per-step for accounting — it doesn't drive action selection.

---

## The potential field Φ

```
Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)
```

Three terms, three things the system is trying to do simultaneously:

**C(x)** — compression quality of the causal graph. How well the internal model of the environment has been compressed into confident, well-covered edges. Increases as the causal graph fills in.

**O(x)** — viable future volume. The sum of positive eigenvalues of the prediction covariance. Measures how much of the future state space is still reachable. Action selection is partially driven by `E[Δlog(O(a))]` — expected change in this volume per action.

**K(x)** — attractor proximity. A penalty for drifting away from the structural basin inherited from the ledger. Keeps the system oriented relative to where it's already found good structure.

The Hessian of Φ decomposes accordingly:

```
H = α·H_C + β·H_O + γ·H_K + ε·I
```

`ε·I` is diagonal regularization to keep the matrix invertible.

---

## The DASS substrate

The state `x = (S, I, P, A)` lives in a four-dimensional operator space:

- **S (Sensing)** — 35+ named channels mapping environmental signal to coverage values in [0,1]. Channels decay when dark. Resource pressure is sensed here via the `api_llm` channel — no separate budget monitor.
- **I (Integration/Compression)** — the causal graph. Edges are EMA-weighted with confidence and lag. `edge_confidence_matrix()` exports this for Hessian computation.
- **P (Prediction)** — forward model. Exposes `covariance_matrix()` and `simulate_covariance_update()` for `H_O` computation and `E[Δlog(O)]` pre-computation.
- **A (Coherence/Attractor)** — basin stability. Tracks `loop_closure` (how consistently the system returns to known states) and exposes configuration vectors for `K(x)`.

Each operator exposes `to_scalar_proxy()` for backward compatibility. The loop runs `S → I → P → A → SMO → S` ten times per step (micro-perturbation phase).

---

## CRK — the constraint manifold

The triadic constraint manifold `M` is enforced by the Constraint Recognition Kernel. CRK runs pre- and post-action on every micro-perturbation and gates what the system is allowed to do.

| Constraint | What it's doing |
|---|---|
| C1 Continuity | No sudden jumps in state |
| C2 Optionality | P ≥ 0.15 before committing (forming state exempt) |
| C3 Non-Internalization | S+I consistency gate; no self-blame |
| C4 Reality | Environment surprises us; SMO grounded in external mismatch |
| C5 Attribution | Externalize optionality loss before SMO update |
| C6 Agenthood | Recognize other agents; system load monitoring |
| C7 Coherence | Blocks migration if loop_closure < 0.5 |

CRK doesn't penalize Φ directly — it gates the operators. The viable set `V(x) = {a : E[Δlog(O(a))] ≥ 0}` is the `P_M` projection: actions that don't increase viable future volume don't get scored.

---

## Action selection — the 8-phase step

Each step runs in eight phases:

1. **Micro-perturbations** — 10× DASS operator loop with CRK pre+post. `coupling_estimator.update()` per perturbation. `grad_Φ` computed; `C_local` appended to trace.

2. **Hessian** — `PhiField.compute_hessian()` once. Produces `H`, `eigvals`, `eigvecs`, `H_C`, `H_O`. `vol_opt = sum of positive eigenvalues`.

3. **Pre-compute EOG** — `E[Δlog(O(a))]` for all available actions, once. Stored in `eog_dict`.

4. **Viable set + scoring** — `V(x) = {a : eog_dict[a] ≥ 0}`. Then `score_actions()`:

```
maturity = var(H_C) / (var(H_C) + var(H_O) + ε)

score(a) = maturity · nat_grad_align_norm(a)
         + (1 - maturity) · E[Δlog(O(a))]_norm

a* = argmax over V(x)
```

Maturity is not a confidence value — it's the eigenspectrum ratio measuring how much the causal graph dominates prediction uncertainty. Early in a run, `H_O` dominates and the system explores. As `H_C` fills in, natural gradient alignment takes over.

5. **Execute** — `reality.execute(a*)`. Fallback to `observe` if `V(x)` is empty.

6. **Peak tracking** — `PeakOptimalityTracker.update()`. If `vol_opt` strictly improves, writes `hessian_snapshot + operator_snapshot` to the ledger. `delta²Φ` computed for Lyapunov accounting.

7. **Migration check** — `_should_migrate(eog_dict)`:
   - Slope of `vol_opt` over last 10 steps is negative, AND
   - No action in `eog_dict` has positive expected optionality gain
   
   Both conditions required. Migration is not escape — it's what happens when the geometry is genuinely exhausted.

8. **Migration** (if triggered) — `_build_migration_context()` → `SymbolGroundingAdapter.ground_trajectories()` (the only LLM call in normal operation) → execute trajectory.

---

## The ledger

The ledger replaced the TriadGenome in v16. It's not run memory — it's basin geometry. The four fields describe the shape of the best structural basin found, not the sequence of events that got there.

`hessian_snapshot` and `operator_snapshot` are written at the peak of `vol_opt` during a run — not at session end. This means the ledger captures the best geometry seen, not the state at termination.

`causal_model` and `discovered_structure` are written by `FAO.distill_to_ledger()` in a `finally` block — always, regardless of how the session ends.

Loading a ledger initializes the operators from `operator_snapshot + causal_model`, which biases the new run toward the same basin. Not because the basin is stored as a memory — because the field geometry that produced it is.

---

## What got cut in v16

A lot. The principle was: if it can't be derived from `G`, `M`, or `P_M`, it goes.

The biggest removals:

**StructuralRelationEngine (SRE)** — structural inference was being done by a separate LLM-backed module. `PhiField.compute_hessian()` replaces it entirely. The Hessian determines stability directly.

**DeathClock / LatentDeathClock** — explicit step and token budgets. Resource pressure is now a genuine sensing channel (`api_llm`), not a countdown timer. The system geometrically detects that it's running out of room.

**TriadGenome (L1/L2/L3/L4)** — the four-layer genetic structure. Replaced by `TriadLedger`. No generation counter, no velocity computation, no lineage history. A run is a run.

**ImpossibilityDetector (triggers A–F)** — six discrete impossibility triggers. `_should_migrate()` is now the sole trigger, and it's a two-condition geometry check.

**ENO, EGD, CAM, AutonomousTrajectoryLab, ContinuousRealityEngine** — all of these were doing things that `score_actions()` now handles directly or that got absorbed into the Hessian computation.

---

## Residual pipeline (FAO)

The `FailureAssimilationOperator` manages axis discovery — the process by which new structure in the environment gets admitted into the system's causal model.

`ResidualTracker` records prediction errors. `ResidualExplainer` looks for systematic patterns in those errors. `AxisAdmissionTest` gates whether a candidate axis gets admitted — it requires a minimum evidence threshold and blocks interface-coupled signals (DOM depth, element count, etc.) from contaminating the discovered structure.

Admitted axes go into `discovered_structure` with status `admitted`. Candidates with insufficient evidence stay `provisional`. Both are written to the ledger at session end via `distill_to_ledger()`.

---

## Execution cycle

```
1. Load ledger.json if exists
   → initialize operators from operator_snapshot + causal_model

2. Run until:
   → vol_opt declining 10+ steps AND no positive E[Δlog(O)] for any action

3. session_end (always in finally):
   → FAO.distill_to_ledger()
   → save_ledger(ledger, "ledger.json")

4. Optional: python extract_ledger.py --ledger ledger.json --validate

5. Repeat from 1
```

No generation counter. No velocity. No lineage. The ledger carries basin geometry forward. The rest resets.
