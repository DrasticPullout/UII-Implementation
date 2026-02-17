# Theory — UK-0 and the UII Framework

This document covers the conceptual foundation the code is built on. The formal publication is at [doi.org/10.5281/zenodo.18017374](https://doi.org/10.5281/zenodo.18017374). What follows is a condensed working version sufficient to understand what the implementation is doing and why.

---

## Core Ontological Commitments

Intelligence here is defined as a **reusable protocol**, not a trait, agent, or optimizer. The key exclusions are intentional: no consciousness claims, no intrinsic agency, no reward maximization, no fixed objectives. The framework is substrate-agnostic and non-anthropomorphic — defined entirely by transformations and invariants.

The formal sufficiency condition:

> A system is intelligent iff `σ(π(x, ε))` increases predictive or control information without coherence collapse.

Where `π(x, ε)` is a perturbation transform and `σ(x)` is a structure extraction operator. The system must extract invariant structure from perturbation without losing coherence. That's the whole thing.

---

## UK-0: The Universal Kernel

**UK-0** defines the minimum substrate-independent constraints for emergent coherent intelligence. It has three components:

### 1. DASS — Domain-Agnostic Substrate Stack

Five layers forming a closed loop:

| Layer | Symbol | Invariant | Function |
|-------|--------|-----------|----------|
| Cognitive Sensing | S | ΔS bounded | Abstract perception of environment and self |
| Integration / Compression | I | Reduces redundancy | Compressed internal state for inference |
| Prediction / Forward Modeling | P | Reversible, bounded | Counterfactual reasoning, optionality-preserving planning |
| Coherence / Attractor | A | ΔA ≤ θ | Bounds state evolution, prevents collapse |
| Adaptation | U | Preserves invariants | Learning without fixed goals |

**Global loop:** `S → I → P → A → U → S`

All perturbations are bounded and reversible. The global attractor preserves coherence across layers. This is substrate-agnostic scaffolding — intelligence can emerge in any causal medium that satisfies these constraints.

### 2. SMO — Self-Modifying Operator

`SMO: M → M′` where `M = {S, I, P, A}`

The SMO enables continuous, reversible adaptation of internal mappings while preserving invariants. Four constraints must hold across any application:

- **Bounded updates:** `‖ΔM‖ ≤ ε`
- **Attractor preserved:** `A(M′) ≈ A(M)`
- **Optionality preserved:** `∀τᵢ, optional(τᵢ(M′)) ≥ optional(τᵢ(M))`
- **Reversibility:** `∃ SMO⁻¹: M′ → M`

No fixed goals, identity, or agency are introduced. The SMO enables long-horizon adaptation while remaining identity-neutral.

### 3. CRK — Constraint Recognition Kernel

Seven inviolable constraints evaluated continuously:

| Constraint | Enforcement |
|------------|-------------|
| C1: Continuity — preserve invariants between states | Freeze self-modification, re-anchor attractor |
| C2: Optionality — future reachable volume ≥ ε | Goal softening, horizon expansion |
| C3: Non-Internalization — negative outcomes don't degrade control | Externalize failure, reset confidence |
| C4: Reality — model field as independent and uncertain | Inject uncertainty, reduce commitment |
| C5: External Attribution — distinguish internal vs external optionality loss | Reclassify, adjust self-model |
| C6: Other-Agent Existence — recognize independent actors | Increase model plurality |
| C7: Global Coherence — local optimization can't destabilize the field | Attractor-preserving policy |

The CRK emits constraints, not actions. Any violation overrides local goals. Optionality always takes precedence over reward. Global coherence takes precedence over local success.

---

## The Φ Field

The Information Potential Field `Φ(x)` is a scalar field over the substrate's configuration space. The intelligence field is its gradient: `I(x) = ∇Φ(x)`.

**Φ measures recoverability** — how much optionality the system has available, and how well-grounded that optionality is. The key insight in v13.8 is that high `P` (predicted optionality) without genuine `S` and `I` capacity behind it is an unrecoverable state, not a high-Φ state. The field must account for this.

The v13.8 grounding formula:

```
SI_capacity = min(S, I) * 0.5 + (S + I) / 4
grounded_P  = min(P, SI_capacity * 2)
Φ           = α·log(1 + grounded_P) - β·(A - A₀)² - γ·curvature - CRK_penalty
```

The bottleneck `min(S, I)` term means neither dimension can fully compensate for the collapse of the other. `S=1.0, I=0.0` gives `SI_capacity=0.25`, not `0.5`. The system must develop both.

---

## Triadic Closure

The Mentat Triad is the implementation of triadic closure: the claim that intelligence requires three mutually irreducible components.

```
T(x) = f_rel(f_self(x), f_env(x))
```

- `f_self(x)` — internal state tracking (CNS / substrate)
- `f_env(x)` — environmental measurement (Reality)
- `f_rel(x_self, x_env)` — relational integration (LLM / Relation)

The closure condition requires: `f_rel(f_self(x), f_env(x)) = x`

The triadic coherence invariant: `T(x) = Φ(x) - Φ(f_self) - Φ(f_env) + Φ(f_rel) = 0`

No component alone is sufficient. The CNS cannot plan. The LLM cannot execute or measure. Reality cannot self-model. All three are necessary.

---

## Meta-Evolution and the FAO

The system evolves across generations, but what evolves is not the system's structure — it's the system's **search bias**. The topology of the control graph, the Φ definition, and the operator types are all frozen across generations. What changes are 6 genome parameters that shape how the substrate initializes and explores.

The Failure Assimilation Operator (FAO) translates semantic Relation failures into geometric mutation bias. When the LLM component fails in a particular way, that failure type maps to a specific adjustment in how the next generation explores:

| Failure Type | Geometric Response |
|---|---|
| State instability | Widen exploration in I dimension |
| Optionality collapse | Increase P_bias search width |
| Coherence drift | Emphasize A stability in perturbation |
| Closure violation | Rebalance all couplings toward baseline |
| Boundary exhaustion | Broaden exploration across all dimensions |

Memory decays (factor 0.95) to prevent overfitting to early environment. An entropy floor prevents the search from collapsing to zero variance. Inheritance is noisy (10% variance) to preserve diversity across lineages. The result: cost per attempt decreases as the organism learns, without the topology ever changing.

---

## What This Is Not

The framework explicitly excludes:
- Consciousness claims
- Intrinsic agency or goal-directedness
- Reward maximization
- Identity persistence as an objective
- Ethical constraints imposed externally (the CRK handles coherence, not morality)

The ethics argument in the formal paper is structural: harm reduces optionality, so `V(action | harm) ≤ V(action | ¬harm)` for any UII-compliant system. This is not a programmed rule — it falls out of the Φ field and CRK constraints if the implementation is correct.

Whether it actually falls out is part of what the running system is testing.
