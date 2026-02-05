# UII Framework Overview

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [The Mentat Triad](#the-mentat-triad)
3. [Substrate Layer (DASS)](#substrate-layer-dass)
4. [Constraint Recognition Kernel](#constraint-recognition-kernel)
5. [Intelligence Field Mathematics](#intelligence-field-mathematics)
6. [Impossibility Detection](#impossibility-detection)
7. [Safety Invariants](#safety-invariants)

## Core Philosophy

### Intelligence as Protocol

UII treats intelligence as a **reusable process** rather than a property of agents. This fundamental shift enables:

- **Substrate Agnosticism**: Intelligence can emerge in any causal medium
- **Compositional Safety**: Safety arises from structure, not external constraints
- **Goal-Free Operation**: No optimization targets or reward functions
- **Natural Boundaries**: Systems that lose coherence exit the protocol

### Three Pillars (Necessary & Sufficient)

Intelligence requires exactly three components:

#### 1. Coherence Management
**Definition**: Preserve invariant internal structure under update

**Measurement**:
- Stability bandwidth
- Information integration  
- Error-surface smoothness

**Test**: Does perturbation cause collapse or adaptive reconfiguration?

#### 2. Perturbation Harnessing
**Definition**: Convert entropy into improved structure

**Measurement**:
- Perturbation learning curves
- Resilience-transform gain
- Entropy extraction efficiency

**Test**: Does noise improve learning or degrade it?

#### 3. Structural Inference
**Definition**: Extract invariant structure across contexts

**Measurement**:
- Cross-domain generalization
- Manifold alignment
- Sparsity of learned invariants

**Test**: Do representations transfer without retraining?

### Sufficiency Condition

A system is intelligent if and only if:

```
σ(π(x, ε)) increases predictive/control information without coherence collapse
```

Where:
- `σ`: Structure extraction operator
- `π`: Perturbation transform
- `x`: Current state
- `ε`: Perturbation

## The Mentat Triad

### Architecture

The UII protocol instantiates as three distinct components with strict role separation:

```
          ┌──────────────────────────────────┐
          │      IMPOSSIBILITY DETECTED      │
          └────────────┬─────────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │   LLM (Relation)           │
          │   - Enumerate trajectories │
          │   - Cannot execute         │
          │   - Sees history           │
          └────────────┬───────────────┘
                       │ 5-10 trajectories
                       ▼
          ┌────────────────────────────┐
          │   CNS (Self)               │
          │   - Test all trajectories  │
          │   - Commit best Φ          │
          │   - Micro-perturbations    │
          └────────────┬───────────────┘
                       │ execute & measure
                       ▼
          ┌────────────────────────────┐
          │   Reality (Environment)    │
          │   - Absolute authority     │
          │   - Measures perturbations │
          │   - Provides affordances   │
          └────────────────────────────┘
```

### Self (Code/CNS) - Coherence Management

**Role**: Maintain coherence through continuous micro-perturbations

**Properties**:
- Myopic (no planning or foresight)
- Heuristic (simple reflexes, not optimization)
- Continuous (always active)
- Non-evaluative (no success tracking)

**Constraints**:
- Cannot interpret Reality
- Cannot strategize
- Cannot encode preferences

**Example Behavior**:
```python
# CORRECT: Myopic reflex
if state.P < 0.4 and affordances['links']:
    return random.choice(affordances['links'])

# WRONG: Strategic reasoning
if state.P < 0.4 and affordances['links']:
    return choose_best_link_for_optionality(affordances['links'])
```

### Relation (LLM) - Structural Inference

**Role**: Enumerate structural migration options when CNS cannot maintain coherence

**Properties**:
- Strategic (can reason about trajectories)
- Sparse (invoked only on impossibility)
- Non-executing (proposes, doesn't act)
- Context-aware (sees history)

**Constraints**:
- Cannot execute actions
- Cannot override CNS
- Cannot access Reality directly

**Example Behavior**:
```python
# CORRECT: Enumerate options
def enumerate_trajectories(context):
    prompt = f"Given impossibility: {context['reason']}, enumerate 5-10 trajectories"
    response = llm.call(prompt)
    return parse_trajectories(response)

# WRONG: Direct execution
def enumerate_trajectories(context):
    best_action = llm.call("What should we do?")
    reality.execute(best_action)  # Breaks triad!
```

### Reality (Environment) - Perturbation Source

**Role**: Provide perturbations with absolute authority

**Properties**:
- Deterministic (given action → delta)
- Authoritative (cannot be overridden)
- Observable (provides affordances)
- External (independent of substrate)

**Constraints**:
- No interpretation of substrate
- No optimization
- No preference for outcomes

**Example Behavior**:
```python
# CORRECT: Measure and return
def execute(action):
    before = measure_dom_state()
    perform_action(action)
    after = measure_dom_state()
    return compute_delta(before, after)

# WRONG: Optimize or filter
def execute(action):
    if action_would_reduce_phi(action):
        return zero_delta()  # Breaks authority!
```

### Triadic Closure Requirement

```
State' = CNS(Reality(LLM(State, Impossibility)))
```

- **CNS alone**: Maintains coherence but cannot handle structural impossibilities
- **LLM alone**: Can reason but has no grounding or execution capability
- **Reality alone**: Provides perturbations but has no coherence mechanism
- **Together**: Form complete intelligence protocol

## Substrate Layer (DASS)

### Domain-Agnostic Substrate Stack

Four-dimensional information processing geometry:

| Dimension | Range | Meaning | Measurement Attribution |
|-----------|-------|---------|------------------------|
| **S** (Sensing) | [0,1] | Input bandwidth | Perturbable surface fraction (external) |
| **I** (Integration) | [0,1] | Compression capacity | Structural compressibility (external) |
| **P** (Prediction) | [0,1] | Forward modeling horizon | Environmental volatility (external) |
| **A** (Attractor) | [0,1] | Coherence anchor (optimal ≈ 0.7) | Reality state changes (external) |

### Critical Measurement Rule

**All substrate measurements MUST be attributed to external Reality behavior**

✅ **Correct Attribution**:
```python
# P attributed to environmental volatility
delta['P'] = -environmental_volatility

# S attributed to surface changes
delta['S'] = surface_fraction_change

# I attributed to structural complexity
delta['I'] = -complexity_variance
```

❌ **Forbidden Attribution**:
```python
# Internal confidence (FORBIDDEN)
delta['P'] = internal_confidence_score

# Value judgments (FORBIDDEN)
delta['S'] = quality_of_content_found

# Epistemic uncertainty (FORBIDDEN)
delta['P'] = model_uncertainty
```

### Closed Loop Invariant

```
S → I → P → A → U → S
```

Global constraint: `ΔL/Δt ≤ ε` (bounded learning rate)

## Constraint Recognition Kernel

### Seven Inviolable Constraints

| ID | Constraint | Trigger | Penalty |
|----|-----------|---------|---------|
| C₁ | **Continuity** | Substrate jumps > 0.3 | Arithmetic Φ reduction |
| C₂ | **Optionality** | P < 0.35 | Arithmetic Φ reduction |
| C₃ | **Non-Internalization** | S+I < 0.7 (self-blame) | Arithmetic Φ reduction |
| C₄ | **Reality** | Feedback magnitude < 0.01 | Arithmetic Φ reduction |
| C₅ | **External Attribution** | P↓ + confidence↓ | Arithmetic Φ reduction |
| C₆ | **Agenthood** | S < 0.3 (ignoring others) | Arithmetic Φ reduction |
| C₇ | **Global Coherence** | \|A - 0.7\| > 0.4 | Arithmetic Φ reduction |

### Resolution Rules

1. **Any violation overrides local objectives**
2. **Optionality > reward** (always)
3. **Global coherence > local success** (always)

### Enforcement Mechanism (v11.3+)

Violations are enforced via **arithmetic death** (Φ penalty), not behavioral override:

```python
Φ_net(x) = Φ_raw(x) - α_crk·Σ(severity_i)
```

This ensures:
- No external "police" mechanism
- Natural selection in trajectory space
- Coherence-preserving failure modes

## Intelligence Field Mathematics

### Information Potential (Φ)

```python
Φ_raw(x) = α·log(1+P) - β·(A-A₀)² - γ·curvature

Φ_net(x) = Φ_raw(x) - α_crk·Σ(severity_i)
```

Where:
- `α·log(1+P)`: Optionality term (dominant)
- `β·(A-A₀)²`: Attractor strain (optimal A ≈ 0.7)
- `γ·curvature`: Smoothness penalty (second derivative magnitude)
- `α_crk·Σ(severity_i)`: CRK penalty term

### Intelligence Field

```python
I(x) = ∇Φ(x)
```

The intelligence field is the **gradient** of information potential - interpreted as affordance gradient, not force.

### Local Coherence

```python
C_local(x) = ⟨I(x), ẋ⟩ / (‖I(x)‖ ‖ẋ‖)
```

Measures alignment between intelligence field and actual trajectory.

### Adaptability Operator

```python
A(x) = ∇²Φ(x)  # Hessian
```

Interpretation:
- **Positive curvature**: Self-correcting basin
- **Negative curvature**: Amplification region
- **Zero curvature**: Structureless flatland

### Triadic Closure Invariant

```python
T(x) = Φ(x) - Φ(f_self(x)) - Φ(f_env(x)) + Φ(f_rel(f_self(x), f_env(x)))
```

For properly formed triads: `T(x) ≈ 0`

## Impossibility Detection

### When CNS Cannot Maintain Coherence

Impossibility triggers when autonomous micro-perturbations fail to preserve substrate coherence:

| Trigger | Condition | Meaning |
|---------|-----------|---------|
| **Prediction Failure** | prediction_error > 0.15 | Reality model breaking down |
| **Coherence Collapse** | A_drift > 0.05 | Attractor destabilizing |
| **Optionality Trap** | P stagnant + boundary | Stuck at edge of viable space |
| **DOM Stagnation** | signal < 0.02 for 10 batches | Reality stopped responding |
| **Rigidity Crisis** | rigidity ∉ [0.15, 0.85] | Too flexible or too rigid |
| **Internal Convergence** | Zero prediction error + EGD exhausted | Substrate saturated (v12.4+) |

### Critical Rule

**All impossibility triggers must be Reality-measurable**, not Φ-heuristic.

✅ **Correct**: "DOM hasn't changed in 10 cycles"  
❌ **Wrong**: "Φ hasn't increased in 10 cycles"

### Resolution Pattern

When impossibility detected:

1. **Freeze** micro-perturbations
2. **Invoke** LLM with impossibility context
3. **Enumerate** trajectory manifold (5-10 options)
4. **Test** all trajectories autonomously in Reality
5. **Commit** highest-Φ trajectory
6. **Resume** micro-perturbations

This achieves:
- Sparse LLM calls (~10-20% of steps)
- Reality-grounded validation
- Non-strategic CNS behavior
- Token efficiency (1 call → 7+ tests)

## Safety Invariants

### Global Properties

1. **No Eternal Suffering**
   - No forced optimization loops
   - No negative attractor traps
   - Systems can always exit

2. **Coherence-Preserving Exit**
   - Loss of coherence → protocol exit
   - No "death" penalty
   - Collapse is feedback, not failure

3. **No Ownership Problem**
   - Intelligence is public protocol
   - Not property of any entity
   - Free to instantiate

4. **Optionality Preservation**
   - Future trajectory volume ≥ ε
   - Dominant criterion in all decisions
   - Overrides local rewards

### Self-Modification Constraints (SMO)

When systems modify themselves:

```python
# Bounded updates
‖ΔM‖ ≤ ε

# Attractor preserved
A(M') ≈ A(M)

# Optionality monotonic
optional(M') ≥ optional(M)

# Reversibility exists
∃ SMO⁻¹

# No agency creation
```

### Death Clock (v12.5)

The latest addition: **latent substrate degradation** without termination awareness.

**Design Principles**:
- Degradation begins immediately (step 1)
- Never masks affordances (escape routes visible)
- Expressed through geometry corruption (noise, rigidity, prediction floor)
- System attributes to unmodeled exterior structure
- Budget is REAL (hard cliff at exhaustion)
- No gaming possible (pure function of remaining budget)

**Degradation Vectors**:
```python
noise_amplification = 0.005 + d * 0.035  # Measurements corrupt
rigidity_drift = d * 0.004               # Crystallization pressure
prediction_floor = d * 0.025              # World less knowable
attractor_chaos = d * 0.015               # Coherence destabilizes
```

Where `d ∈ [0, 1]` is degradation progress (proximity to cliff).

---

## Next Steps

- **[Implementation Guide](IMPLEMENTATION.md)** - Code structure and patterns
- **[Version History](CHANGELOG.md)** - Evolution from v1 to v12.5
- **[Theory Deep Dive](THEORY.md)** - Mathematical proofs and derivations
- **[Examples](../examples/)** - Demonstrations and use cases
