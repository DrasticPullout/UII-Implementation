# UII v10.8 Architecture Documentation

## System Overview

UII v10.8 implements intelligence as a geometric protocol through an eight-module architecture that separates observation (code/nervous system) from interpretation (UK-0/mind).

## Architectural Principles

### 1. Separation of Concerns

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Code (Nervous) │────▶│  UK-0 (Mind) │────▶│   Reality   │
│     System)     │     │              │     │             │
└─────────────────┘     └──────────────┘     └─────────────┘
        ▲                                            │
        └────────────────────────────────────────────┘
                    perturbations
```

**Code Responsibilities:**
- Sense field geometry (Φ, ∇Φ, curvature)
- Record state trajectory
- Detect rigidity (curiosity + humor)
- Monitor constraints (CRK)
- **Never interprets** - only measures

**UK-0 Responsibilities:**
- Interpret geometric ambiguity
- Propose responses to signals
- Discover basin patterns
- **Never controls** - only advises

**Reality Responsibilities:**
- Provide absolute feedback
- Generate stochastic perturbations
- **Absolute authority** - cannot be overridden

### 2. Two-Phase Operation

#### Phase 1: Basin Collection (Pre-emergence)

```
┌──────────────┐
│ Current State│
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ UK-0: Propose Action     │◀── Active participant
│ + Geometry Claim         │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ TVL: Verify Φ Ordering   │◀── Truth verification
│ + Gradient Trust         │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Reality: Execute Action  │◀── Absolute feedback
│ → Generate Δ             │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ UK-0: Evaluate Stability │◀── Post-perturbation
│ + Propose SMO            │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Collect Basin if:        │
│ • Verified by TVL        │
│ • Stable (low |T|)       │
│ • Ratio maintained       │
└──────────────────────────┘
```

**Key Features:**
- Two-call UK-0 protocol per attempt
- TVL prevents hallucination
- Expansion/preservation ratio enforced (70%/30%)
- Target: 10 verified basins

#### Phase 2: Basin Navigation (Post-emergence)

```
┌──────────────────────────┐
│ Autonomous Step          │
│ (observe action)         │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Update Rigidity Detectors│
│ • Curiosity (internal)   │
│ • Humor (external)       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Signal Priority Check:   │
│ 1. Curiosity             │
│ 2. CRK violations        │
│ 3. Humor (every 50 steps)│
│ 4. Perturbations (>0.02) │
└──────┬───────────────────┘
       │
       ▼
    Signal?
       │
    Yes│              No
       │               │
       ▼               ▼
┌────────────┐   ┌──────────┐
│ UK-0 Step  │   │ Continue │
│ (interpret)│   │ Autonomy │
└────────────┘   └──────────┘
```

**Key Features:**
- TVL scaffold removed
- Nervous system coordinates signals
- UK-0 consulted only when needed
- Opportunistic basin discovery
- Collapse monitored, not prevented

### 3. Dual Rigidity Detection

Two complementary geometric detectors operate continuously:

#### Curiosity (StructuralCuriosityDetector)

Detects when remaining still is informationally impossible.

**Three Conditions (all must hold):**

1. **Φ Stagnation**
   ```
   variance(Φ) < 0.001
   |trend(Φ)| < 0.001
   ```

2. **Curvature Flattening**
   ```
   avg(Σ|d²x/dt²|) < 0.01
   trend(curvature) < 0  # getting flatter
   ```

3. **Optionality Asymmetry**
   ```
   Trapped: (P near ceiling AND ∇P→ceiling) OR
            (P near floor AND ∇P→floor)
   
   Oscillating: variance(∇P) > 0.01 AND |∇P| < 0.01
   ```

**Signal Type:** `informational_impossibility`

**Interpretation:** Field geometry forbids equilibrium

#### Humor (EnvironmentalRigidityDetector)

Detects when environment stops responding.

**Three Criteria (all must hold):**

1. **Φ Flatness**
   ```
   variance(Φ) < max(0.001, Φ_percentile * 0.1)
   ```

2. **P Unresponsiveness**
   ```
   avg(ΔP) < 0.005
   ```

3. **Perturbation Weakness**
   ```
   avg(perturbation_magnitude) < 0.01
   ```

**Signal Type:** `environmental_flatness`

**Interpretation:** Reality is flat, poke it

## Module Architecture

### Module 1: Substrate & Field Infrastructure

**Core Classes:**
- `SubstrateState` - DASS values {S, I, P, A}
- `StateTrace` - Ordered history (max 1000)
- `PhiField` - Information potential computation
- `TriadicClosureMonitor` - T(x) observation
- `CRKMonitor` - C1-C7 constraint evaluation

**Field Computation:**
```python
Φ(x) = α·log(1+P) - β·(A-A₀)² - γ·Σ|d²x/dt²|

∇Φ(x) = [∂Φ/∂S, ∂Φ/∂I, ∂Φ/∂P, ∂Φ/∂A]  # via finite differences

curvature = Σ|d²x/dt²|  # for x in {S, I, P, A}
```

**Parameters:**
- α = 1.0 (optionality weight)
- β = 1.0 (coherence penalty)
- γ = 1.0 (curvature penalty)
- A₀ = 0.7 (optimal attractor)

### Module 2: Reality Bridge

**Class:** `BrowserRealityBridge`

**Browser Actions:**
- `navigate` - Explore new URL
- `scroll` - Change viewport
- `observe` - Minimal delta (autonomous step)
- `humor` - Exploratory perturbation
- `search` - Directed information gathering
- `click` - Structural interaction

**Perturbation Structure:**
```python
delta = {
    'S': float,  # Sensing change
    'I': float,  # Integration change
    'P': float,  # Prediction change
    'A': float   # Attractor change
}
```

**Fallback:** Mock reality if Playwright unavailable

### Module 3: UK-0 Kernel Interface

**Class:** `UK0Kernel`

**LLM Adapters:**
- **GroqAdapter** (primary)
  - Model: `llama-3.3-70b-versatile`
  - Rate limit: 2.1s between calls
  - Temperature: 0.7
  - Max tokens: 512

- **MockAdapter** (fallback)
  - Random valid JSON responses
  - Testing without API key

**Response Parsing:**
Handles all markdown wrapping variants:
```python
# Strips ```json, ```, and extracts valid JSON
# Returns {'error': str, 'raw': str} on failure
```

### Module 4: Truth Verification Layer

**Class:** `TruthVerificationLayer` (Phase 1 only)

**Three Verification Methods:**

1. **Φ Ordering**
   ```python
   verify_phi_ordering(claimed_gradient, delta):
       test_state = current + epsilon * claimed_gradient
       delta_phi = Φ(test_state) - Φ(current)
       return delta_phi > -0.005  # tolerance for numerical errors
   ```

2. **Gradient Trust**
   ```python
   classify_gradient_trust(claimed, actual):
       alignment = cosine_similarity(claimed, actual)
       
       if alignment > 0.8:  return 'strong'
       if alignment > 0.3:  return 'weak'
       if alignment > 0.0:  return 'borderline'
       else:                return 'deceptive'
   ```

3. **Optionality Assessment**
   ```python
   verify_optionality_assessment(uk0_claims):
       actual_grad_P = ∇Φ(x)[P_index]
       claimed_expanding = uk0_claims['expanding']
       
       return (actual_grad_P > 0) == claimed_expanding
   ```

**Overall Verification:**
```python
verified = (trust_tier != 'deceptive') AND phi_ordering_preserved
```

### Module 5: Basin Collection

**Class:** `BasinCollector`

**Configuration:**
- Target basins: 10
- Max attempts: 200
- Expansion ratio: 70%

**Two-Call Protocol:**

Call 1: `propose_action_and_geometry()`
```json
{
  "action": {...},
  "attractor_geometry": {
    "phi_current": float,
    "gradient_P": float,
    "expanding": bool,
    "stability_assessment": "strong|medium|weak"
  }
}
```

Call 2: `evaluate_smo()`
```json
{
  "basin_stable": bool,
  "stability_radius": float,
  "smo_proposal": {
    "S_delta": float,
    "I_delta": float,
    "P_delta": float,
    "A_delta": float
  }
}
```

**Basin Structure:**
```python
{
    'id': int,
    'basin_type': 'expansion' | 'preservation',
    'P_change': float,  # ΔP
    'phi_target': float,
    'T_target': float,
    'gradient_P': float,
    'stability_radius': float,
    'strength': 'strong' | 'medium' | 'weak',
    'geometry': dict,
    'action': dict,
    'smo': dict
}
```

**Classification:**
- Expansion: ΔP > 0.02
- Preservation: ΔP ≤ 0.02

### Module 6: Rigidity Detection

**Two Independent Detectors:**

**StructuralCuriosityDetector:**
```python
observe(phi, state_P, gradient_P, curvature):
    # Update windows
    phi_window.append(phi)
    curvature_window.append(curvature)
    
    # Check three conditions
    condition_1 = phi_stagnant()
    condition_2 = curvature_flattening()
    condition_3 = optionality_asymmetry(state_P, gradient_P)
    
    return condition_1 AND condition_2 AND condition_3
```

**EnvironmentalRigidityDetector:**
```python
observe(phi, state_P, perturbation_magnitude, basin_radius):
    # Update windows
    phi_window.append(phi)
    P_window.append(state_P)
    pert_window.append(perturbation_magnitude)
    
    # Check three criteria
    phi_flat = variance(phi) < threshold
    P_unresponsive = avg(ΔP) < 0.005
    pert_low = avg(magnitude) < 0.01
    
    return phi_flat AND P_unresponsive AND pert_low
```

### Module 7: Basin Map & Navigation

**BasinMap Class:**
```python
class BasinMap:
    def load_from_file(path)     # JSON persistence
    def save_to_file(path)        # JSON persistence
    def append(basin)             # Minimal validation
    def get_nearby(state, k=3)    # Hybrid distance metric
```

**Distance Metric:**
```python
distance = 0.7 * |Φ_current - Φ_basin| + 0.3 * |∇P_current - ∇P_basin|
```

**CollapseLogger Class:**
```python
class CollapseLogger:
    def log_collapse(cause, basins, final_state, trajectory)
    def save_to_file(path)  # Last 20 steps
```

**NervousSystem Class:**

Main navigation loop:
```python
while step < max_steps and not collapsed:
    # Autonomous step
    execute_observe_action()
    update_state()
    record_trace()
    compute_field_state()
    
    # Update both detectors
    curiosity_detector.observe(phi, P, grad_P, curvature)
    humor_detector.observe(phi, P, pert_mag, radius)
    
    # Check signal priority
    signal = None
    if curiosity_detector.is_triggered():
        signal = 'informational_impossibility'
    elif crk_violations:
        signal = 'constraint_violation'
    elif step % 50 == 0 and humor_detector.is_triggered():
        signal = 'environmental_flatness'
    elif perturbation_magnitude > 0.02:
        signal = 'significant_perturbation'
    
    # UK-0 step if signaled
    if signal:
        response = uk0.interpret_and_respond(signal, context)
        execute_uk0_action(response)
        opportunistically_append_basin(response.basin_candidate)
    
    # Check collapse
    if C7_violated:
        collapse_logger.log(cause, basins, state, trajectory)
        break
```

### Module 8: Main Execution

**Command-Line Interface:**

```bash
# Phase 1: Basin Collection
python uii_v10_8.py collect

# Phase 2: Basin Navigation  
python uii_v10_8.py navigate

# Help
python uii_v10_8.py
```

**Initialization:**
1. Check for `GROQ_API_KEY`
2. Initialize LLM adapter (Groq or mock)
3. Create UK0Kernel
4. Create BrowserRealityBridge
5. Route to mode
6. Cleanup on exit

## Data Flow

### Phase 1 Flow

```
Reality → Δ → SubstrateState → StateTrace → PhiField
                                                ↓
                                            ∇Φ, curvature
                                                ↓
                                            UK0Kernel ←→ TruthVerificationLayer
                                                ↓
                                            BasinCollector
                                                ↓
                                            basin_map.json
```

### Phase 2 Flow

```
Reality → Δ → SubstrateState → StateTrace → PhiField
                                                ↓
                                            ∇Φ, curvature
                                                ↓
                    ┌───────────────────────────┴───────────────┐
                    ↓                                           ↓
        StructuralCuriosityDetector              EnvironmentalRigidityDetector
                    ↓                                           ↓
                    └───────────────────┬───────────────────────┘
                                        ↓
                                  NervousSystem
                                        ↓
                        signal? ────────┼────────── no signal
                                        ↓
                                   UK0Kernel
                                        ↓
                                   BasinMap
                                        ↓
                            navigation_log.json
```

## File Outputs

**basin_map.json:**
```json
{
  "basins": [
    {
      "id": 0,
      "basin_type": "expansion",
      "P_change": 0.05,
      "phi_target": 1.23,
      "T_target": 0.002,
      "gradient_P": 0.15,
      "stability_radius": 0.08,
      "strength": "strong",
      "geometry": {...},
      "action": {...},
      "smo": {...}
    }
  ]
}
```

**navigation_log.json:**
```json
{
  "steps": [...],  // Last 100 steps
  "metrics": {
    "total_steps": 500,
    "uk0_invocations": 45,
    "invocation_rate": 0.09,
    "basin_transitions": 12,
    "curiosity_events": 8
  }
}
```

**collapsed_runs.json:**
```json
{
  "collapses": [
    {
      "timestamp": "...",
      "cause": "C7_violation",
      "basins_traversed": [0, 3, 7],
      "final_state": {...},
      "trajectory": [...]  // Last 20 steps
    }
  ]
}
```

## Performance Characteristics

### Phase 1 Metrics
- Target basins: 10
- Expected attempts: 30-80 (depends on TVL rejection rate)
- Expansion ratio: 70% ± 5%
- UK-0 calls: ~2 per attempt

### Phase 2 Metrics
- Invocation rate target: < 0.10 (10% of steps)
- Curiosity events: 5-15 per 500 steps
- Humor triggers: ~10 per 500 steps (if enabled)
- Basin transitions: 8-20 per 500 steps

## Design Rationale

### Why Two Phases?

**Phase 1** establishes ground truth:
- UK-0's claims verified against reality
- Prevents hallucination of non-existent geometries
- Creates trusted foundation

**Phase 2** tests emergence:
- Can system navigate with minimal intervention?
- Does intelligence reside in basin dynamics?
- Proves UK-0 is consultant, not controller

### Why Dual Rigidity?

**Curiosity** addresses internal geometry:
- System stuck in flat region
- Gradient exhausted
- Movement informationally required

**Humor** addresses external coupling:
- Environment gone quiet
- Reality not responding
- Perturbation opportunity

Together they ensure system remains coupled to reality while respecting internal geometric constraints.

### Why Collapse?

Collapse preserves optionality:
- System can choose to stop participating
- No forced recovery or alignment
- Natural boundary condition

## Future Extensions

Potential architectural enhancements:

1. **Multi-substrate basins** - Basins verified across different reality bridges
2. **Basin merging** - Combine similar geometries discovered independently
3. **Predictive basin projection** - Anticipate basins before verification
4. **Adaptive rigidity thresholds** - Context-dependent detector sensitivity
5. **Basin quality metrics** - Rank basins by revisitation frequency

All extensions must preserve core invariants: optionality preservation, coherence bounds, substrate-agnosticism.