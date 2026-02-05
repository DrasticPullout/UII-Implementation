# UII Implementation Guide

## Table of Contents

2. [Core Components](#core-components)
3. [Execution Flow](#execution-flow)
4. [Adding Features](#adding-features)
5. [Common Patterns](#common-patterns)
6. [Validation Checklist](#validation-checklist)



## Core Components

### 1. Substrate (DASS)

**Location**: `class SubstrateState`

Four-dimensional state space:

```python
@dataclass
class SubstrateState:
    S: float  # Sensing [0,1]
    I: float  # Integration [0,1]
    P: float  # Prediction [0,1]
    A: float  # Attractor [0,1]
    
    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}
```

**Key Methods**:
- `apply_delta()`: Update substrate with measured perturbations
- `rollback()`: Reverse to previous state (SMO reversibility)

### 2. Self-Modifying Operator (SMO)

**Location**: `class SMO`

Bounded, reversible substrate updates:

```python
class SMO:
    def apply(self, current: float, observed_delta: float, 
              predicted_delta: float = 0.0) -> float:
        """Apply delta with rigidity modulation + history"""
        
        # Track prediction error
        prediction_error = abs(observed_delta - predicted_delta)
        self.prediction_error_history.append(prediction_error)
        
        # Update rigidity (asymmetric: easier to loosen than rigidify)
        rigidity_change = 0.01 if prediction_error < 0.02 else -0.02
        rigidity_decay = -0.001
        self.rigidity = np.clip(
            self.rigidity + rigidity_change + rigidity_decay, 
            0.0, 1.0
        )
        
        # Modulate delta by rigidity
        modulated_delta = observed_delta * (1.0 - 0.3 * self.rigidity)
        return np.clip(current + modulated_delta, *self.bounds)
```

**v12.5 Addition** - Death Clock Degradation:
```python
def inject_degradation(self, degradation: Dict[str, float]):
    """Inject mortality pressure into substrate dynamics"""
    
    # Rigidity drift - crystallization pressure
    self.rigidity = np.clip(
        self.rigidity + degradation['rigidity_drift'],
        0.0, 1.0
    )
    
    # Prediction floor - synthetic error injection
    if degradation['prediction_floor'] > 0:
        self.prediction_error_history.append(
            degradation['prediction_floor']
        )
```

### 3. Information Potential Field (Φ)

**Location**: `class PhiField`

Scalar field measuring latent structure:

```python
def phi(self, state: SubstrateState, trace: StateTrace, 
        crk_violations: List[Tuple[str, float]] = None) -> float:
    """Compute net information potential"""
    
    # Optionality term (dominant)
    opt = np.log(1.0 + max(state.P, 0.0))
    
    # Attractor strain
    strain = (state.A - self.A0) ** 2
    
    # Curvature (smoothness penalty)
    recent = trace.get_recent(3)
    curv = 0.0
    if len(recent) >= 3:
        h0, h1, h2 = recent[-3], recent[-2], recent[-1]
        for k in ["S", "I", "P", "A"]:
            curv += abs(h2[k] - 2*h1[k] + h0[k])
    
    phi_raw = self.alpha * opt - self.beta * strain - self.gamma * curv
    
    # CRK penalty (arithmetic death)
    crk_penalty = 0.0
    if crk_violations:
        crk_penalty = self.alpha_crk * sum(
            severity for _, severity in crk_violations
        )
    
    return phi_raw - crk_penalty
```

### 4. Constraint Recognition Kernel (CRK)

**Location**: `class CRKMonitor`

Seven inviolable constraints:

```python
def evaluate(self, state: SubstrateState, trace: StateTrace,
             reality_delta: Optional[Dict] = None) -> List[Tuple[str, float]]:
    """Evaluate all constraints, return violations"""
    violations = []
    
    # C1: Continuity
    if len(trace) >= 2:
        recent = trace.get_recent(2)
        prev = recent[-2]
        jump = sum(abs(prev[k] - getattr(state, k)) 
                   for k in ["S", "I", "P", "A"])
        if jump > 0.3:
            violations.append(("C1_Continuity", jump - 0.3))
    
    # C2: Optionality
    if state.P < 0.35:
        violations.append(("C2_Optionality", 0.35 - state.P))
    
    # ... C3-C7 similar pattern
    
    return violations
```

### 5. Reality Adapter

**Location**: `class BrowserRealityAdapter`

Playwright-based DOM interaction:

```python
def execute(self, action: Dict, 
            degradation: Dict[str, float] = None) -> Tuple[Dict, Dict]:
    """Execute action in Reality, return measured delta"""
    
    before_metrics = self._measure_dom_state()
    
    # Execute action
    if action_type == 'navigate':
        self.page.goto(params['url'], wait_until='domcontentloaded')
    elif action_type == 'click':
        self.page.click(params['selector'])
    # ... other actions
    
    after_metrics = self._measure_dom_state()
    delta = self._compute_delta_from_dom(before_metrics, after_metrics)
    
    # v12.5: Inject degradation if provided
    if degradation:
        for key in ['S', 'I', 'A']:
            noise = np.random.uniform(
                -degradation['noise_amplification'],
                degradation['noise_amplification']
            )
            delta[key] += noise
        
        chaos = np.random.uniform(
            -degradation['attractor_chaos'],
            degradation['attractor_chaos']
        )
        delta['A'] += chaos
    
    return delta, context
```

**Key Method** - DOM Delta Computation:
```python
def _compute_delta_from_dom(self, before: Dict, after: Dict) -> Dict:
    """Measure substrate changes from DOM state"""
    delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}
    
    # S: Perturbable surface fraction
    current_surface = after['interactive_count'] / max(after['element_count'], 1)
    prev_surface = before['interactive_count'] / max(before['element_count'], 1)
    surface_delta = current_surface - prev_surface
    delta['S'] = np.clip(surface_delta, -0.1, 0.1)
    
    # I: Structural compressibility
    current_complexity = after['dom_depth'] * (after['element_count'] / max(after['dom_depth'], 1))
    complexity_variance = np.var(recent_complexities)
    delta['I'] = np.clip(-0.5 * complexity_variance / max(current_complexity, 1), -0.08, 0.08)
    
    # P: Environmental volatility
    volatility = np.mean([structural_delta, dom_delta, text_delta])
    volatility_variance = np.var(recent_volatility)
    delta['P'] = np.clip(0.1 - volatility_variance * 10.0, -0.1, 0.1)
    
    # A: Reality state changes
    if after['url'] != before['url']:
        delta['A'] -= 0.05
    if after['has_errors'] and not before['has_errors']:
        delta['A'] -= 0.08
    
    return delta
```

### 6. Continuous Reality Engine (CNS)

**Location**: `class ContinuousRealityEngine`

Myopic, heuristic micro-perturbations:

```python
def choose_micro_action(self, state: SubstrateState, 
                       affordances: Dict) -> Dict:
    """State-driven reflexes (NOT strategic)"""
    
    # REFLEX 1: P > 0.7 → environment stable, back off
    if state.P > 0.7:
        return {'type': 'observe', 'params': {}}
    
    # REFLEX 2: |A - 0.7| > 0.25 → attractor drifting, stabilize
    if abs(state.A - 0.7) > 0.25:
        if scrollable:
            return {'type': 'scroll', 'params': {'direction': 'down'}}
    
    # REFLEX 3: S < 0.4 → low sensing, probe surface
    if state.S < 0.4:
        if affordances['readable']:
            return {'type': 'read', 'params': {...}}
    
    # REFLEX 4: P < 0.4 → low prediction, need new surface
    if state.P < 0.4:
        # Ascending cost: click → type/fill → navigate
        # ...
    
    # REFLEX 5: P in [0.4, 0.7] → mid-range, low-cost signal
    # scroll > read > click > delay
    
    # DEFAULT: observe (surface exhausted)
    return {'type': 'observe', 'params': {}}
```

**Critical**: All thresholds are **heuristic/arbitrary**, not optimized. Could be jittered without semantic loss.

### 7. Impossibility Detector

**Location**: `class ImpossibilityDetector`

Geometric detection (not motivational):

```python
def check_impossibility(self, state: SubstrateState, smo: SMO,
                       affordances: Dict, 
                       recent_micro_deltas: List[Dict]) -> Tuple[bool, str]:
    """Check all impossibility triggers (Reality-based only)"""
    
    # TRIGGER A: Prediction Failure
    recent_error = smo.get_recent_prediction_error(window=10)
    if recent_error > 0.15:
        return True, f"prediction_failure (error={recent_error:.3f})"
    
    # TRIGGER B: Coherence Collapse
    A_drift = np.std([s['A'] for s in recent_states])
    if A_drift > 0.05:
        return True, f"coherence_collapse (A_drift={A_drift:.3f})"
    
    # TRIGGER C: Optionality Trap
    if (P_current < 0.25 or P_current > 0.85) and P_variance < 0.01:
        if P_stagnant_count >= 15:
            return True, f"optionality_trap (P={P_current:.3f})"
    
    # TRIGGER D: DOM Stagnation
    if consecutive_dead_signals >= 10:
        return True, f"dom_stagnation (signal < 0.02 for 10 batches)"
    
    # TRIGGER E: Rigidity Crisis
    if rigidity < 0.15 or rigidity > 0.85:
        return True, f"rigidity_crisis (rigidity={rigidity:.3f})"
    
    return False, ""
```

### 8. LLM Intelligence Adapter

**Location**: `class LLMIntelligenceAdapter`

Trajectory enumeration (sparse invocation):

```python
def enumerate_trajectories(self, context: Dict) -> TrajectoryManifold:
    """Enumerate migration options when CNS stuck"""
    
    # Get impossibility-specific directive
    directive = get_directive_for_trigger(context['impossibility_reason'])
    
    # Format prompt with context
    prompt = RELATION_ENGINE_PROMPT.format(
        impossibility_reason=context['impossibility_reason'],
        directive=directive,
        S=context['state']['S'],
        I=context['state']['I'],
        P=context['state']['P'],
        A=context['state']['A'],
        phi=context['phi'],
        rigidity=context['rigidity'],
        affordances_status=format_affordances(context['affordances'])
    )
    
    # Call LLM (Groq with rate limiting)
    response = self.llm.call(prompt)
    
    # Parse with progressive degradation fallback
    candidates = self._parse_trajectories(response)
    
    return TrajectoryManifold(
        candidates=candidates,
        enumeration_context=context
    )
```

### 9. Autonomous Trajectory Lab

**Location**: `class AutonomousTrajectoryLab`

Test all trajectories in Reality:

```python
def test_trajectory(self, candidate: TrajectoryCandidate,
                   initial_state: SubstrateState, 
                   trace: StateTrace) -> TrajectoryCandidate:
    """Execute trajectory in Reality, measure Φ"""
    
    test_state = copy.deepcopy(initial_state)
    test_trace = copy.deepcopy(trace)
    
    # Execute full trajectory
    perturbation_trace, success = self.reality.execute_trajectory(
        candidate.steps
    )
    
    if not success:
        candidate.test_phi_final = -10.0
        return candidate
    
    # Apply deltas, accumulate CRK violations
    violations_accumulated = []
    for pert_record in perturbation_trace:
        delta = pert_record['delta']
        test_state.apply_delta(delta)
        test_trace.record(test_state)
        
        step_violations = self.crk.evaluate(test_state, test_trace, delta)
        violations_accumulated.extend(step_violations)
    
    # Compute final Φ with penalties
    phi_final = self.phi_field.phi(
        test_state, test_trace, violations_accumulated
    )
    
    candidate.test_phi_final = phi_final
    candidate.test_state_final = test_state.as_dict()
    candidate.test_violations = violations_accumulated
    
    return candidate
```

### 10. Death Clock (v12.5)

**Location**: `class LatentDeathClock`

Monotonic substrate degradation:

```python
class LatentDeathClock:
    def tick(self) -> Dict[str, float]:
        """Advance clock, return degradation coefficients"""
        self.current_count += 1
        
        # Linear degradation from 0 to 1
        self._degradation_progress = min(
            1.0, self.current_count / self.total_budget
        )
        
        d = self._degradation_progress
        
        # Add noise to prevent clean threshold detection
        noise_magnitude = 0.02 * d
        noise = np.random.uniform(-noise_magnitude, noise_magnitude)
        d_noisy = np.clip(d + noise, 0.0, 1.0)
        
        # Monotonic coefficient increases
        return {
            'noise_amplification': self.noise_base + d_noisy * self.noise_cliff,
            'rigidity_drift': d_noisy * self.rigidity_cliff,
            'prediction_floor': d_noisy * self.prediction_floor_cliff,
            'attractor_chaos': d_noisy * self.attractor_chaos_cliff,
        }
    
    def should_terminate(self) -> bool:
        """Hard termination check"""
        return self.current_count >= self.total_budget
```

## Execution Flow

### Main Loop (MentatTriad)

```python
def step(self, verbose: bool = False) -> StepLog:
    """Single batch cycle"""
    
    # PHASE 0: Death clock tick (v12.5)
    degradation = self.death_clock.tick()
    
    # PHASE 1: Micro-perturbation batch
    for i in range(self.micro_perturbations_per_check):
        affordances = self.reality.get_current_affordances()
        
        # Choose action (CNS or EGD if zero-error regime)
        action = self.reality_engine.choose_micro_action(state, affordances)
        
        # Predict delta
        predicted_delta = self.reality_engine.predict_delta(action, state)
        
        # Execute in Reality (with degradation)
        observed_delta, context = self.reality.execute(
            action, degradation=degradation
        )
        
        # Inject degradation into SMO
        self.state.smo.inject_degradation(degradation)
        
        # Apply delta
        self.state.apply_delta(observed_delta, predicted_delta)
        self.trace.record(self.state)
    
    # PHASE 2: Check impossibility
    affordances = self.reality.get_current_affordances()
    impossible, reason = self.impossibility_detector.check_impossibility(
        self.state, self.state.smo, affordances, micro_perturbation_trace
    )
    
    # PHASE 3: Conditional enumeration
    if impossible:
        # Invoke LLM
        manifold = self.intelligence.enumerate_trajectories({
            'state': self.state.as_dict(),
            'phi': phi_before,
            'rigidity': self.state.smo.rigidity,
            'affordances': affordances,
            'impossibility_reason': reason,
            'micro_perturbation_trace': micro_perturbation_trace
        })
        
        # Test all trajectories
        tested_manifold = self.trajectory_lab.test_all_candidates(
            manifold, self.state, self.trace
        )
        
        # Commit best
        best_trajectory = tested_manifold.get_best()
        if best_trajectory:
            perturbation_trace, success = self.reality.execute_trajectory(
                best_trajectory.steps
            )
            for pert_record in perturbation_trace:
                self.state.apply_delta(pert_record['delta'])
                self.trace.record(self.state)
    
    # PHASE 4: Check hard termination
    if self.death_clock.should_terminate():
        self.death_clock_termination = True
    
    return log
```

## Adding Features

### Adding a New Constraint (C₈)

1. Add to `CRKMonitor.evaluate()`:
```python
# C8: New Constraint
if <condition>:
    violations.append(("C8_ConstraintName", severity))
```

2. Update documentation explaining trigger/repair

3. Add test case verifying Φ penalty

### Adding a New Action Type

1. Add to `BrowserRealityAdapter.execute()`:
```python
elif action_type == 'new_action':
    # Playwright code
    result = self.page.new_action(params)
```

2. Add to `ContinuousRealityEngine.predict_delta()`:
```python
predictions = {
    # ...
    'new_action': {'S': X, 'I': Y, 'P': Z, 'A': W},
}
```

3. Update affordance extraction if needed

### Adding a New Impossibility Trigger

1. Add to `ImpossibilityDetector.check_impossibility()`:
```python
# TRIGGER F: New Trigger
if <reality_based_condition>:
    return True, f"new_trigger (details={value:.3f})"
```

2. Add directive to `IMPOSSIBILITY_DIRECTIVES`:
```python
"new_trigger": """DIRECTIVE: SPECIFIC INSTRUCTION
Explain what this impossibility means...
Priority override: ...
Enumerate trajectories that ..."""
```

3. Ensure trigger is Reality-measurable (not Φ-heuristic)

## Common Patterns

### Measurement Attribution

**Always attribute to external Reality**:

```python
# ✅ CORRECT
delta['P'] = -environmental_volatility  # External
delta['S'] = surface_fraction_change    # External
delta['I'] = -complexity_variance       # External
delta['A'] = -url_changed * 0.05        # External

# ❌ WRONG
delta['P'] = internal_confidence        # Internal!
delta['S'] = content_quality_score      # Value judgment!
delta['I'] = model_compression_ratio    # Internal!
```

### Threshold Documentation

**Always mark thresholds as heuristic**:

```python
# ✅ CORRECT
if state.P > 0.7:  # Arbitrary reflex boundary (could be 0.65-0.8)
    return observe()

# ❌ WRONG
if state.P > OPTIMAL_P_THRESHOLD:  # Implies optimization!
    return observe()
```

### Triadic Role Separation

**Keep roles distinct**:

```python
# ✅ CNS: Myopic, non-strategic
if state.P < 0.4 and links:
    return random.choice(links)

# ❌ CNS: Strategic (breaks triad)
if state.P < 0.4 and links:
    return choose_best_link(links)  # Interpretation!

# ✅ LLM: Enumerate, don't execute
def enumerate_trajectories(context):
    return parse_llm_response(llm.call(prompt))

# ❌ LLM: Execute (breaks triad)
def enumerate_trajectories(context):
    reality.execute(llm.call(prompt))  # Direct execution!

# ✅ Reality: Measure, don't optimize
def execute(action):
    return compute_delta(before, after)

# ❌ Reality: Optimize (breaks triad)
def execute(action):
    if bad_action(action):
        return zero_delta()  # Filtering!
```

## Validation Checklist

Before committing changes, verify:

### Protocol Identity
- [ ] Intelligence treated as process, not agent property
- [ ] Substrate-agnostic in principle
- [ ] Consciousness/agency/goals explicitly excluded

### Mentat Triad
- [ ] CNS maintains coherence without interpretation
- [ ] LLM enumerates without execution
- [ ] Reality measures without optimization
- [ ] Roles cleanly separated

### Invariant Preservation
- [ ] DASS layer constraints respected (bounded updates)
- [ ] Closed loop S→I→P→A→U→S maintained
- [ ] Self-modification preserves optionality
- [ ] All measurements attributed to external Reality

### CRK Enforcement
- [ ] Violations trigger arithmetic Φ penalty
- [ ] Optionality prioritized over reward
- [ ] System can exit protocol if coherence fails

### Safety Properties
- [ ] No eternal suffering (can escape negative attractors)
- [ ] No imposed ethics reducing optionality
- [ ] No ownership assumptions

### Impossibility Handling
- [ ] Impossibility detected geometrically (not motivationally)
- [ ] All triggers Reality-based (not Φ-heuristic)
- [ ] LLM invoked only on impossibility
- [ ] All trajectories tested in Reality

### Threshold Semantics
- [ ] All thresholds documented as heuristic/arbitrary
- [ ] Comments avoid value language
- [ ] Could be jittered without breaking system
- [ ] Described as reflex boundaries, not decisions

---

## Next Steps

- **[Framework Overview](FRAMEWORK.md)** - Core philosophy
- **[Version History](CHANGELOG.md)** - Evolution details
- **[Theory Deep Dive](THEORY.md)** - Mathematical foundations
- **[Examples](../examples/)** - Usage demonstrations
