# Universal Intelligence Interface (UII)

Personal implementation of the Universal Intelligence Interface framework — a substrate-agnostic protocol for intelligence defined by geometric potential fields, triadic closure, and heritable causal structure across generations.

Formal framework: DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374

This repository is not a library. It is not production software. It is a working research implementation for personal use.

---

## Mathematical Spine

### State Space

System state is a four-component vector:

```
x = (S, I, P, A)
```

where `S` = sensing coverage, `I` = compression quality, `P` = viable future volume, `A` = attractor proximity. Each component lies in `[0, 1]`. The system evolves along a trajectory `x(t) ∈ X` with local velocity `ẋ = dx/dt`.

### Structure Potential Field

System dynamics are governed by a scalar potential:

```
Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)
```

where:
- `C(x)` — compression quality of the causal graph
- `O(x)` — viable future volume from prediction covariance eigenspectrum
- `K(x)` — attractor proximity drag relative to inherited structure

Φ is not a reward function. It is a geometric field that organizes trajectories.

### Intelligence Flow

```
I(x) = ∇Φ(x)
```

The intelligence field is the gradient of Φ. System trajectories evolve by aligning with this field.

### Local Coherence

```
C_local = ⟨∇Φ(x), ẋ⟩ / (‖∇Φ(x)‖ · ‖ẋ‖)
```

`C_local > 0` → aligned with structural improvement. `C_local < 0` → degradation. Global coherence is the time average of `C_local`. A system is coherent when `C_global > 0`.

### Perturbation Stability

Local stability is determined by the Hessian of Φ:

```
A(x) = ∇²Φ(x)
δ²Φ = ½ δxᵀ A(x) δx
```

`δ²Φ > 0` → perturbation increases structure (stable). Intelligent systems operate in regions where the Hessian is positive semi-definite.

### Triadic Closure

Intelligence emerges from coupling three transformations:

```
f_self(x),  f_env(x),  f_rel(x_self, x_env)
```

Closure condition:

```
T(x) = f_rel(f_self(x), f_env(x))
```

The system remains coherent only when this relational mapping holds across state transitions.

### Minimal Substrate Loop

Any compliant implementation must instantiate:

```
S → I → P → A → SMO → S
```

The SMO (Self-Modifying Operator), gated by the Constraint Recognition Kernel, modifies the operators themselves before returning to sensing. This is what distinguishes the loop from a fixed pipeline — the substrate reconfigures under structural pressure.

### Core Invariant

The system remains within the intelligence protocol when:

```
C_global > 0
δ²Φ ≥ 0
T(x) preserved
```

Under these conditions, perturbations increase structure without coherence collapse.
