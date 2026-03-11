# Universal Intelligence Interface (UII)
**v16.0 — Phase 11: Geometric Dynamics Era**

> DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic Framework.* Zenodo. https://doi.org/10.5281/zenodo.18017374

---

## What this is

UII is a framework that defines intelligence as a dynamical protocol rather than an agent property. Not a neural net, not a reward maximizer, not a cognitive architecture in any familiar sense.

The core claim: a system is behaving intelligently when it preserves internal coherence while converting environmental perturbations into improved structural representation — and this can happen on any causal substrate. No goals, no identity, no externally imposed optimization targets.

The whole thing is governed by one equation:

```
ẋ = P_M( G⁻¹(x) · ∇Φ(x) )
```

- `Φ(x)` is a geometric potential field over the system's state space — not a reward function
- `G(x) = H(x)` is the Hessian of Φ, which serves as the information metric
- `G⁻¹ · ∇Φ` is natural gradient ascent on Φ — information-geometry-optimal
- `P_M` projects the motion onto the triadic constraint manifold `M = {x : T(x) = 0}`

The system is the equation. Every module in this codebase maps to one of those terms.

---

## Structure

```
uii_geometry.py     SubstrateState, PhiField (Hessian + action scoring),
                    CRKMonitor, SymbolGroundingAdapter, eigen_decompose(),
                    expected_optionality_gain()

uii_operators.py    DASS operators: Sensing, Compression, Prediction,
                    Coherence, SelfModifying

uii_ledger.py       TriadLedger, PeakOptimalityTracker, load/save_ledger()

uii_fao.py          ResidualTracker, AxisAdmissionTest,
                    FailureAssimilationOperator

uii_reality.py      BrowserRealityAdapter, CouplingMatrixEstimator,
                    AttractorMonitor

uii_triad.py        MentatTriad — the 8-phase step loop and run()

extract_ledger.py   Diagnostic inspection tool (not a pipeline step)
```

---

## Running it

```bash
# Run a session
python uii_triad.py

# Load an existing ledger and continue
python uii_triad.py --load-ledger ledger.json

# Inspect ledger state between sessions
python extract_ledger.py --ledger ledger.json --validate
```

The ledger (`ledger.json`) is written automatically at session end. `extract_ledger.py` is just for inspection — you don't need it in the pipeline.

---

## The ledger

State persists across sessions via a four-field ledger, not a genome or trajectory memory:

| Field | Written by | When |
|---|---|---|
| `hessian_snapshot` | PeakOptionalityTracker | At peak Vol_opt during run |
| `operator_snapshot` | PeakOptionalityTracker | At peak Vol_opt during run |
| `causal_model` | FAO.distill_to_ledger() | Session end (always) |
| `discovered_structure` | FAO.distill_to_ledger() | Session end (always) |

A new run loading a ledger converges on the same basin because the basin exists in the field — not in any particular instance.

---

## What it isn't

- Not a language model wrapper
- Not a reinforcement learning agent  
- Not trying to simulate human cognition
- Not built around any notion of consciousness, goals, or identity

The math doesn't require any of that. Neither does the implementation.
