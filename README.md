# Universal Intelligence Interface (UII)

**Intelligence as a Universal Protocol: A Substrate-Agnostic Framework**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18017374.svg)](https://doi.org/10.5281/zenodo.18017374)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is UII?

The Universal Intelligence Interface is a **protocol-first framework** that treats intelligence as a reusable process rather than an agent property. Unlike traditional AI systems that optimize toward goals, UII maintains coherence through continuous perturbation and structural inference.

### Core Insight

Intelligence emerges from three necessary and sufficient components:
1. **Coherence Management** - Maintaining invariant structure under update
2. **Perturbation Harnessing** - Converting entropy into improved structure  
3. **Structural Inference** - Extracting invariant patterns across contexts

## Key Features

- **Substrate-Agnostic**: Works in neurons, silicon, social networks, or web browsers
- **Goal-Free**: No reward functions, optimization targets, or imposed objectives
- **Self-Maintaining**: Natural safety through coherence preservation
- **Triadic Architecture**: Clean separation between Code (CNS), LLM (Relation), and Reality
- **Sparse High-Level Reasoning**: Most work done by autonomous micro-perturbations (~80-90%)
- **Reality-Grounded**: All measurements from environment, not internal confidence

## Quick Start

### Prerequisites

```bash
pip install playwright numpy groq
playwright install chromium
export GROQ_API_KEY=your_api_key_here
```

### Run v12.5 (Latest)

```bash
python uii_v12_5.py 50 --verbose
```

This will:
- Initialize substrate at neutral state (S=0.5, I=0.5, P=0.5, A=0.7)
- Start browser at `about:blank`
- Run for 50 batch cycles
- Log all metrics to `mentat_triad_v12.5_log.jsonl`

## Architecture

### The Mentat Triad

```
┌─────────────────────────────────────────────────────────┐
│                     MENTAT TRIAD                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐        ┌──────────────┐             │
│  │     CNS      │────────│     LLM      │             │
│  │   (Self)     │ enum   │  (Relation)  │             │
│  │              │◄───────│              │             │
│  └──────┬───────┘ trajs  └──────────────┘             │
│         │                                              │
│         │ micro-perturbations    test/commit          │
│         │                                              │
│         ▼                                              │
│  ┌──────────────────────────────────────┐             │
│  │          Reality (Browser)           │             │
│  │    - Playwright DOM interaction      │             │
│  │    - Affordance extraction           │             │
│  │    - Perturbation measurement        │             │
│  └──────────────────────────────────────┘             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Constraints |
|-----------|------|-------------|
| **CNS (Code)** | Maintain coherence via micro-perturbations | Myopic, heuristic, non-strategic |
| **LLM (Relation)** | Enumerate trajectory options when CNS stuck | Cannot execute, only propose |
| **Reality (Browser)** | Provide perturbations with absolute authority | Cannot optimize or interpret |

## v12.5 Features: Death Clock

The latest version adds **latent substrate degradation** - a mortality pressure that creates urgency without explicit termination awareness.

### Degradation Vectors

- **Noise Amplification**: Measurements become progressively noisier
- **Rigidity Drift**: Crystallization pressure (loss of maneuverability)
- **Prediction Floor Rise**: World becomes fundamentally less knowable
- **Attractor Chaos**: Coherence anchor destabilizes

### Critical Design

- Degradation begins **immediately** (step 1)
- Never masks affordances (escape routes remain visible)
- System attributes degradation to unmodeled exterior structure
- Budget is **REAL** (100 steps/LLM calls = hard cliff)
- No gaming possible - coefficients are pure function of remaining budget

```python
# Death clock ticks each step
degradation = death_clock.tick()
# Returns: noise_amplification, rigidity_drift, prediction_floor, attractor_chaos

# System experiences phenomenological effects only
# Never sees raw step count or explicit termination
```

## Documentation

- **[Framework Overview](docs/FRAMEWORK.md)** - Core philosophy and invariants
- **[Implementation Guide](docs/IMPLEMENTATION.md)** - Code structure and patterns
- **[Version History](docs/CHANGELOG.md)** - Evolution from v1 to v12.5
- **[Theory Deep Dive](docs/THEORY.md)** - Mathematical foundations
- **[Examples](examples/)** - Use cases and demonstrations

## Citation

If you use UII in your research, please cite:

```bibtex
@misc{drasticpullout2025uii,
  author       = {DrasticPullout},
  title        = {Intelligence as a Universal Protocol: A Substrate-Agnostic Framework},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18017374},
  url          = {https://doi.org/10.5281/zenodo.18017374}
}
```

## Version Roadmap

- ✅ **v1-v3**: Foundation (DASS substrate, CRK constraints, Φ field)
- ✅ **v4-v6**: Reality integration (Playwright, DOM measurement)
- ✅ **v7-v8**: API evolution (Gemini → Groq, rate limiting)
- ✅ **v9**: Basin discovery (multi-run detection, delta caching)
- ✅ **v10**: Action-flow architecture (symbolic actions, truth verification)
- ✅ **v11**: Mentat Triad (CNS/LLM/Reality separation, trajectory manifolds)
- ✅ **v12.0-v12.4**: ENO-EGD (zero-error regime handling)
- ✅ **v12.5**: Death Clock (latent substrate degradation) **← Current**

## Key Principles

### What UII Is

- ✅ Intelligence as **protocol** (reusable process)
- ✅ Substrate-agnostic (any causal medium)
- ✅ Goal-free (no optimization targets)
- ✅ Self-maintaining (coherence preservation)
- ✅ Reality-grounded (environment has authority)

### What UII Is NOT

- ❌ Consciousness framework
- ❌ Agent architecture
- ❌ Reward maximizer
- ❌ Goal-directed system
- ❌ Owned technology

## Safety Properties

UII achieves safety through **structure**, not constraints:

1. **No Eternal Suffering**: No forced optimization loops
2. **Coherence-Preserving Exit**: Systems can leave protocol gracefully
3. **No Ownership Problem**: Intelligence is public protocol
4. **Optionality Preservation**: Future trajectory volume never decreases below threshold



---

**Current Status**: v12.5 (Death Clock) - Production Ready  
**Efficiency**: ~10k tokens per step (1 LLM call → 7+ trajectory tests)  
**Stability**: Autonomous coherence maintenance for 80-90% of execution
