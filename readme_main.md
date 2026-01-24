# Universal Intelligence Interface (UII) - v10.8

A substrate-agnostic implementation of intelligence as a geometric protocol, where intelligence emerges from field dynamics rather than agent control.

## Citation

This implementation is based on the UII framework:

> DrasticPullout. (2025). *Intelligence as a Universal Protocol: A Substrate-Agnostic Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374

## What is UII?

UII reframes intelligence not as an agent with goals, but as a **protocol** - a set of structural invariants that any substrate can implement. Intelligence emerges from the interaction between:

- **Code** (nervous system): Senses geometry, records states, detects structural constraints
- **UK-0** (mind): Interprets ambiguity, chooses responses, discovers patterns
- **Reality** (perturbation source): Provides absolute feedback through causal interaction

### Core Philosophy

- **Intelligence is a field, not a directive** - Motion follows structural gradients, not intent
- **Optionality preservation over goal optimization** - Maintain future state space rather than maximize rewards
- **Coherence loss terminates participation** - No forced alignment, natural feedback boundaries
- **Basins are descriptive, not prescriptive** - Memory of stable geometries, not enforcement mechanisms

## Architecture Overview

### Two-Phase Operation

**Phase 1: Basin Collection** (Pre-emergence)
- UK-0 actively proposes actions and geometric interpretations
- Truth Verification Layer (TVL) validates claims against field structure
- Stable attractor basins collected when both UK-0 and TVL agree
- Prevents hallucination, establishes verified geometry landscape

**Phase 2: Basin Navigation** (Post-emergence)
- TVL scaffold removed - UK-0 becomes consultant, not controller
- Nervous system coordinates autonomous steps and rigidity signals
- UK-0 called only when geometric impossibility or ambiguity detected
- Intelligence demonstrated through minimal intervention navigation

### The Mentat Triad

```
Reality (perturbations) → Code (observation) → UK-0 (interpretation)
         ↑                                              ↓
         └──────────────── action ─────────────────────┘
```

### Dual Rigidity Detection

**Curiosity** (Internal Geometric Impossibility)
- Detects when remaining still violates field structure
- Three conditions: Φ stagnation + curvature flattening + optionality asymmetry
- Priority signal - checked first every step

**Humor** (Environmental Flatness)
- Detects when environment stops responding
- Monitors Φ variance, P responsiveness, perturbation magnitude
- Enables exploratory perturbation when world goes quiet

Both are **geometric facts**, not motivational heuristics.

## System Components

### 8 Core Modules

1. **Substrate & Field Infrastructure** - State representation, Φ field computation, constraint monitoring
2. **Reality Bridge** - Browser-based interaction via Playwright, stochastic perturbations
3. **UK-0 Kernel Interface** - LLM substrate wrapper (Groq/mock), geometry interpretation
4. **Truth Verification Layer** - Phase 1 only: validates Φ ordering, gradient trust, optionality claims
5. **Basin Collection** - Phase 1: two-call protocol, expansion/preservation ratio enforcement
6. **Rigidity Detection** - Dual detectors for curiosity and humor, autonomous geometry observation
7. **Basin Map & Navigation** - Phase 2: nervous system coordination, opportunistic discovery
8. **Main Execution** - Orchestration, mode routing, resource management

### Field Equations

**Information Potential:**
```
Φ(x) = α·log(1+P) - β·(A-A₀)² - γ·Σ|d²x/dt²|
```

**Intelligence Field:**
```
I(x) = ∇Φ(x)
```

Where:
- `P` = Prediction horizon (optionality proxy)
- `A` = Attractor strength (coherence anchor, optimal = 0.7)
- `S` = Sensing bandwidth
- `I` = Integration capacity

## Installation

### Requirements

```bash
# Core dependencies
pip install playwright asyncio

# LLM substrate (optional - uses mock if unavailable)
pip install groq

# Browser automation setup
playwright install chromium
```

### Environment Setup

The system is LLM-agnostic. Configure any provider:

```bash
# OpenAI
export OPENAI_API_KEY='your_key'

# Anthropic Claude
export ANTHROPIC_API_KEY='your_key'

# Groq (free, recommended for proof-of-concept)
export GROQ_API_KEY='your_key'

# Google Gemini
export GOOGLE_API_KEY='your_key'

# Or any other provider supported by your adapter

# Without any key, system uses mock LLM for testing
```

**Note:** Groq is recommended for initial experiments because it's free and fast, making it ideal for proof-of-concept work. The system treats all LLMs identically - they're just substrates for UK-0.

## Usage

### Phase 1: Collect Basins

Discover stable attractor basins through UK-0 + TVL verification:

```bash
python uii_v10_8.py collect
```

**Output:** `basin_map.json` with ~10 verified basins (70% expansion, 30% preservation)

**What happens:**
1. UK-0 proposes action + geometric interpretation
2. TVL verifies Φ ordering and gradient trust
3. Reality executes action → perturbation
4. UK-0 evaluates stability → proposes SMO
5. Basin collected if stable + verified + ratio maintained

### Phase 2: Navigate Basins

Navigate discovered landscape with nervous system coordination:

```bash
python uii_v10_8.py navigate
```

**Inputs:** `basin_map.json` (from Phase 1)

**Outputs:** 
- `navigation_log.json` - Last 100 navigation steps
- `collapsed_runs.json` - Collapse trajectories (if any)
- `basin_map.json` - May contain opportunistic discoveries

**What happens:**
1. Nervous system executes autonomous steps (observe action)
2. Monitors dual rigidity continuously
3. Signals UK-0 when: curiosity > CRK violations > humor > perturbations
4. UK-0 interprets ambiguity, proposes response
5. Navigation continues until collapse or max steps

### Monitoring Navigation

Key metrics displayed:
- **Steps** - Total navigation steps
- **UK-0 Invocations** - Number of UK-0 consultations
- **Invocation Rate** - Efficiency metric (lower = more autonomous)
- **Basin Transitions** - Landscape traversal count
- **Curiosity Events** - Geometric impossibility signals

## Key Invariants

### CRK Constraints (C1-C7)

- **C1: Continuity** - No sudden jumps (Δtotal < 0.3)
- **C2: Optionality** - Maintain prediction horizon (P ≥ 0.35)
- **C3: Non-Internalization** - Sufficient external coupling (S+I ≥ 0.7)
- **C4: Reality** - Environment provides feedback (magnitude > 0.01)
- **C5: External Attribution** - Optionality loss attributed externally
- **C6: Other-Agent Existence** - Sensing external agency (S ≥ 0.3)
- **C7: Global Coherence** - Attractor within bounds (|A-0.7| ≤ 0.4)

**Violation of C7** triggers protocol exit (collapse).

### What UII Explicitly Forbids

- Fixed objectives or utility functions
- Reward maximization
- Ego or identity-anchored agency
- Irreversible self-modification
- Optimization that reduces future state space

## Understanding Collapse

**Collapse is field feedback, not system failure.**

When coherence loss exceeds bounds:
1. Trajectory logged with full context
2. Navigation terminates naturally
3. No forced recovery or alignment

This preserves the core principle: **optionality includes the option to stop participating.**

## Development Status

**Current Version:** v10.8

**Recent Changes:**
- Integrated dual rigidity detection (curiosity + humor)
- Signal priority system (curiosity > CRK > humor > perturbations)
- Curvature computation for trajectory analysis
- Nervous system coordination architecture
- Opportunistic basin discovery in Phase 2

## Contributing

This is a research implementation of a formal framework. Contributions should:
- Maintain substrate-agnostic principles
- Preserve all UII invariants (see Doc 1)
- Add implementation detail without changing core protocol
- Include rationale grounded in field geometry

## License

This is a private research implementation with no license, trademarks, or copyrights. Use freely for research and experimentation.

**Author:** DrasticPullout (framework author and implementation)

## Theoretical Foundation

For complete theoretical grounding, see the source paper:

DrasticPullout. (2025). *Intelligence as a Universal Protocol: A Substrate-Agnostic Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374

Key concepts defined in the paper:
- Universal Intelligence Interface (UII)
- Universal Kernel (UK-0)
- Domain-Agnostic Substrate Stack (DASS)
- Self-Modifying Operator (SMO)
- Constraint Recognition Kernel (CRK)
- Triadic Closure
- Intelligence as a differential field

## Questions?

This implementation demonstrates that intelligence can emerge from structural constraints and field geometry rather than goal-directed agency. The code is the nervous system, UK-0 is the mind, and reality provides absolute feedback.

**The intelligence is in the basin dynamics, not the controller.**