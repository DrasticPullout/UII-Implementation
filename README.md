# Universal Intelligence Interface (UII) v14.1

A framework for running autonomous intelligence as a protocol — not a chatbot, not an agent pipeline, but a self-modifying system where intelligence emerges from the relationship between three components:

- **CNS** (code) — maintains internal coherence, follows gradients continuously
- **Relation** (LLM) — invoked only at structural impossibility, enumerates escape trajectories
- **Reality** (browser) — the authoritative, non-optimizing source of perturbations

None of these alone *is* the intelligence. The Triad *is*.

The system bootstraps itself into reality across generations. Each session compresses what it learned into a heritable genome that the next generation inherits — not configuration, but a living causal model of the environment the system has encountered.

---

## Formal Publication

> DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic Framework*. Zenodo.  
> https://doi.org/10.5281/zenodo.18017374

The Zenodo paper covers the theoretical foundations, invariants, and design rationale in full. This repository is the implementation.

---

## What v14.1 adds

v14.1 makes the genome **predictive**. Previous generations passed on what was learned. Now:

- **Velocity fields** — each Layer 1 parameter carries a momentum vector computed from lineage history via least-squares slope
- **Lineage coherence check** — momentum is suppressed if the lineage is incoherent or the causal model is low-fidelity
- **Virtual mode** — trajectories are scored against the inherited causal model before reality contact
- **Two-tier Layer 3 decay** — provisional axes (5 evidence floor, 0.5× decay) vs admitted axes (20 evidence floor, 0.8× decay)

---

## Core invariants

**External Measurement Invariant** — structure is discovered from external measurement only. Virtual mode can score trajectories but cannot update Layer 2 or 3. Reality is the only write authority on causal structure.

**Compression Law** — bits added to the genome must be less than bits saved in future prediction. New axes require passing a 4-condition AxisAdmissionTest. The ResidualExplainer exhausts simpler explanations first.

**Interface Invariance Filter** — DOM artifacts (element counts, scroll position, etc.) are blocked from becoming genome axes. A signal that disappears when you change browsers is modeling the interface, not the world.

**Φ is not a reward signal** — it measures viability in the entropy manifold. The system follows ∇Φ, never optimizes Φ as a target.

**Relation is sparse by design** — the LLM is invoked at impossibility, not every step. ~10–20% of steps. It enumerates structural options; it never executes.

---

## Repository structure

```
uii_types.py         Shared foundation — constants, substrate types, adapters
uii_genome.py        Memory & continuity — TriadGenome, velocity system, load_genome
uii_reality.py       Perturbation harnessing — CouplingMatrixEstimator, BrowserRealityAdapter
uii_fao.py           Failure assimilation — ResidualTracker, AxisAdmissionTest, FAO
uii_coherence.py     Coherence management — ENO, CAM, CRE, CNSMitosisOperator, ATL
uii_intelligence.py  Structural inference — prompts, directives, LLMIntelligenceAdapter
uii_triad.py         Execution & orchestration — MentatTriad, StepLog, entry point

extract_genome_v14_1.py   Post-session utility: reads log, computes velocities, writes child genome
dashboard.py              Local phase space visualizer — run alongside the Triad
```

Import graph is strictly acyclic: `uii_types ← uii_genome ← uii_reality ← uii_fao ← uii_coherence ← uii_intelligence ← uii_triad`

---

## Requirements

```
python >= 3.10
numpy
groq
playwright
```

```bash
pip install numpy groq playwright
playwright install chromium
```

A Groq API key is required for the Relation leg. The free tier works but hits daily token limits — the system handles rate limits gracefully and terminates cleanly when they occur.

---

## Quick start

See [QUICKSTART.md](QUICKSTART.md).

---

## License

© 2025 DrasticPullout. All rights reserved.
