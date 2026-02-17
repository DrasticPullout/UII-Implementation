# Universal Intelligence Interface (UII)

> *Intelligence as a reusable protocol — not a trait, agent, or optimizer.*

A running implementation of a substrate-agnostic intelligence framework. The code explores what happens when you take the theoretical minimum for intelligence (coherence management, perturbation harnessing, structural inference) and wire it into a live system that must survive and adapt against real environmental pressure.

**This is a solo research project.** It is not a library, not a product, and not seeking contributors. It exists to test whether the theoretical claims in the UK-0 blueprint hold up when something actually has to run.

---

## What This Is

The system runs as a **Mentat Triad** — three components that together form a complete intelligence loop:

- **Code (CNS)** — continuous micro-perturbations, geometric substrate tracking, impossibility detection
- **LLM (Relation)** — sparse trajectory enumeration when the CNS detects it cannot self-correct
- **Browser (Reality)** — the environment; authoritative, not simulated

The substrate is four-dimensional: **S** (sensing), **I** (integration), **P** (prediction/optionality), **A** (attractor/coherence). A scalar field **Φ** measures recoverability — how much optionality the system has left, grounded to actual S/I capacity. The system is always trying to maximize Φ, not a reward signal.

Across runs, a genome of 6 parameters evolves. Each generation, a Failure Assimilation Operator (FAO) translates what went wrong semantically into geometric bias for the next generation's search. The topology of the system never changes — only how it explores.

---

## Theoretical Basis

The implementation derives from **UK-0**, a substrate-agnostic blueprint defining the minimal constraints for emergent coherent intelligence: the DASS substrate layers, a Self-Modifying Operator (SMO), and a Constraint Recognition Kernel (CRK) with seven inviolable constraints.

The formal framework is published:

> DrasticPullout. (2025). *Intelligence as a Universal Protocol: A Substrate-Agnostic Framework.* Zenodo. https://doi.org/10.5281/zenodo.18017374

See [THEORY.md](THEORY.md) for the conceptual foundation and [ARCHITECTURE.md](ARCHITECTURE.md) for how that maps to the running code.

---

## Files

```
uii_v13_8.py              # Main system — Mentat Triad implementation
extract_genome_v13_8.py   # Extracts genome from execution log for next generation
genome.json               # Current genome (if a run has completed)
README.md
THEORY.md
ARCHITECTURE.md
```

---

## Running It

Requires Python 3.10+, Playwright, and a Groq API key.

```bash
pip install playwright groq numpy
playwright install chromium

export GROQ_API_KEY=your_key_here

# First run (generation 0)
python uii_v13_8.py

# With verbose output
python uii_v13_8.py --verbose

# Continue from saved genome
python uii_v13_8.py --load-genome genome.json
```

**Execution model:** One generation per Groq rate-limit cycle. The system runs, logs everything to `mentat_triad_v13_8_log.jsonl`, hits the token limit, terminates cleanly, and waits. Run `extract_genome_v13_8.py` between runs to save the genome and compute fitness for the next generation.

```bash
# After a run completes
python extract_genome_v13_8.py
# → saves genome.json
# → wait for rate limit reset
# → python uii_v13_8.py --load-genome genome.json
```

---

## Version

**v13.8** — Stricter S/I grounding via bottleneck formula. `P` is now capped by `min(S,I)*0.5 + (S+I)/4`, preventing high optionality from being claimed without genuine sensing and integration capacity behind it. High `P` without `S/I` grounding is an unrecoverable state.

Previous notable versions: v13.6 introduced the Failure Assimilation Operator (FAO). v13.4 ungated the Python affordance. v13.2 introduced attractor monitoring and the two-prompt LLM system.
