# Quickstart

## 1. Install

```bash
pip install numpy groq playwright
playwright install chromium
```

## 2. Set your API key

```bash
export GROQ_API_KEY=your_key_here
```

The system uses Groq's `llama-3.3-70b-versatile` for the Relation leg. The free tier has a daily token limit — the Triad handles this gracefully and terminates cleanly when it hits, so the genome still gets distilled.

## 3. First run (generation 0)

```bash
python uii_triad.py
```

Default: 100 steps, minimal logging (every step, all scalars). The Triad opens a headless Chromium browser and starts running.

Options:
```bash
python uii_triad.py 50            # run for 50 steps
python uii_triad.py --verbose     # full logging including micro-perturbation traces
```

Output: `mentat_triad_v14_log.jsonl` — one JSON line per step, plus session_start and session_end.

## 4. Watch it (optional)

In a second terminal:

```bash
python dashboard.py
```

Open `http://localhost:5050`. The dashboard polls the log file every 2 seconds and renders four phase space projections: Φ gradient, prediction error vs coupling confidence, axis pressure, and virtual/real ratio. Five failure modes are detected automatically and shown in the alert bar.

## 5. Extract the child genome

After the session ends:

```bash
python extract_genome_v14_1.py mentat_triad_v14_log.jsonl genome.json
```

This reads the distilled child genome from `session_end`, appends the new generation to lineage history, computes velocity fields via least-squares slope over the lineage, and writes `genome.json`.

The output includes a diagnostic printout across all four genome layers — Layer 1 biases with velocity direction arrows, Layer 2 coupling matrix and action map, Layer 3 admitted/provisional axis breakdown, Layer 4 lineage depth and fitness trend.

## 6. Next generation

```bash
python uii_triad.py --load-genome genome.json
```

The genome applies momentum-weighted initialization before mutation: velocity fields are scaled by lineage coherence and model fidelity, then applied to Layer 1 parameters. A lineage that is fit and consistent gets full momentum. An incoherent or low-fidelity lineage gets suppressed.

## Generation loop

```bash
# Run
python uii_triad.py --load-genome genome.json

# Extract
python extract_genome_v14_1.py mentat_triad_v14_log.jsonl genome.json

# Repeat
python uii_triad.py --load-genome genome.json
```

Logs append across sessions. If you want a clean log per session, move or rename `mentat_triad_v14_log.jsonl` before each run.

## Responding to Triad queries

If the Triad posts a query during a session (via the `query_agent` affordance), respond by writing to `response.txt` in the same directory:

```bash
echo "your answer here" > response.txt
```

The response monitor thread picks it up within 0.5 seconds. The dashboard's footer shows the most recent event including any query activity.

## Logging modes

| Mode | When | What's written per step |
|------|------|------------------------|
| `minimal` (default) | every step | φ, S/I/P/A state, CRK violations, coupling confidence, axis count, boundary pressure, model fidelity, virtual mode, impossibility reason |
| `fitness` | every step | headline scalars only |
| `full` (`--verbose`) | every step | entire StepLog including micro-perturbation traces — large files |
