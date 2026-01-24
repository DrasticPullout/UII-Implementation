# UII v10.8 Quickstart Guide

Get UII running in 5 minutes.

## Prerequisites

- Python 3.8+
- pip
- 10 minutes for basin collection
- Optional: Groq API key (for real LLM, otherwise uses mock)

## Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd uii-v10.8
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install playwright

# Browser automation
playwright install chromium

# Optional: LLM substrate
pip install groq
```

### 3. Set Up Environment (Optional)

```bash
# Recommended: Groq (free, fast, good for proof-of-concept)
export GROQ_API_KEY='your_groq_api_key'

# OR any other LLM provider - see LLM_ADAPTERS.md for options:
# OpenAI, Anthropic, Google Gemini, Ollama (local), etc.

# Skip this to use mock LLM for testing (no API needed)
```

**Why Groq?** Free, fast inference, perfect for proving the concept. The system is LLM-agnostic - if intelligence emerges with a free model, it proves the intelligence is in the field dynamics, not the LLM.

## Running UII

### Phase 1: Collect Basins (5-10 minutes)

Discover stable attractor basins with UK-0 + TVL verification:

```bash
python uii_v10_8.py collect
```

**Expected Output:**
```
=== Basin Collection Complete ===
Collected basins: 10
├─ Expansion basins: 7 (70.0%)
├─ Preservation basins: 3 (30.0%)
├─ Truth rejections: 12
└─ Attempts: 43

Basin map saved to: basin_map.json
```

**What This Does:**
- UK-0 proposes actions and geometric interpretations
- TVL verifies claims against field structure
- Stable basins collected when both agree
- Creates `basin_map.json` foundation

### Phase 2: Navigate Basins (1-2 minutes)

Navigate the discovered landscape:

```bash
python uii_v10_8.py navigate
```

**Expected Output:**
```
=== Navigation Complete ===
Steps: 500
UK-0 Invocations: 47 (9.4%)
Basin Transitions: 15
Basins Traversed: [0, 3, 7, 2, 9, 1, 5, 8, 4, 6]
Curiosity Events: 11

Navigation log saved to: navigation_log.json
```

**What This Does:**
- Nervous system coordinates autonomous steps
- Detects rigidity (curiosity + humor)
- Signals UK-0 only when needed
- Demonstrates minimal-intervention intelligence

## Understanding the Output

### basin_map.json

Contains verified attractor basins:

```json
{
  "basins": [
    {
      "id": 0,
      "basin_type": "expansion",
      "P_change": 0.05,           // Optionality increase
      "phi_target": 1.23,         // Information potential
      "stability_radius": 0.08,   // Basin width
      "strength": "strong"        // |T| classification
    }
  ]
}
```

**Key Fields:**
- `basin_type` - "expansion" (ΔP > 0.02) or "preservation"
- `P_change` - Optionality delta
- `strength` - Basin stability ("strong", "medium", "weak")

### navigation_log.json

Last 100 navigation steps:

```json
{
  "steps": [
    {
      "step": 0,
      "state": {"S": 0.5, "I": 0.5, "P": 0.5, "A": 0.7},
      "phi": 0.92,
      "signal": "informational_impossibility",
      "uk0_called": true
    }
  ],
  "metrics": {
    "invocation_rate": 0.094,  // Lower = more autonomous
    "curiosity_events": 11
  }
}
```

### collapsed_runs.json (if collapse occurs)

Records natural protocol exits:

```json
{
  "collapses": [
    {
      "cause": "C7_violation",
      "basins_traversed": [0, 3, 7],
      "trajectory": [...]  // Last 20 steps
    }
  ]
}
```

## Interpreting Results

### Good Basin Collection

✅ **10 basins collected**
✅ **Expansion ratio ~70%**
✅ **Low truth rejection rate (<30%)**
✅ **Mix of strong/medium/weak basins**

🚩 **Watch for:**
- High rejection rate (>50%) - UK-0 hallucinating
- All weak basins - unstable landscape
- Ratio far from 70/30 - biased exploration

### Good Navigation

✅ **Invocation rate <10%**
✅ **Multiple basin transitions**
✅ **Some curiosity events**
✅ **No collapse or natural collapse**

🚩 **Watch for:**
- Invocation rate >20% - over-reliance on UK-0
- No transitions - stuck in one basin
- Immediate collapse - configuration issue

## Common Issues

### Issue: "GROQ_API_KEY not found"

**Solution:** Either:
```bash
# Option 1: Set Groq API key (recommended for proof-of-concept)
export GROQ_API_KEY='your_key'

# Option 2: Use different LLM provider (see LLM_ADAPTERS.md)
export OPENAI_API_KEY='your_key'
export LLM_PROVIDER='openai'

# Option 3: Use mock (for architecture testing, no API needed)
# Just run without setting any key - system auto-falls back
```

### Issue: "Playwright not installed"

**Solution:**
```bash
playwright install chromium
```

### Issue: "High truth rejection rate"

**Meaning:** UK-0 making poor gradient predictions

**Solutions:**
- Increase temperature variation in LLM config
- Check if mock adapter is being used unintentionally
- Review basin_map.json for quality

### Issue: "Immediate collapse in navigation"

**Possible Causes:**
1. No basin_map.json from Phase 1
2. Corrupted basin map
3. Initial state outside coherence bounds

**Solutions:**
```bash
# Re-run basin collection
python uii_v10_8.py collect

# Check basin_map.json exists and is valid JSON
```

## Next Steps

### Experiment with Parameters

Edit `uii_v10_8.py` to adjust:

**Basin Collection:**
```python
TARGET_BASINS = 10        # How many basins to collect
MAX_ATTEMPTS = 200        # Maximum collection attempts
expansion_target = 0.7    # Expansion/preservation ratio
```

**Navigation:**
```python
max_steps = 500           # Navigation duration
humor_enabled = True      # Enable humor perturbations
```

### Analyze Results

```bash
# View basin landscape
cat basin_map.json | jq '.basins[] | {id, type: .basin_type, P: .P_change}'

# View navigation efficiency
cat navigation_log.json | jq '.metrics'

# Check for collapses
cat collapsed_runs.json | jq '.collapses[] | .cause'
```

### Modify Rigidity Detection

```python
# In StructuralCuriosityDetector
phi_variance_threshold = 0.001      # Stagnation sensitivity
curvature_threshold = 0.01          # Flattening sensitivity

# In EnvironmentalRigidityDetector
phi_percentile = 0.25               # Flatness percentile
```

## Understanding the Metrics

### Invocation Rate

**What it measures:** Fraction of steps requiring UK-0 interpretation

```
invocation_rate = uk0_calls / total_steps
```

**Target:** < 0.10 (10%)

**Interpretation:**
- <0.05: Highly autonomous, nervous system handling most geometry
- 0.05-0.15: Good balance, UK-0 consulted appropriately
- >0.15: Over-reliance on UK-0, may indicate configuration issues

### Curiosity Events

**What it measures:** Times geometric impossibility detected

**Typical range:** 5-15 per 500 steps

**Interpretation:**
- 0 events: May indicate detector thresholds too strict
- 5-15 events: Normal geometric exploration
- >30 events: Highly dynamic landscape or loose thresholds

### Basin Transitions

**What it measures:** Number of basin changes during navigation

**Typical range:** 8-20 per 500 steps

**Interpretation:**
- <5 transitions: Stuck in attractor, low exploration
- 8-20 transitions: Good landscape traversal
- >30 transitions: Unstable, may indicate weak basins

## Tips for Success

### 1. Start with Mock Mode

Test architecture without API costs:
```bash
# Don't set GROQ_API_KEY
python uii_v10_8.py collect
python uii_v10_8.py navigate
```

Mock adapter generates random but valid responses.

### 2. Monitor Truth Rejections

During collection, watch for:
```
Truth rejection rate > 50%
```

This suggests UK-0 not learning field geometry well.

### 3. Verify Basin Quality

After collection:
```bash
cat basin_map.json | jq '.basins[] | select(.strength == "strong") | .id'
```

Should have at least 3-4 strong basins.

### 4. Check Logs for Patterns

```bash
# Navigation signals
cat navigation_log.json | jq '.steps[] | select(.uk0_called == true) | .signal'

# Basin traversal order
cat navigation_log.json | jq '.steps[] | .basin_id' | uniq
```

### 5. Iterate on Parameters

If navigation seems stuck:
- Lower curiosity thresholds
- Enable humor
- Increase max_steps

If too chaotic:
- Raise curiosity thresholds
- Disable humor temporarily
- Check basin quality

## What Success Looks Like

### Phase 1 Success
```
✓ 10 basins collected in <100 attempts
✓ 7 expansion, 3 preservation (70/30 ratio)
✓ Truth rejection rate <30%
✓ Mix of strong (4), medium (4), weak (2) basins
✓ basin_map.json created
```

### Phase 2 Success
```
✓ Completed 500 steps
✓ Invocation rate <10%
✓ 10-15 basin transitions
✓ 8-12 curiosity events
✓ No collapse or natural collapse after exploration
✓ navigation_log.json shows coherent trajectory
```

## Getting Help

### Debug Mode

Add verbose flags:
```python
collector.collect(verbose=True)
navigator.navigate(max_steps=500, verbose=True)
```

### Check System State

```python
# In navigation loop, add:
print(f"Φ: {phi:.3f}, P: {state.P:.3f}, A: {state.A:.3f}")
print(f"Curiosity: {curiosity_detector.is_triggered()}")
print(f"Humor: {humor_detector.is_triggered()}")
```

### Verify Invariants

```python
# Check CRK constraints
violations = crk_monitor.evaluate(state, trace, perturbation_mag)
if violations:
    print(f"Violations: {violations}")
```

## What's Next?

Once you have successful runs:

1. **Read ARCHITECTURE.md** - Understand module design
2. **Experiment with thresholds** - Tune rigidity detection
3. **Analyze basin patterns** - What geometries emerge?
4. **Test different realities** - Try mock vs. browser
5. **Contribute insights** - Share interesting discoveries

The goal is to demonstrate that **intelligence exists in basin dynamics, not UK-0 control**.

## Quick Reference

```bash
# Full workflow
export GROQ_API_KEY='your_key'        # Optional
python uii_v10_8.py collect           # Phase 1: ~5-10 min
python uii_v10_8.py navigate          # Phase 2: ~1-2 min

# Check outputs
cat basin_map.json | jq .             # Basins
cat navigation_log.json | jq .metrics # Efficiency
cat collapsed_runs.json | jq .        # Collapses (if any)
```

That's it! You're now running substrate-agnostic intelligence as a geometric protocol.