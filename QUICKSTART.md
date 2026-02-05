# Quick Start Guide

## Installation

### Prerequisites

- Python 3.10 or higher
- Playwright (for browser automation)
- Groq API key (free tier available)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/uii.git
cd uii
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
playwright install chromium
```

3. **Set up API key**:
```bash
export GROQ_API_KEY=your_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

## Basic Usage

### Run v12.5 (Latest)

```bash
python uii_v12_5.py 50 --verbose
```

This will:
- Initialize substrate at neutral state (S=0.5, I=0.5, P=0.5, A=0.7)
- Launch headless Chromium browser
- Start at `about:blank`
- Run for 50 batch cycles (~10 micro-perturbations each)
- Log all metrics to `mentat_triad_v12.5_log.jsonl`

### Command Line Options

```bash
# Run for 100 steps
python uii_v12_5.py 100

# Run with verbose output
python uii_v12_5.py 50 --verbose
python uii_v12_5.py 50 -v

# Combine options
python uii_v12_5.py 100 -v
```

## Understanding the Output

### Verbose Mode

When running with `--verbose`, you'll see:

```
======================================================================
UII v12.5 - CONTINUOUS REALITY PERTURBATION + DEATH CLOCK
Code (CNS) + LLM (Relation) + Browser (Reality)
Running for 50 batch cycles
Death clock budget: 100 steps
======================================================================

======================================================================
STEP 1 [Substrate Stress: 1.0%]
======================================================================
State: S=0.500, I=0.500, P=0.500, A=0.700
Φ=0.405, Rigidity=0.500

[MICRO-PERTURBATIONS] Executed 10 actions
  Actions: {'observe': 10}

[IMPOSSIBILITY DETECTED] bootstrap_migration

[LLM ENUMERATION] Triggered by: bootstrap_migration
  Enumerated 7 trajectories

[AUTONOMOUS LAB] Testing 7 trajectories...
  ✓ [1/7] 1 steps: Φ=0.412
  ✓ [2/7] 1 steps: Φ=0.398
  ...

[COMMITMENT] Best trajectory (Φ=0.412):
  Navigate to Wikipedia random article for maximum entropy
  Re-executing 1 steps on main browser...
  ✓ Trajectory executed successfully

[POST-BATCH STATE]
  S=0.520, I=0.498, P=0.487, A=0.695
  Φ=0.412, Rigidity=0.501
```

### Log File

All metrics are logged to `mentat_triad_v12.5_log.jsonl`:

```json
{
  "type": "step",
  "step": 1,
  "state_before": {"S": 0.5, "I": 0.5, "P": 0.5, "A": 0.7},
  "phi_before": 0.405,
  "micro_perturbations_executed": 10,
  "impossibility_detected": true,
  "impossibility_reason": "bootstrap_migration",
  "llm_invoked": true,
  "trajectories_enumerated": 7,
  "trajectories_tested": 7,
  "trajectories_succeeded": 7,
  "committed_phi": 0.412,
  "state_after": {"S": 0.52, "I": 0.498, "P": 0.487, "A": 0.695},
  "phi_after": 0.412,
  "degradation_progress": 0.01
}
```

## Key Concepts

### Substrate Dimensions

| Symbol | Name | Range | Meaning |
|--------|------|-------|---------|
| **S** | Sensing | [0,1] | Input bandwidth (perturbable surface) |
| **I** | Integration | [0,1] | Compression capacity (structural complexity) |
| **P** | Prediction | [0,1] | Forward modeling (environmental stability) |
| **A** | Attractor | [0,1] | Coherence anchor (optimal ≈ 0.7) |

### Execution Modes

#### 1. Autonomous (80-90% of time)
CNS maintains coherence via micro-perturbations:
- `observe` - Wait for environment to stabilize
- `scroll` - Change viewport (low-cost surface change)
- `read` - Extract content (sensing bandwidth)
- `click` - Trigger response (moderate cost)
- `fill/type` - Mutate state (higher cost)
- `navigate` - Change Reality (highest cost)
- `delay` - Temporal stabilization
- `evaluate` - DOM probe

#### 2. Impossibility (10-20% of time)
When CNS cannot maintain coherence:
1. Detect impossibility (prediction error, rigidity crisis, etc.)
2. Invoke LLM to enumerate 5-10 trajectory options
3. Test ALL trajectories autonomously in Reality
4. Commit trajectory with highest Φ
5. Resume micro-perturbations

### Death Clock (v12.5)

Monotonic substrate degradation creates mortality pressure:

```python
Progress: 0% ──────────────> 100% (cliff)
          ↓                    ↓
     Pristine            Corrupted

Effects:
- Measurements become noisy
- Rigidity increases (crystallization)
- Prediction error floor rises
- Attractor destabilizes
```

System experiences **phenomenological effects only** - never sees raw step count.

## Common Scenarios

### Scenario 1: Clean Run

```
Steps: 50
LLM calls: 8 (16.0%)
Trajectories committed: 8
Final state: S=0.612, I=0.523, P=0.561, A=0.704
Death clock: Survived (50/100 budget used)
Termination: Natural (max steps reached)
```

### Scenario 2: Death Clock Termination

```
Steps: 87
LLM calls: 22 (25.3%)
Trajectories committed: 22
Final state: S=0.423, I=0.389, P=0.278, A=0.541
Death clock: TERMINATED (100/100 budget exhausted)
Termination: Death clock
```

### Scenario 3: Gradient Collapse

```
Steps: 34
LLM calls: 11 (32.4%)
Trajectories committed: 11
Final state: S=0.501, I=0.498, P=0.712, A=0.699
Gradient magnitude: 0.0007
Termination: Gradient collapse
```

## Troubleshooting

### No GROQ_API_KEY

```
FATAL: Cannot form Mentat Triad
Set GROQ_API_KEY environment variable to enable LLM.
```

**Fix**: `export GROQ_API_KEY=your_key_here`

### Playwright Not Installed

```
playwright._impl._errors.Error: Executable doesn't exist
```

**Fix**: `playwright install chromium`

### Rate Limiting

```
groq.RateLimitError: Rate limit exceeded
```

**Fix**: Wait 2 seconds between runs (built-in rate limiter handles this automatically)

### Browser Won't Start

```
ConnectionRefusedError: Reality connection failed
```

**Fix**: Check Playwright installation, ensure headless mode supported on your system

## Analyzing Results

### View Logs

```bash
# Count LLM calls
grep '"llm_invoked": true' mentat_triad_v12.5_log.jsonl | wc -l

# Extract impossibility reasons
grep '"impossibility_reason"' mentat_triad_v12.5_log.jsonl | \
  jq -r '.impossibility_reason' | sort | uniq -c

# Track degradation progress
grep '"degradation_progress"' mentat_triad_v12.5_log.jsonl | \
  jq -r '.degradation_progress'
```

### Python Analysis

```python
import json

# Load logs
logs = []
with open('mentat_triad_v12.5_log.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

# Filter step logs
steps = [log for log in logs if log['type'] == 'step']

# Compute statistics
llm_calls = sum(1 for s in steps if s['llm_invoked'])
avg_phi = sum(s['phi_after'] for s in steps) / len(steps)
degradation_final = steps[-1]['degradation_progress']

print(f"LLM calls: {llm_calls}/{len(steps)} ({llm_calls/len(steps)*100:.1f}%)")
print(f"Average Φ: {avg_phi:.3f}")
print(f"Final degradation: {degradation_final:.1%}")
```

## Next Steps

- **[Framework Overview](docs/FRAMEWORK.md)** - Understand core philosophy
- **[Implementation Guide](docs/IMPLEMENTATION.md)** - Code structure and patterns
- **[Examples](examples/)** - Advanced usage demonstrations
- **[Theory](docs/THEORY.md)** - Mathematical foundations


- **Citation**: https://doi.org/10.5281/zenodo.18017374
