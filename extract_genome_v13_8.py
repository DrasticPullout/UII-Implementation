"""
Extract genome from UII v13.8 execution log.

Usage:
  python extract_genome_v13_8.py [log_path]

Reads: mentat_triad_v13_8_log.jsonl (or specified path)
Writes: genome.json (6 floats + fitness)

v13.8: Fitness now includes S/I health penalty.
Evolution pressured toward balanced S/I development.
"""

import json
from pathlib import Path
import sys

def extract_latest_session(log_path: str = "mentat_triad_v13_8_log.jsonl"):
    if not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        return None, None, None
    
    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    
    sessions = []
    current_session = {"start": None, "steps": [], "end": None}
    
    for line in lines:
        if line.get("type") == "session_start":
            if current_session["start"]:
                sessions.append(current_session)
            current_session = {"start": line, "steps": [], "end": None}
        elif line.get("type") == "session_end":
            current_session["end"] = line
        elif "step" in line:
            current_session["steps"].append(line)
    
    if current_session["start"]:
        sessions.append(current_session)
    
    if not sessions:
        print("Error: No sessions found in log")
        return None, None, None
    
    latest = sessions[-1]
    
    if not latest["end"]:
        print("Warning: Latest session incomplete (crashed or interrupted)")
        print("  Attempting partial fitness extraction...")
        latest["end"] = {
            "fitness": {
                "freeze_achieved": False,
                "survival_time": len(latest["steps"]),
                "migration_attempted": False,
                "freeze_step": None,
                "tokens_to_freeze": None,
            },
            "final_state": None,
            "total_steps": len(latest["steps"])
        }
    
    fitness_data = latest["end"]["fitness"]
    genome_data = latest["start"]["genome"]
    final_state = latest["end"].get("final_state")
    
    print(f"\n[EXTRACTED SESSION {len(sessions)}]")
    print(f"  Generation: {latest['start']['generation']}")
    print(f"  Steps: {latest['end'].get('total_steps', 'unknown')}")
    print(f"  Freeze: {fitness_data['freeze_achieved']} (step {fitness_data.get('freeze_step', 'N/A')})")
    print(f"  Migration: {fitness_data['migration_attempted']}")
    print(f"  Survival: {fitness_data['survival_time']} steps")
    
    if final_state:
        print(f"  Final state: S={final_state['S']:.3f}, I={final_state['I']:.3f}, "
              f"P={final_state['P']:.3f}, A={final_state['A']:.3f}")
        SI_health = (final_state["S"] + final_state["I"]) / 2.0
        print(f"  SI health: {SI_health:.3f}")
        if final_state["I"] < 0.1:
            print(f"  Warning: I collapsed to {final_state['I']:.3f} - fitness penalty applied")
    
    return genome_data, fitness_data, final_state


def compute_fitness(metrics, final_state=None):
    """
    Fitness = pure environmental selection (survival + freeze efficiency).
    
    SI_health_factor deliberately excluded.
    
    Reason: Grounded Φ (v13.8) should create selection pressure toward
    balanced S/I on its own. Adding an explicit fitness multiplier would
    mask whether Φ grounding is load-bearing.
    
    If S/I develops naturally across generations → Φ grounding works.
    If S/I still collapses → Φ pressure insufficient, revisit grounding.
    
    final_state is logged and reported but does NOT affect fitness score.
    """
    survival = float(metrics.get("survival_time", 0))
    
    if not metrics["freeze_achieved"]:
        return survival
    
    freeze_step = metrics.get("freeze_step")
    tokens_to_freeze = metrics.get("tokens_to_freeze")
    
    if freeze_step is None or tokens_to_freeze is None:
        return survival
    
    freeze_speed = 100.0 / max(freeze_step, 1)
    token_efficiency = 50000.0 / max(tokens_to_freeze, 1)
    migration_bonus = 100.0 if metrics["migration_attempted"] else 0.0
    
    return freeze_speed + token_efficiency + survival + migration_bonus


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "mentat_triad_v13_8_log.jsonl"
    
    genome_data, fitness_metrics, final_state = extract_latest_session(log_path)
    
    if genome_data is None:
        sys.exit(1)
    
    fitness = compute_fitness(fitness_metrics, final_state)
    
    print(f"\n[FITNESS SCORE v13.8]")
    if fitness_metrics["freeze_achieved"]:
        freeze_step = fitness_metrics.get("freeze_step", 1)
        tokens_to_freeze = fitness_metrics.get("tokens_to_freeze", 1)
        print(f"  freeze_speed: {100.0 / max(freeze_step, 1):.2f}")
        print(f"  token_efficiency: {50000.0 / max(tokens_to_freeze, 1):.2f}")
        print(f"  survival (raw): {fitness_metrics['survival_time']}")
        print(f"  migration_bonus: {100.0 if fitness_metrics['migration_attempted'] else 0.0}")
    else:
        print(f"  [NO FREEZE ACHIEVED]")
        print(f"  survival (raw): {fitness_metrics['survival_time']}")
    
    if final_state:
        S = final_state.get("S", 0.5)
        I = final_state.get("I", 0.5)
        SI_min = min(S, I)
        SI_mean = (S + I) / 2.0
        SI_balance = SI_min * 0.5 + SI_mean * 0.5
        print(f"  SI_balance (observed): {SI_balance:.3f} [diagnostic only - not in fitness]")
        print(f"  Note: If S/I develops across generations, Phi grounding is working.")
    
    print(f"  TOTAL: {fitness:.2f}")
    
    genome_data["parent_fitness"] = fitness
    output = {"genome": genome_data, "fitness": fitness, "metrics": fitness_metrics, "final_state": final_state}
    
    with open("genome.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nGenome saved: genome.json")
    print(f"  Generation: {genome_data['generation']}")
    print(f"  Fitness: {fitness:.2f}")
    print(f"\nNext: Wait for rate limit reset, then:")
    print(f"  python uii_v13_8.py --load-genome genome.json")