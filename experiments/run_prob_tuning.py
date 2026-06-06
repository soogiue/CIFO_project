import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "core"))

import os
import json
import time
import numpy as np
from datetime import datetime

from triangle_solution import TrianglePaintingSolution
from library.algorithms.geneticalgorithms.ga import genetic_algorithm
from library.algorithms.geneticalgorithms.selection import tournament_selection
from library.algorithms.geneticalgorithms.crossover import (
    uniform_crossover, two_point_crossover, one_point_crossover,
)
from library.algorithms.geneticalgorithms.mutation import gaussian_mutation, random_reset_mutation
from render import save_image


# =============================================================================
#  BEST OPERATOR CONFIG (Section 3 winner, kept fixed for this entire phase)
# =============================================================================

BEST_OPERATOR_CONFIG = {
    "selection_fn": lambda pop, maxi: tournament_selection(pop, maxi, 5),
    "xo_method":    uniform_crossover,
    "mut_method":   gaussian_mutation,
}

PHASE1_WINNER_CONFIG_ID = "tournament_k5__uniform__gaussian"


# =============================================================================
#  PROBABILITY GRID
#  2 crossover probs x 3 mutation probs = 6 configurations.
# =============================================================================

XO_PROBS  = [0.7, 0.9]
MUT_PROBS = [0.01, 0.05, 0.10]


def _make_config_id(xo_prob: float, mut_prob: float) -> str:
    return f"xo{int(round(xo_prob * 10)):02d}_mut{int(round(mut_prob * 100)):03d}"


PROB_GRID = [
    {"xo_prob": xo, "mut_prob": mu, "id": _make_config_id(xo, mu)}
    for xo in XO_PROBS
    for mu in MUT_PROBS
]


# =============================================================================
#  FIXED PARAMETERS  (same pop and gen budget as Sections 2 and 3)
# =============================================================================

N_RUNS        = 10
POP_SIZE      = 100
MAX_GEN       = 800
ELITESIZE     = 2
SNAPSHOT_GENS = [200, 400, 600, 800]

RESULTS_DIR = str(_ROOT / "results" / "prob_tuning")
PHASE1_DIR  = str(_ROOT / "results" / "validation")


# =============================================================================
#  HELPERS
# =============================================================================

def log_path(config_id: str, run_idx: int) -> str:
    return os.path.join(RESULTS_DIR, config_id, "logs", f"run{run_idx + 1:02d}.json")


def is_done(config_id: str, run_idx: int) -> bool:
    return os.path.exists(log_path(config_id, run_idx))


def make_dirs(config_id: str):
    os.makedirs(os.path.join(RESULTS_DIR, config_id, "logs"),   exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, config_id, "images"), exist_ok=True)


def extract_snapshots(fitness_log: list) -> dict:
    last = len(fitness_log) - 1
    return {f"rmse_gen{g}": round(float(fitness_log[min(g - 1, last)]), 4)
            for g in SNAPSHOT_GENS}


def count_done() -> int:
    return sum(
        1 for cfg in PROB_GRID for i in range(N_RUNS) if is_done(cfg["id"], i)
    )


# =============================================================================
#  OPTIONAL: SEED BASELINE FROM PHASE 1
#  Copies the validation logs for (xo=0.7, mut=0.05) so we don't re-run
#  what we already have. Uncomment the call at the bottom to use it.
# =============================================================================

def seed_baseline_from_phase1():
    baseline_id = _make_config_id(0.7, 0.05)
    src_dir = os.path.join(PHASE1_DIR, PHASE1_WINNER_CONFIG_ID, "logs")

    if not os.path.isdir(src_dir):
        print(f"  [seed] Phase 1 logs not found at {src_dir} -- skipping seed.")
        return

    make_dirs(baseline_id)
    copied = 0
    for run_idx in range(N_RUNS):
        dst = log_path(baseline_id, run_idx)
        if os.path.exists(dst):
            continue
        src = os.path.join(src_dir, f"run{run_idx + 1:02d}.json")
        if not os.path.exists(src):
            print(f"  [seed] Phase 1 run{run_idx + 1:02d} not found -- stopping at {copied} copied.")
            break
        with open(src, encoding="utf-8") as f:
            record = json.load(f)
        record["config_id"]   = baseline_id
        record["is_baseline"] = True
        record["seeded_from"] = PHASE1_WINNER_CONFIG_ID
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        copied += 1

    if copied:
        print(f"  [seed] Copied {copied} Phase 1 log(s) -> {baseline_id}")
    else:
        print(f"  [seed] Nothing to copy ({baseline_id} already has {N_RUNS} logs).")


# =============================================================================
#  MAIN LOOP
# =============================================================================

def run_prob_tuning():
    for cfg in PROB_GRID:
        make_dirs(cfg["id"])

    total_runs   = len(PROB_GRID) * N_RUNS
    already_done = count_done()

    print(f"CIFO Phase 2 -- Probability Tuning")
    print(f"  Operators: tournament k=5 | uniform xo | gaussian mut")
    print(f"  Grid: xo_prob={XO_PROBS}  x  mut_prob={MUT_PROBS}")
    print(f"  {len(PROB_GRID)} configs x {N_RUNS} runs = {total_runs} total runs")
    print(f"  pop={POP_SIZE}, gen={MAX_GEN}, elitesize={ELITESIZE}")
    print(f"  Already done: {already_done}/{total_runs}  "
          f"(remaining: {total_runs - already_done})")
    print(f"  Output -> {RESULTS_DIR}/")
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_records = []
    counter     = 0

    for cfg in PROB_GRID:
        config_id = cfg["id"]
        xo_prob   = cfg["xo_prob"]
        mut_prob  = cfg["mut_prob"]
        finals    = []

        print(f"\n-- {config_id}  (xo={xo_prob}, mut={mut_prob})")

        for run_idx in range(N_RUNS):
            counter += 1
            run_num  = run_idx + 1

            if is_done(config_id, run_idx):
                path = log_path(config_id, run_idx)
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                all_records.append(record)
                finals.append(record["rmse_final"])
                print(f"  [SKIP {counter:3d}/{total_runs}]  run {run_num:02d}")
                continue

            print(f"\n  [{counter:3d}/{total_runs}]  run {run_num:02d}/{N_RUNS}")

            population   = [TrianglePaintingSolution() for _ in range(POP_SIZE)]
            rmse_initial = float(population[0].fitness())

            t0 = time.time()

            fitness_log_run   = []
            diversity_log_run = []

            best = _run_ga_with_diversity(
                cfg, population, fitness_log_run, diversity_log_run,
                xo_prob, mut_prob,
            )

            elapsed    = time.time() - t0
            rmse_final = float(best.fitness())
            finals.append(rmse_final)

            snapshots = extract_snapshots(fitness_log_run)

            img_path = os.path.join(
                RESULTS_DIR, config_id, "images", f"run{run_num:02d}.png"
            )
            save_image(best.render(), img_path)

            record = {
                "config_id":       config_id,
                "xo_prob":         xo_prob,
                "mut_prob":        mut_prob,
                "run":             run_num,
                "rmse_initial":    round(rmse_initial, 4),
                "rmse_final":      round(rmse_final,   4),
                "snapshots":       snapshots,
                "elapsed_sec":     round(elapsed, 1),
                "fitness_curve":   [round(v, 4) for v in fitness_log_run],
                # diversity_curve is saved but not used for any plot in this phase
                "diversity_curve": [round(v, 4) for v in diversity_log_run],
                "timestamp":       datetime.now().isoformat(),
                "config": {
                    "pop_size":  POP_SIZE,
                    "max_gen":   MAX_GEN,
                    "xo_prob":   xo_prob,
                    "mut_prob":  mut_prob,
                    "elitesize": ELITESIZE,
                },
            }

            with open(log_path(config_id, run_idx), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            all_records.append(record)
            print(f"    RMSE: {rmse_initial:.2f} -> {rmse_final:.4f}  |  "
                  f"{elapsed / 60:.1f} min")
            print(f"  JSON  -> {log_path(config_id, run_idx)}")
            print(f"  Image -> {img_path}")

        if finals:
            m, s = np.mean(finals), np.std(finals)
            print(f"\n  [OK] {config_id}: "
                  f"mean={m:.4f}  std={s:.4f}  best={min(finals):.4f}  "
                  f"n={len(finals)}/{N_RUNS}")

    print(f"\n  Skipped (already done): {count_done()}/{total_runs}")

    _print_winner(all_records)
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _run_ga_with_diversity(cfg, population, fitness_log_run, diversity_log_run,
                            xo_prob, mut_prob):
    best, fitness_log_ret = genetic_algorithm(
        initial_population  = population,
        max_generations     = MAX_GEN,
        selection_algorithm = BEST_OPERATOR_CONFIG["selection_fn"],
        xo_method           = BEST_OPERATOR_CONFIG["xo_method"],
        mut_method          = BEST_OPERATOR_CONFIG["mut_method"],
        maximization        = False,
        xo_prob             = xo_prob,
        mut_prob            = mut_prob,
        elitesize           = ELITESIZE,
        verbose             = False,
        diversity_log       = diversity_log_run,
    )
    fitness_log_run.extend(fitness_log_ret)
    return best


def _print_winner(records: list):
    if not records:
        return
    by_config = {}
    for r in records:
        by_config.setdefault(r["config_id"], []).append(r["rmse_final"])
    means = {cid: np.mean(vals) for cid, vals in by_config.items()}
    winner_id = min(means, key=means.get)
    print(f"  WINNER: {winner_id}  (mean RMSE = {means[winner_id]:.4f})")
    print(f"  -> Set as FINAL_PROBABILITY_CONFIG for Phase 3+")


# =============================================================================

if __name__ == "__main__":
    # Uncomment to copy Phase 1 baseline logs before starting.
    # seed_baseline_from_phase1()

    run_prob_tuning()
