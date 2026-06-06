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
from library.algorithms.geneticalgorithms.selection import (
    fitness_proportionate_selection,
    rank_selection,
    tournament_selection,
)
from library.algorithms.geneticalgorithms.crossover import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)
from library.algorithms.geneticalgorithms.mutation import (
    gaussian_mutation,
    random_reset_mutation,
)
from render import save_image


# =============================================================================
#  CONFIG
#  Fixed hyperparameters shared across all 30 operator combinations.
#  xo_prob and mut_prob are intentionally kept at neutral defaults here;
#  they are tuned separately in run_prob_tuning.py (Section 5).
# =============================================================================

N_RUNS        = 10
POP_SIZE      = 80
MAX_GEN       = 800
XO_PROB       = 0.7
MUT_PROB      = 0.05
ELITESIZE     = 2
SNAPSHOT_GENS = [200, 400, 600, 800]

RESULTS_DIR = str(_ROOT / "results" / "standard_grid")
IMAGE_DIR   = os.path.join(RESULTS_DIR, "images")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")


# =============================================================================
#  GRID DEFINITION
#  All combinations of 5 selection methods, 3 crossovers, 2 mutations = 30 configs.
# =============================================================================

def build_grid() -> list:
    selections = [
        ("roulette",      lambda pop, maxi: fitness_proportionate_selection(pop, maxi)),
        ("rank",          lambda pop, maxi: rank_selection(pop, maxi)),
        ("tournament_k2", lambda pop, maxi: tournament_selection(pop, maxi, 2)),
        ("tournament_k3", lambda pop, maxi: tournament_selection(pop, maxi, 3)),
        ("tournament_k5", lambda pop, maxi: tournament_selection(pop, maxi, 5)),
    ]
    crossovers = [
        ("one_point",  one_point_crossover),
        ("two_point",  two_point_crossover),
        ("uniform",    uniform_crossover),
    ]
    mutations = [
        ("gaussian",     gaussian_mutation),
        ("random_reset", random_reset_mutation),
    ]

    grid = []
    for sel_name, sel_fn in selections:
        for xo_name, xo_fn in crossovers:
            for mut_name, mut_fn in mutations:
                config_id = f"sel_{sel_name}__xo_{xo_name}__mut_{mut_name}"
                grid.append({
                    "id":           config_id,
                    "selection_fn": sel_fn,
                    "xo_method":    xo_fn,
                    "mut_method":   mut_fn,
                })
    return grid


# =============================================================================
#  HELPERS
# =============================================================================

def make_dirs():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)


def log_path(config_id: str, run_idx: int) -> str:
    return os.path.join(LOG_DIR, f"{config_id}_run{run_idx + 1}.json")


def is_done(config_id: str, run_idx: int) -> bool:
    return os.path.exists(log_path(config_id, run_idx))


def extract_snapshots(fitness_log: list) -> dict:
    """Pull RMSE at fixed generation checkpoints for later convergence plots."""
    last = len(fitness_log) - 1
    return {f"rmse_gen{g}": float(fitness_log[min(g - 1, last)])
            for g in SNAPSHOT_GENS}


# =============================================================================
#  MAIN LOOP
# =============================================================================

def run_grid():
    make_dirs()
    grid        = build_grid()
    total_runs  = len(grid) * N_RUNS
    counter     = 0
    skipped     = 0

    print(f"CIFO -- GA Grid Search")
    print(f"  {len(grid)} configs x {N_RUNS} runs = {total_runs} total runs")
    print(f"  pop={POP_SIZE}, gen={MAX_GEN}, xo_prob={XO_PROB}, "
          f"mut_prob={MUT_PROB}, elitesize={ELITESIZE}")
    print(f"  Output -> {RESULTS_DIR}/")
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for cfg in grid:
        config_id = cfg["id"]
        finals    = []

        for run_idx in range(N_RUNS):
            counter += 1

            if is_done(config_id, run_idx):
                skipped += 1
                path = log_path(config_id, run_idx)
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                finals.append(record["rmse_final"])
                print(f"  [SKIP {counter}/{total_runs}]  {config_id}  run {run_idx + 1}")
                continue

            print(f"\n[{counter}/{total_runs}]  {config_id}  --  run {run_idx + 1}/{N_RUNS}")

            population   = [TrianglePaintingSolution() for _ in range(POP_SIZE)]
            rmse_initial = float(population[0].fitness())

            t0 = time.time()

            best, fitness_log = genetic_algorithm(
                initial_population  = population,
                max_generations     = MAX_GEN,
                selection_algorithm = cfg["selection_fn"],
                xo_method           = cfg["xo_method"],
                mut_method          = cfg["mut_method"],
                maximization        = False,
                xo_prob             = XO_PROB,
                mut_prob            = MUT_PROB,
                elitesize           = ELITESIZE,
                verbose             = False,
            )

            elapsed    = time.time() - t0
            rmse_final = float(best.fitness())
            finals.append(rmse_final)

            snapshots = extract_snapshots(fitness_log)

            img_path = os.path.join(IMAGE_DIR, f"{config_id}_run{run_idx + 1}.png")
            save_image(best.render(), img_path)

            record = {
                "config_id":     config_id,
                "run":           run_idx + 1,
                "rmse_initial":  round(rmse_initial, 4),
                "rmse_final":    round(rmse_final,   4),
                "snapshots":     {k: round(v, 4) for k, v in snapshots.items()},
                "elapsed_sec":   round(elapsed, 1),
                "fitness_curve": [round(v, 4) for v in fitness_log],
                "timestamp":     datetime.now().isoformat(),
                "config": {
                    "pop_size":  POP_SIZE,
                    "max_gen":   MAX_GEN,
                    "xo_prob":   XO_PROB,
                    "mut_prob":  MUT_PROB,
                    "elitesize": ELITESIZE,
                },
            }
            with open(log_path(config_id, run_idx), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            print(f"    RMSE: {rmse_initial:.2f} -> {rmse_final:.4f}  |  {elapsed / 60:.1f} min")
            print(f"  JSON  -> {log_path(config_id, run_idx)}")
            print(f"  Image -> {img_path}")

        if finals:
            m, s = np.mean(finals), np.std(finals)
            print(f"\n  [OK]  {config_id}: "
                  f"mean={m:.4f}  std={s:.4f}  best={min(finals):.4f}")

    print(f"\n  Skipped (already done): {skipped}/{total_runs}")
    print(f"  Grid search complete!")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_grid()
