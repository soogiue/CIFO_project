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
from library.algorithms.geneticalgorithms.crossover import uniform_crossover, two_point_crossover
from library.algorithms.geneticalgorithms.mutation import gaussian_mutation
from render import save_image


# =============================================================================
#  CONFIG
#  Same hyperparameters as run_standard_grid, but 30 runs per config instead
#  of 10 to meet the minimum needed for statistical testing (Kruskal-Wallis
#  and Mann-Whitney with Bonferroni correction).
# =============================================================================

N_RUNS    = 30
POP_SIZE  = 100
MAX_GEN   = 800
XO_PROB   = 0.7
MUT_PROB  = 0.05
ELITESIZE = 2
SNAPSHOT_GENS = [200, 400, 600, 800]

RESULTS_DIR = str(_ROOT / "results" / "validation")


# =============================================================================
#  TOP 3 CONFIGS from the grid search (Section 2).
# =============================================================================

TOP_CONFIGS = [
    {
        "id":           "tournament_k5__uniform__gaussian",
        "selection_fn": lambda pop, maxi: tournament_selection(pop, maxi, 5),
        "xo_method":    uniform_crossover,
        "mut_method":   gaussian_mutation,
    },
    {
        "id":           "tournament_k5__two_point__gaussian",
        "selection_fn": lambda pop, maxi: tournament_selection(pop, maxi, 5),
        "xo_method":    two_point_crossover,
        "mut_method":   gaussian_mutation,
    },
    {
        "id":           "tournament_k3__uniform__gaussian",
        "selection_fn": lambda pop, maxi: tournament_selection(pop, maxi, 3),
        "xo_method":    uniform_crossover,
        "mut_method":   gaussian_mutation,
    },
]


# =============================================================================
#  HELPERS  (same pattern as run_standard_grid, per-config folder structure)
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
        1 for cfg in TOP_CONFIGS for i in range(N_RUNS) if is_done(cfg["id"], i)
    )


# =============================================================================
#  MAIN LOOP
# =============================================================================

def run_validation():
    for cfg in TOP_CONFIGS:
        make_dirs(cfg["id"])

    total_runs   = len(TOP_CONFIGS) * N_RUNS
    already_done = count_done()

    print(f"Validation -- {len(TOP_CONFIGS)} configs x {N_RUNS} runs = {total_runs} total")
    print(f"Done: {already_done}/{total_runs}  |  Remaining: {total_runs - already_done}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    counter = 0

    for cfg in TOP_CONFIGS:
        config_id = cfg["id"]
        finals = []

        for run_idx in range(N_RUNS):
            counter  += 1
            run_num   = run_idx + 1

            if is_done(config_id, run_idx):
                with open(log_path(config_id, run_idx), encoding="utf-8") as f:
                    record = json.load(f)
                finals.append(record["rmse_final"])
                print(f"  [SKIP {counter:3d}/{total_runs}]  {config_id}  run {run_num:02d}")
                continue

            print(f"\n[{counter:3d}/{total_runs}]  {config_id}  run {run_num:02d}/{N_RUNS}")

            population   = [TrianglePaintingSolution() for _ in range(POP_SIZE)]
            rmse_initial = float(population[0].fitness())

            t0 = time.time()

            fitness_log_run   = []
            diversity_log_run = []

            best, fitness_log_run, diversity_log_run = _run_ga_with_diversity(
                cfg, population, fitness_log_run, diversity_log_run
            )

            elapsed    = time.time() - t0
            rmse_final = float(best.fitness())
            finals.append(rmse_final)

            snapshots = extract_snapshots(fitness_log_run)

            img_path = os.path.join(RESULTS_DIR, config_id, "images", f"run{run_num:02d}.png")
            save_image(best.render(), img_path)

            record = {
                "config_id":       config_id,
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
                    "xo_prob":   XO_PROB,
                    "mut_prob":  MUT_PROB,
                    "elitesize": ELITESIZE,
                },
            }

            with open(log_path(config_id, run_idx), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            print(f"  RMSE: {rmse_initial:.2f} -> {rmse_final:.4f}  |  {elapsed / 60:.1f} min")
            print(f"  JSON  -> {log_path(config_id, run_idx)}")
            print(f"  Image -> {img_path}")

        if finals:
            m, s = np.mean(finals), np.std(finals)
            print(f"\n  {config_id}: mean={m:.4f}  std={s:.4f}  best={min(finals):.4f}  n={len(finals)}/{N_RUNS}")

    print(f"\nDone. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _run_ga_with_diversity(cfg, population, fitness_log_run, diversity_log_run):
    best, fitness_log_ret = genetic_algorithm(
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
        diversity_log       = diversity_log_run,
    )
    fitness_log_run.extend(fitness_log_ret)
    return best, fitness_log_run, diversity_log_run


if __name__ == "__main__":
    run_validation()
