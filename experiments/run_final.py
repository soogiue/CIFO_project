import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "core"))

import os
import json
import time
import copy
import numpy as np
from datetime import datetime

from triangle_solution import TrianglePaintingSolution
from library.algorithms.geneticalgorithms.selection import tournament_selection
from library.algorithms.geneticalgorithms.crossover import uniform_crossover
from library.algorithms.geneticalgorithms.mutation import gaussian_mutation
from library.algorithms.geneticalgorithms.ga import get_elite, get_best
from render import save_image


# =============================================================================
#  FINAL CONFIGURATION
#  Fully determined at the end of Section 5: tournament k5, uniform xo,
#  gaussian mut, xo_prob=0.9, mut_prob=0.01.
#  Generation budget is raised to 5000 to allow longer convergence.
# =============================================================================

SELECTION_FN = lambda pop, maxi: tournament_selection(pop, maxi, tournament_size=5)
XO_METHOD    = uniform_crossover
MUT_METHOD   = gaussian_mutation

XO_PROB   = 0.9
MUT_PROB  = 0.01
POP_SIZE  = 100
MAX_GEN   = 5000
ELITESIZE = 2
N_RUNS    = 10

SNAPSHOT_GENS = [1000, 2000, 3000, 4000, 5000]

CONFIG_ID = (
    f"final__standard__pop{POP_SIZE}"
    f"__gen{MAX_GEN}__tourk5__uniform__gaussian"
    f"__xo{int(XO_PROB*10):02d}__mut{int(MUT_PROB*100):03d}"
)

RESULTS_DIR = str(_ROOT / "results" / "final")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")


# =============================================================================
#  HELPERS
# =============================================================================

def log_path(run_idx: int) -> str:
    return os.path.join(LOGS_DIR, f"run{run_idx + 1:02d}.json")


def is_done(run_idx: int) -> bool:
    return os.path.exists(log_path(run_idx))


def make_dirs():
    os.makedirs(LOGS_DIR,   exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def extract_snapshots(fitness_log: list) -> dict:
    last = len(fitness_log) - 1
    return {f"rmse_gen{g}": round(float(fitness_log[min(g - 1, last)]), 4)
            for g in SNAPSHOT_GENS}


# =============================================================================
#  GA CORE
#  Same generational loop as the library ga.py, written inline here so we can
#  collect diversity at every generation without modifying the shared module.
# =============================================================================

def evolve_one_generation(population: list) -> list:
    elite       = get_elite(population, maximization=False, elitesize=ELITESIZE)
    target_size = len(population) - ELITESIZE
    new_pop     = []

    while len(new_pop) < target_size:
        p1 = SELECTION_FN(population, False)
        p2 = SELECTION_FN(population, False)
        o1, o2 = XO_METHOD(p1, p2, XO_PROB)
        o1 = MUT_METHOD(o1, MUT_PROB)
        o2 = MUT_METHOD(o2, MUT_PROB)
        new_pop.append(o1)
        if len(new_pop) < target_size:
            new_pop.append(o2)

    new_pop.extend(elite)
    return new_pop


def standard_ga(verbose: bool = True):
    population    = [TrianglePaintingSolution() for _ in range(POP_SIZE)]
    fitness_log   = []
    diversity_log = []  # fitness std across the population, used for analysis in Section 6
    best_global   = None

    for gen in range(1, MAX_GEN + 1):
        population = evolve_one_generation(population)

        gen_best     = get_best(population, maximization=False)
        gen_best_fit = float(gen_best.fitness())
        fitness_log.append(gen_best_fit)

        if best_global is None or gen_best_fit < best_global.fitness():
            best_global = copy.deepcopy(gen_best)

        all_fitnesses = [ind.fitness() for ind in population]
        mean_f = sum(all_fitnesses) / len(all_fitnesses)
        std_f  = (sum((f - mean_f) ** 2 for f in all_fitnesses) / len(all_fitnesses)) ** 0.5
        diversity_log.append(round(std_f, 4))

        if verbose and gen % 100 == 0:
            print(f"  Gen {gen:5d}/{MAX_GEN} | "
                  f"Best: {gen_best_fit:.4f} | "
                  f"Overall best: {best_global.fitness():.4f} | "
                  f"Diversity: {std_f:.4f}")

    return best_global, fitness_log, diversity_log


# =============================================================================
#  MAIN LOOP
# =============================================================================

def run_final():
    make_dirs()

    already_done = sum(1 for i in range(N_RUNS) if is_done(i))

    print(f"Final Standard GA Run")
    print(f"  Config: {CONFIG_ID}")
    print(f"  Pop: {POP_SIZE}  |  Gen: {MAX_GEN}  |  Elitesize: {ELITESIZE}")
    print(f"  Operators: tournament k=5 | uniform xo | gaussian mut")
    print(f"  xo_prob={XO_PROB}  mut_prob={MUT_PROB}")
    print(f"  n_runs={N_RUNS}  |  Already done: {already_done}/{N_RUNS}")
    print(f"  Output -> {RESULTS_DIR}/")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for run_idx in range(N_RUNS):
        run_num = run_idx + 1

        if is_done(run_idx):
            print(f"  [SKIP]  run {run_num:02d}/{N_RUNS}")
            continue

        print(f"\n  -- run {run_num:02d}/{N_RUNS} --")

        init_ind     = TrianglePaintingSolution()
        rmse_initial = float(init_ind.fitness())

        t0 = time.time()
        best, fitness_log, diversity_log = standard_ga(verbose=True)
        elapsed    = time.time() - t0
        rmse_final = float(best.fitness())

        snapshots = extract_snapshots(fitness_log)

        img_path = os.path.join(IMAGES_DIR, f"run{run_num:02d}.png")
        save_image(best.render(), img_path)

        record = {
            "config_id":       CONFIG_ID,
            "run":             run_num,
            "rmse_initial":    round(rmse_initial, 4),
            "rmse_final":      round(rmse_final,   4),
            "snapshots":       snapshots,
            "fitness_curve":   [round(v, 4) for v in fitness_log],
            "diversity_curve": diversity_log,
            "elapsed_sec":     round(elapsed, 1),
            "timestamp":       datetime.now().isoformat(),
            "config": {
                "xo_prob":   XO_PROB,
                "mut_prob":  MUT_PROB,
                "pop_size":  POP_SIZE,
                "max_gen":   MAX_GEN,
                "elitesize": ELITESIZE,
            },
        }

        with open(log_path(run_idx), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print(f"  RMSE: {rmse_initial:.2f} -> {rmse_final:.4f}  |  {elapsed / 60:.1f} min")
        print(f"  JSON  -> {log_path(run_idx)}")
        print(f"  Image -> {img_path}")

    _print_summary()
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _print_summary():
    finals = []
    for i in range(N_RUNS):
        if is_done(i):
            with open(log_path(i), encoding="utf-8") as f:
                d = json.load(f)
            finals.append(d["rmse_final"])

    if not finals:
        print("  No completed runs to summarise.")
        return

    arr = np.array(finals)
    print(f"  Final Standard GA Results ({len(finals)} runs completed)")
    print(f"  Mean RMSE:   {arr.mean():.4f}")
    print(f"  Std  RMSE:   {arr.std():.4f}")
    print(f"  Best RMSE:   {arr.min():.4f}")
    print(f"  Worst RMSE:  {arr.max():.4f}")


if __name__ == "__main__":
    run_final()
