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
#  FINAL CONFIGURATION  (identical to run_final)
#  The only differences vs run_final are: population is split into N_ISLANDS
#  sub-populations that evolve independently and exchange individuals every
#  MIGRATION_INTERVAL generations in a ring topology.
#  Total population (4 x 25 = 100) and generation budget are kept the same
#  as run_final to make the comparison fair.
# =============================================================================

SELECTION_FN = lambda pop, maxi: tournament_selection(pop, maxi, tournament_size=5)
XO_METHOD    = uniform_crossover
MUT_METHOD   = gaussian_mutation

XO_PROB  = 0.9
MUT_PROB = 0.01

N_ISLANDS          = 4
ISLAND_SIZE        = 25
MAX_GEN            = 5000
MIGRATION_INTERVAL = 20   # migrate every 20 generations
MIGRATION_SIZE     = 1    # send the best individual from each island
ELITESIZE          = 2
N_RUNS             = 10

SNAPSHOT_GENS = [1000, 2000, 3000, 4000, 5000]

CONFIG_ID = (
    f"island2__n{N_ISLANDS}__size{ISLAND_SIZE}"
    f"__gen{MAX_GEN}__migr{MIGRATION_INTERVAL}__migr{MIGRATION_SIZE}__ring"
    f"__xo{int(XO_PROB*10):02d}__mut{int(MUT_PROB*100):03d}"
)

RESULTS_DIR = str(_ROOT / "results" / "island_model")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")


# =============================================================================
#  HELPERS  (same as run_final)
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
#  ISLAND MODEL CORE
# =============================================================================

def evolve_one_generation(island: list) -> list:
    """Standard GA step applied to a single island."""
    elite       = get_elite(island, maximization=False, elitesize=ELITESIZE)
    target_size = len(island) - ELITESIZE
    new_pop     = []

    while len(new_pop) < target_size:
        p1 = SELECTION_FN(island, False)
        p2 = SELECTION_FN(island, False)
        o1, o2 = XO_METHOD(p1, p2, XO_PROB)
        o1 = MUT_METHOD(o1, MUT_PROB)
        o2 = MUT_METHOD(o2, MUT_PROB)
        new_pop.append(o1)
        if len(new_pop) < target_size:
            new_pop.append(o2)

    new_pop.extend(elite)
    return new_pop


def migrate_ring(islands: list) -> list:
    """
    Ring migration: each island sends its best individual to the next island
    (clockwise), replacing the worst individual there.
    """
    n           = len(islands)
    new_islands = [list(isl) for isl in islands]

    for i in range(n):
        dest      = (i + 1) % n
        emigrants = sorted(islands[i], key=lambda x: x.fitness())[:MIGRATION_SIZE]
        new_islands[dest].sort(key=lambda x: x.fitness(), reverse=True)
        for j, em in enumerate(emigrants):
            new_islands[dest][j] = copy.deepcopy(em)

    return new_islands


def island_model_ga(verbose: bool = True):
    islands = [
        [TrianglePaintingSolution() for _ in range(ISLAND_SIZE)]
        for _ in range(N_ISLANDS)
    ]

    fitness_log   = []
    diversity_log = []  # fitness std across all islands combined, used for analysis in Section 6
    best_global   = None

    for gen in range(1, MAX_GEN + 1):
        for i in range(N_ISLANDS):
            islands[i] = evolve_one_generation(islands[i])

        if gen % MIGRATION_INTERVAL == 0:
            islands = migrate_ring(islands)

        all_individuals = [ind for isl in islands for ind in isl]
        gen_best        = get_best(all_individuals, maximization=False)
        gen_best_fit    = float(gen_best.fitness())
        fitness_log.append(gen_best_fit)

        if best_global is None or gen_best_fit < best_global.fitness():
            best_global = copy.deepcopy(gen_best)

        all_fitnesses = [ind.fitness() for ind in all_individuals]
        mean_f = sum(all_fitnesses) / len(all_fitnesses)
        std_f  = (sum((f - mean_f) ** 2 for f in all_fitnesses) / len(all_fitnesses)) ** 0.5
        diversity_log.append(round(std_f, 4))

        if verbose and gen % 100 == 0:
            print(f"  Gen {gen:5d}/{MAX_GEN} | "
                  f"Global best: {gen_best_fit:.4f} | "
                  f"Overall best: {best_global.fitness():.4f} | "
                  f"Diversity: {std_f:.4f}")

    return best_global, fitness_log, diversity_log


# =============================================================================
#  MAIN LOOP
# =============================================================================

def run_island_model():
    make_dirs()

    already_done = sum(1 for i in range(N_RUNS) if is_done(i))

    print(f"Island Model GA (5000 gen extended run)")
    print(f"  Config: {CONFIG_ID}")
    print(f"  Islands: {N_ISLANDS} x {ISLAND_SIZE} = {N_ISLANDS * ISLAND_SIZE} total pop")
    print(f"  Migration: every {MIGRATION_INTERVAL} gen, size={MIGRATION_SIZE}, ring topology")
    print(f"  xo_prob={XO_PROB}  mut_prob={MUT_PROB}  elitesize={ELITESIZE}/island")
    print(f"  max_gen={MAX_GEN}  n_runs={N_RUNS}")
    print(f"  Already done: {already_done}/{N_RUNS}")
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
        best, fitness_log, diversity_log = island_model_ga(verbose=True)
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
            "island_config": {
                "n_islands":            N_ISLANDS,
                "island_size":          ISLAND_SIZE,
                "migration_interval":   MIGRATION_INTERVAL,
                "migration_size":       MIGRATION_SIZE,
                "topology":             "ring",
                "elitesize_per_island": ELITESIZE,
            },
            "config": {
                "xo_prob":   XO_PROB,
                "mut_prob":  MUT_PROB,
                "total_pop": N_ISLANDS * ISLAND_SIZE,
                "max_gen":   MAX_GEN,
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
    print(f"  Island Model 5000-gen Results ({len(finals)} runs completed)")
    print(f"  Mean RMSE:   {arr.mean():.4f}")
    print(f"  Std  RMSE:   {arr.std():.4f}")
    print(f"  Best RMSE:   {arr.min():.4f}")
    print(f"  Worst RMSE:  {arr.max():.4f}")


if __name__ == "__main__":
    run_island_model()
