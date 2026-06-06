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

from triangle_solution_alpha import TrianglePaintingSolutionAlpha
from mutation_alpha import gaussian_mutation_alpha
from library.algorithms.geneticalgorithms.ga import genetic_algorithm
from library.algorithms.geneticalgorithms.selection import tournament_selection
from library.algorithms.geneticalgorithms.crossover import uniform_crossover
from render import save_image


# =============================================================================
#  CONFIG
#  Tests the RGBA representation (Section 4.1): same operator set as the
#  validation winner (tournament k5, uniform, gaussian) but using 10-gene
#  triangles and an alpha-aware mutation function.
#  Probs and pop are kept at the same defaults used throughout Sections 2 and 3.
# =============================================================================

N_RUNS        = 10
POP_SIZE      = 80
MAX_GEN       = 800
XO_PROB       = 0.7
MUT_PROB      = 0.05
ELITESIZE     = 2
SNAPSHOT_GENS = [200, 400, 600, 800]

RESULTS_DIR = str(_ROOT / "results" / "alpha")
IMAGE_DIR   = os.path.join(RESULTS_DIR, "images")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")

assert all(g <= MAX_GEN for g in SNAPSHOT_GENS), (
    f"SNAPSHOT_GENS contains values beyond MAX_GEN={MAX_GEN}"
)

CONFIG_ID = "alpha__tournament_k5__uniform__gaussian_alpha"


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print(f"Experiment -- Alpha Transparency")
    print(f"  {N_RUNS} runs | pop={POP_SIZE} | gen={MAX_GEN}")
    print(f"  Config: tournament_k5 + uniform xo + gaussian_alpha mut")
    print(f"  Output -> {RESULTS_DIR}/")
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finals = []

    for run_idx in range(N_RUNS):
        log_file = os.path.join(LOG_DIR, f"run{run_idx + 1}.json")

        if os.path.exists(log_file):
            print(f"  [SKIP] run {run_idx + 1} already done")
            with open(log_file) as f:
                record = json.load(f)
            finals.append(record["rmse_final"])
            continue

        print(f"\n[{run_idx + 1}/{N_RUNS}]  {CONFIG_ID}")

        population   = [TrianglePaintingSolutionAlpha() for _ in range(POP_SIZE)]
        rmse_initial = float(population[0].fitness())

        t0 = time.time()
        best, fitness_log = genetic_algorithm(
            initial_population  = population,
            max_generations     = MAX_GEN,
            selection_algorithm = lambda pop, maxi: tournament_selection(pop, maxi, 5),
            xo_method           = uniform_crossover,
            mut_method          = gaussian_mutation_alpha,
            maximization        = False,
            xo_prob             = XO_PROB,
            mut_prob            = MUT_PROB,
            elitesize           = ELITESIZE,
            verbose             = False,
        )
        elapsed    = time.time() - t0
        rmse_final = float(best.fitness())
        finals.append(rmse_final)

        snapshots = {f"rmse_gen{g}": round(float(fitness_log[g - 1]), 4)
                     for g in SNAPSHOT_GENS}

        img_path = os.path.join(IMAGE_DIR, f"run{run_idx + 1}.png")
        save_image(best.render(), img_path)

        record = {
            "config_id":     CONFIG_ID,
            "run":           run_idx + 1,
            "rmse_initial":  round(rmse_initial, 4),
            "rmse_final":    round(rmse_final,   4),
            "snapshots":     snapshots,
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
        with open(log_file, "w") as f:
            json.dump(record, f, indent=2)

        print(f"    RMSE: {rmse_initial:.2f} -> {rmse_final:.4f}  |  {elapsed / 60:.1f} min")
        print(f"  JSON  -> {log_file}")
        print(f"  Image -> {img_path}")

    if finals:
        print(f"\n  Results: mean={np.mean(finals):.4f}  "
              f"std={np.std(finals):.4f}  best={min(finals):.4f}")

    print(f"\n  Done! Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
