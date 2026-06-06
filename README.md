# Girl with a Pearl Earring — Genetic Algorithm Image Approximation

**Course:** Computational Intelligence for Optimization (CIFO) — Nova IMS  
**Group:** C07_Finch

---

## Overview

This project evolves an approximation of Vermeer's *Girl with a Pearl Earring* (300×400 px) using a **Genetic Algorithm** that operates on a population of candidate images. Each image is represented as a sequence of 100 semi-transparent triangles drawn on a blank canvas; the algorithm minimises the pixel-level **RMSE** between the rendered canvas and the original painting.

After an extensive hyperparameter search across selection, crossover, and mutation operators, the final configuration runs for **5,000 generations × 10 independent runs**, producing the pre-computed results stored in `results/`.

> ⚠️ **Do not re-run the experiment scripts.** Each full run takes several hours. The notebook is designed to load the pre-computed JSON logs and images without executing any experiment.

---

## Key Concepts Demonstrated

- **Genetic Algorithm design:** population initialisation, generational loop, elitism
- **Representation:** real-valued genome of 900 floats (9 values per triangle: `x1, y1, x2, y2, x3, y3, R, G, B`)
- **Fitness function:** pixel-level RMSE between the GA-rendered image (PIL) and the target painting
- **Selection operators:** tournament selection (k=3, k=5)
- **Crossover operators:** uniform crossover, two-point crossover
- **Mutation operators:** Gaussian perturbation of triangle parameters; per-triangle alpha-channel mutation (separate ablation)
- **Elitism:** top-k individuals carried over unchanged each generation
- **Island model:** parallel sub-populations that periodically migrate individuals — tested as an alternative to the standard single-population GA
- **Hyperparameter search:** grid search over `(xo_prob, mut_prob, selection_k)` combinations; R-squared and convergence curves used to select the final config
- **Diversity tracking:** per-generation fitness standard deviation logged alongside the fitness curve

---

## Repository Structure

```
CIFO_project/
├── ga_image_approximation.ipynb # Main notebook — loads results, visualises, analyses
├── report.pdf                   # Full written report
├── data/
│   └── girl_pearl_earring.png   # Target image (300×400 px)
├── core/                        # Problem-specific classes
│   ├── render.py                    # Renders a list of triangles → PIL image
│   ├── render_alpha.py              # Variant with per-triangle alpha channel
│   ├── triangle_solution.py         # Solution class: genome + RMSE fitness
│   ├── triangle_solution_alpha.py   # Alpha-channel variant
│   ├── triangle_solution_coverage.py # Coverage-aware variant
│   └── mutation_alpha.py            # Mutation operators for alpha-channel genome
├── library/                     # Reusable GA library (operator implementations)
│   ├── algorithms/
│   │   └── geneticalgorithms/
│   │       ├── ga.py            # Core GA loop: get_best(), get_elite(), genetic_algorithm()
│   │       ├── selection.py     # tournament_selection()
│   │       ├── crossover.py     # uniform_crossover(), two_point_crossover()
│   │       └── mutation.py      # gaussian_mutation()
│   └── problems/
│       └── solution.py          # Abstract Solution base class
├── experiments/                 # Standalone scripts that produced the results
│   ├── run_alpha.py             # Ablation: per-triangle alpha-channel mutation
│   ├── run_coverage.py          # Ablation: coverage-aware fitness
│   ├── run_standard_grid.py     # Operator grid search
│   ├── run_prob_tuning.py       # xo_prob × mut_prob grid search
│   ├── run_validation.py        # Validation of top-3 configs (3 × 10 runs)
│   ├── run_island_model.py      # Island-model GA variant
│   └── run_final.py             # Final config: 5 000 gen × 10 runs
└── results/                     # Pre-computed outputs (JSON logs + rendered images)
    ├── alpha/
    ├── coverage/
    ├── standard_grid/
    ├── prob_tuning/
    ├── validation/
    ├── island_model/
    └── final/
```

---

## How to Run the Notebook

The notebook **does not re-run any experiment**. It reads the pre-computed JSON logs from `results/` and the rendered PNG images.

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install numpy pillow matplotlib
   ```
3. Open `ga_image_approximation.ipynb` and run all cells top to bottom.

All results, fitness curves, diversity plots, and rendered approximations are loaded from disk — no compute-intensive code is executed.

---

## Re-running Experiments (optional, not recommended)

If you want to reproduce a specific experiment from scratch, each script in `experiments/` is self-contained. For example:

```bash
# From the project root
python experiments/run_final.py
```

Scripts are **idempotent**: they skip runs whose JSON log already exists, so they can be interrupted and resumed.

> **Expected runtimes on a modern CPU:**
> - `run_final.py` — ~2–4 hours per run × 10 runs = 20–40 hours total
> - `run_standard_grid.py` / `run_prob_tuning.py` — several hours each
> - `run_validation.py` — ~6–12 hours total

---

## Final Configuration

Selected after the full hyperparameter search (validation section of the notebook):

| Parameter | Value |
|-----------|-------|
| Selection | Tournament, k = 5 |
| Crossover | Uniform, p = 0.9 |
| Mutation | Gaussian, p = 0.01 |
| Population size | 100 |
| Generations | 5 000 |
| Elitesize | 2 |
| Independent runs | 10 |

---

## Sample Results

The final GA reduces RMSE from ~60 (random triangles) to below 20 after 5,000 generations. Rendered approximations at generation snapshots (1k, 2k, 3k, 4k, 5k) are saved in `results/final/images/`.
