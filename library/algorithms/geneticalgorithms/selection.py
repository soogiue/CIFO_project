import random
from copy import deepcopy
from library.problems.solution import Solution


def fitness_proportionate_selection(population: list, maximization: bool):
    """
    Roulette-wheel selection: each individual's chance of being picked is
    proportional to its fitness (or 1/fitness for minimization).
    """
    if maximization:
        fitness_values = [ind.fitness() for ind in population]
    else:
        # Invert so lower fitness = higher selection probability
        fitness_values = [1.0 / (ind.fitness() + 1e-9) for ind in population]

    total_fitness = sum(fitness_values)
    random_nr = random.uniform(0, total_fitness)

    sliding_value = 0
    for idx, ind in enumerate(population):
        sliding_value += fitness_values[idx]
        if random_nr <= sliding_value:
            return deepcopy(ind)

    return deepcopy(population[-1])  # fallback for floating-point edge cases


def tournament_selection(population: list, maximization: bool, tournament_size: int = 3):
    """
    Pick `tournament_size` individuals at random and return the best one.
    Higher tournament_size = more selection pressure.
    """
    k = min(tournament_size, len(population))
    competitors = random.sample(population, k)

    fitness_values = [ind.fitness() for ind in competitors]

    if maximization:
        winner = competitors[fitness_values.index(max(fitness_values))]
    else:
        winner = competitors[fitness_values.index(min(fitness_values))]

    return deepcopy(winner)


def rank_selection(population: list, maximization: bool):
    """
    Assign selection weights based on rank rather than raw fitness.
    Reduces the dominance of outliers compared to fitness-proportionate selection.
    """
    n = len(population)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=maximization)

    # Best individual gets weight n, worst gets weight 1
    weights = list(range(n, 0, -1))
    total = sum(weights)

    r = random.uniform(0, total)
    cumulative = 0.0
    for ind, w in zip(sorted_pop, weights):
        cumulative += w
        if r <= cumulative:
            return deepcopy(ind)

    return deepcopy(sorted_pop[-1])  # fallback
