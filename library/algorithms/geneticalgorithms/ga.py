import random
from copy import deepcopy
from typing import Callable


def get_best(population: list, maximization: bool):
    """Return the best individual in the population."""
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]


def get_elite(population: list, maximization: bool, elitesize: int) -> list:
    """Return deep copies of the top `elitesize` individuals, preserving them into the next generation."""
    if elitesize <= 0:
        return []
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(),
                        reverse=maximization)
    return [deepcopy(ind) for ind in sorted_pop[:elitesize]]


def genetic_algorithm(
    initial_population: list,
    max_generations: int,
    selection_algorithm: Callable,
    xo_method: Callable,
    mut_method: Callable,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.1,
    elitism: bool = True,
    elitesize: int = None,
    verbose: bool = True,
    return_population: bool = False,
    diversity_log: list = None,
):
    """
    Standard generational GA loop.

    Each generation:
      1. Extract elite individuals (carried over unchanged).
      2. Fill the rest of the new population via selection → crossover → mutation.
      3. Track the best individual seen across all generations.

    Args:
        diversity_log: if a list is passed in, per-generation fitness std is appended to it.
        return_population: if True, also returns the final population alongside the best and fitness log.
    """
    population = initial_population
    fitness_log = []

    best_overall = get_best(population, maximization)

    for generation in range(1, max_generations + 1):

        if elitesize is None:
            effective_elitesize = 1 if elitism else 0
        else:
            effective_elitesize = elitesize

        elite = get_elite(population, maximization, effective_elitesize)

        # Fill new population up to (pop_size - elitesize) slots
        target_size = len(population) - effective_elitesize
        new_population = []

        while len(new_population) < target_size:
            parent1 = selection_algorithm(population, maximization)
            parent2 = selection_algorithm(population, maximization)

            offspring1, offspring2 = xo_method(parent1, parent2, xo_prob)

            offspring1 = mut_method(offspring1, mut_prob)
            offspring2 = mut_method(offspring2, mut_prob)

            new_population.append(offspring1)
            if len(new_population) < target_size:
                new_population.append(offspring2)

        new_population.extend(elite)
        population = new_population

        # Optional diversity tracking (std of fitness across population)
        if diversity_log is not None:
            fitnesses = [ind.fitness() for ind in population]
            mean_f = sum(fitnesses) / len(fitnesses)
            std_f  = (sum((f - mean_f) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            diversity_log.append(round(std_f, 4))

        gen_best = get_best(population, maximization)
        gen_best_fitness = gen_best.fitness()
        fitness_log.append(gen_best_fitness)

        if maximization and gen_best_fitness > best_overall.fitness():
            best_overall = deepcopy(gen_best)
        elif not maximization and gen_best_fitness < best_overall.fitness():
            best_overall = deepcopy(gen_best)

        if verbose:
            print(f"Generation {generation:4d}/{max_generations} | "
                  f"Best fitness: {gen_best_fitness:.4f} | "
                  f"Overall best: {best_overall.fitness():.4f}")

    if return_population:
        return best_overall, fitness_log, population
    return best_overall, fitness_log
