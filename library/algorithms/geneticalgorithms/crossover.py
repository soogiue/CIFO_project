import random
from copy import deepcopy


def one_point_crossover(parent1, parent2, crossover_prob: float):
    """
    Split both parents at a random point and swap the tails.
    Each parent contributes one contiguous segment to each offspring.
    """
    if random.random() <= crossover_prob:
        n = len(parent1.repr)
        point = random.randint(1, n - 1)

        repr1 = deepcopy(parent1.repr[:point]) + deepcopy(parent2.repr[point:])
        repr2 = deepcopy(parent2.repr[:point]) + deepcopy(parent1.repr[point:])
    else:
        repr1 = deepcopy(parent1.repr)
        repr2 = deepcopy(parent2.repr)

    offspring1 = parent1.__class__(repr=repr1)
    offspring2 = parent1.__class__(repr=repr2)
    return offspring1, offspring2


def uniform_crossover(parent1, parent2, crossover_prob: float):
    """
    For each triangle slot, flip a coin to decide which parent it comes from.
    Produces more gene mixing than point-based methods.
    """
    if random.random() <= crossover_prob:
        repr1 = []
        repr2 = []
        for t1, t2 in zip(parent1.repr, parent2.repr):
            if random.random() < 0.5:
                repr1.append(deepcopy(t1))
                repr2.append(deepcopy(t2))
            else:
                repr1.append(deepcopy(t2))
                repr2.append(deepcopy(t1))
    else:
        repr1 = deepcopy(parent1.repr)
        repr2 = deepcopy(parent2.repr)

    offspring1 = parent1.__class__(repr=repr1)
    offspring2 = parent1.__class__(repr=repr2)
    return offspring1, offspring2


def two_point_crossover(parent1, parent2, crossover_prob: float):
    """
    Pick two cut points; the middle segment is swapped between parents.
    Preserves more structure than uniform while mixing more than one-point.
    """
    if random.random() <= crossover_prob:
        n = len(parent1.repr)
        p1 = random.randint(1, n - 2)
        p2 = random.randint(p1 + 1, n - 1)

        repr1 = (deepcopy(parent1.repr[:p1])
                 + deepcopy(parent2.repr[p1:p2])
                 + deepcopy(parent1.repr[p2:]))
        repr2 = (deepcopy(parent2.repr[:p1])
                 + deepcopy(parent1.repr[p1:p2])
                 + deepcopy(parent2.repr[p2:]))
    else:
        repr1 = deepcopy(parent1.repr)
        repr2 = deepcopy(parent2.repr)

    offspring1 = parent1.__class__(repr=repr1)
    offspring2 = parent1.__class__(repr=repr2)
    return offspring1, offspring2
