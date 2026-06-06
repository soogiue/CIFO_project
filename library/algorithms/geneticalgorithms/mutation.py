import random
from copy import deepcopy

IMG_WIDTH = 300
IMG_HEIGHT = 400


def gaussian_mutation(individual, mut_prob: float, sigma: float = 15.0):
    """
    Perturb each gene of each triangle with Gaussian noise (std=sigma).
    Coordinates are clamped to image bounds; RGB channels to [0, 255].
    """
    new_repr = deepcopy(individual.repr)

    for triangle in new_repr:
        for i in range(9):  # [x1,y1,x2,y2,x3,y3,R,G,B]
            if random.random() < mut_prob:
                noise = random.gauss(0, sigma)
                triangle[i] += noise

                if i in (0, 2, 4):      # x coords
                    triangle[i] = max(0, min(IMG_WIDTH, triangle[i]))
                elif i in (1, 3, 5):    # y coords
                    triangle[i] = max(0, min(IMG_HEIGHT, triangle[i]))
                else:                   # R, G, B
                    triangle[i] = max(0, min(255, triangle[i]))

    return individual.__class__(repr=new_repr)


def random_reset_mutation(individual, mut_prob: float):
    """
    With probability mut_prob, replace an entire triangle with a brand-new random one.
    Useful for escaping local optima by introducing large jumps in the search space.
    """
    new_repr = deepcopy(individual.repr)

    for i in range(len(new_repr)):
        if random.random() < mut_prob:
            new_repr[i] = [
                random.uniform(0, IMG_WIDTH),
                random.uniform(0, IMG_HEIGHT),
                random.uniform(0, IMG_WIDTH),
                random.uniform(0, IMG_HEIGHT),
                random.uniform(0, IMG_WIDTH),
                random.uniform(0, IMG_HEIGHT),
                random.uniform(0, 255),
                random.uniform(0, 255),
                random.uniform(0, 255),
            ]

    return individual.__class__(repr=new_repr)
