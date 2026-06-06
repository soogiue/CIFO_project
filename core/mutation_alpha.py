import random
import copy


def gaussian_mutation_alpha(individual, mut_prob):
    """
    Gaussian mutation for RGBA triangles.
    Each gene is perturbed with prob=mut_prob by a Gaussian noise (std=30).
    Coordinates are clamped to image bounds (300x400), RGBA to [0, 255].
    """
    new_repr = copy.deepcopy(individual.repr)
    for tri in new_repr:
        for gene_idx in range(10):  # [x1,y1,x2,y2,x3,y3,R,G,B,A]
            if random.random() < mut_prob:
                tri[gene_idx] += random.gauss(0, 30)
                if gene_idx in (0, 2, 4):       # x coords
                    tri[gene_idx] = max(0.0, min(300.0, tri[gene_idx]))
                elif gene_idx in (1, 3, 5):     # y coords
                    tri[gene_idx] = max(0.0, min(400.0, tri[gene_idx]))
                else:                           # R, G, B, A
                    tri[gene_idx] = max(0.0, min(255.0, tri[gene_idx]))
    return individual.__class__(repr=new_repr)
