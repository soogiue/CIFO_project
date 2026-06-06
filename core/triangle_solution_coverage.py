import random
from triangle_solution import TrianglePaintingSolution

IMG_WIDTH     = 300
IMG_HEIGHT    = 400
NUM_TRIANGLES = 100
GRID_COLS     = 10
GRID_ROWS     = 10
CELL_W        = IMG_WIDTH  / GRID_COLS
CELL_H        = IMG_HEIGHT / GRID_ROWS


class TrianglePaintingSolutionCoverage(TrianglePaintingSolution):
    """
    Variant of TrianglePaintingSolution with a smarter initialization.
    The canvas is divided into a 10x10 grid and one triangle is placed per cell,
    ensuring full spatial coverage from generation 0 instead of random clustering.
    Everything else (fitness, mutation, crossover) is inherited unchanged.
    """

    def random_initial_representation(self) -> list:
        triangles = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Constrain all 3 vertices within the current cell
                x_min = col * CELL_W
                x_max = x_min + CELL_W
                y_min = row * CELL_H
                y_max = y_min + CELL_H
                triangle = [
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                    random.uniform(0, 255),
                    random.uniform(0, 255),
                    random.uniform(0, 255),
                ]
                triangles.append(triangle)
        return triangles
