import random
import numpy as np
from PIL import Image
from pathlib import Path

from library.problems.solution import Solution
from render import render_triangles

IMG_WIDTH     = 300
IMG_HEIGHT    = 400
NUM_TRIANGLES = 100
TARGET_PATH   = Path(__file__).parent.parent / "data" / "girl_pearl_earring.png"

# Load target once at module level to avoid re-reading on every fitness call
_target_img   = Image.open(TARGET_PATH).convert("RGB")
_target_array = np.array(_target_img, dtype=np.float64)


class TrianglePaintingSolution(Solution):
    """
    GA solution that approximates a target image using 100 RGB triangles.
    Each triangle is encoded as [x1,y1,x2,y2,x3,y3,R,G,B].
    Fitness is the RMSE between the rendered image and the target (lower = better).
    """

    def random_initial_representation(self) -> list:
        """Generate 100 triangles with random positions and colors."""
        triangles = []
        for _ in range(NUM_TRIANGLES):
            triangle = [
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
            triangles.append(triangle)
        return triangles

    def fitness(self) -> float:
        """RMSE vs target image. Result is cached to avoid redundant renders."""
        if hasattr(self, '_fitness_cache'):
            return self._fitness_cache

        rendered_img   = render_triangles(self.repr)
        rendered_array = np.array(rendered_img, dtype=np.float64)

        mse  = np.mean((rendered_array - _target_array) ** 2)
        rmse = np.sqrt(mse)
        self._fitness_cache = float(rmse)
        return self._fitness_cache

    def render(self) -> Image.Image:
        return render_triangles(self.repr)

    def __repr__(self):
        return f"TrianglePaintingSolution(fitness={self.fitness():.4f})"
