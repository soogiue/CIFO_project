import random
import numpy as np
from PIL import Image
from pathlib import Path

from triangle_solution import TrianglePaintingSolution
from render_alpha import render_triangles_alpha

IMG_WIDTH     = 300
IMG_HEIGHT    = 400
NUM_TRIANGLES = 100

_TARGET_PATH = Path(__file__).parent.parent / "data" / "girl_pearl_earring.png"
_TARGET_ARR  = np.array(Image.open(_TARGET_PATH).convert("RGB"), dtype=np.float64)


class TrianglePaintingSolutionAlpha(TrianglePaintingSolution):
    """
    Extension of TrianglePaintingSolution with an extra alpha channel.
    Each triangle is encoded as [x1,y1,x2,y2,x3,y3,R,G,B,A].
    Alpha is initialized in [50, 220] to avoid fully opaque/transparent triangles.
    """

    def random_initial_representation(self):
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
                random.uniform(50, 220),  # alpha: avoid extremes
            ]
            triangles.append(triangle)
        return triangles

    def fitness(self):
        """RMSE vs target, computed on the alpha-composited render.
F       itness caching is omitted: render_alpha composites a separate RGBA layer per triangle
        (100 Image.alpha_composite calls per render), making each evaluation more expensive than
        the standard RGB renderer. Running this variant for 5 000 generations or 30 independent
        runs would be prohibitively costly; it was therefore kept as an exploratory variant only.
        """
        rendered = render_triangles_alpha(self.repr)
        arr = np.array(rendered, dtype=np.float64)
        return float(np.sqrt(np.mean((arr - _TARGET_ARR) ** 2)))

    def render(self):
        return render_triangles_alpha(self.repr)
