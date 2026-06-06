import numpy as np
from PIL import Image, ImageDraw


def render_triangles_alpha(repr_):
    """
    Render RGBA triangles using alpha compositing.
    Each triangle is drawn on a transparent layer and blended onto the canvas,
    so overlapping triangles mix correctly. Returns a final RGB image.
    """
    canvas = Image.new("RGBA", (300, 400), (0, 0, 0, 255))
    for tri in repr_:
        x1, y1 = int(round(tri[0])), int(round(tri[1]))
        x2, y2 = int(round(tri[2])), int(round(tri[3]))
        x3, y3 = int(round(tri[4])), int(round(tri[5]))
        R, G, B, A = int(tri[6]), int(tri[7]), int(tri[8]), int(tri[9])
        R = max(0, min(255, R))
        G = max(0, min(255, G))
        B = max(0, min(255, B))
        A = max(0, min(255, A))
        layer = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(R, G, B, A))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas.convert("RGB")
