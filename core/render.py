from PIL import Image, ImageDraw

IMG_WIDTH = 300
IMG_HEIGHT = 400


def render_triangles(triangles: list, background: tuple = (0, 0, 0)) -> Image.Image:
    """Render a list of RGB triangles [x1,y1,x2,y2,x3,y3,R,G,B] onto a black canvas."""
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=background)
    draw = ImageDraw.Draw(img)

    for tri in triangles:
        x1, y1, x2, y2, x3, y3, R, G, B = tri
        polygon = [
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (int(round(x3)), int(round(y3))),
        ]
        color = (
            int(max(0, min(255, round(R)))),
            int(max(0, min(255, round(G)))),
            int(max(0, min(255, round(B)))),
        )
        draw.polygon(polygon, fill=color)

    return img


def save_image(img: Image.Image, path: str):
    img.save(path)
    print(f"Image saved to {path}")
