"""Draws assets/textures/app_icon.png: a square cut in two along the diagonal, orange and blue (the colors the program uses
for the two classes), with a big white magnifying glass over the cut. Few, large shapes, so that it reads at small sizes.
Needs Pillow. Run: python tools/make_icon.py"""
import os
from PIL import Image, ImageDraw

SIZE = 256
SCALE = 4  # drawn four times larger and shrunk, for smooth edges
S = SIZE * SCALE

ORANGE = (250, 150, 30, 255)
BLUE = (50, 120, 245, 255)
WHITE = (255, 255, 255, 255)


def main():
    image = Image.new("RGBA", (S, S), ORANGE)
    draw = ImageDraw.Draw(image)
    draw.polygon([(S, 0), (S, S), (0, S)], fill=BLUE)

    # The magnifying glass: a thick ring and a thick handle, with a soft shadow beneath.
    cx, cy, radius = S * 0.43, S * 0.43, S * 0.24
    ring = int(S * 0.075)
    handle_from = (cx + radius * 0.72, cy + radius * 0.72)
    handle_to = (S * 0.82, S * 0.82)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    off = S * 0.018
    sd.ellipse([cx - radius + off, cy - radius + off, cx + radius + off, cy + radius + off], outline=(0, 0, 0, 70), width=ring)
    sd.line([handle_from[0] + off, handle_from[1] + off, handle_to[0] + off, handle_to[1] + off], fill=(0, 0, 0, 70),
            width=int(S * 0.095))
    image.alpha_composite(shadow)
    draw = ImageDraw.Draw(image)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=WHITE, width=ring)
    draw.line([*handle_from, *handle_to], fill=WHITE, width=int(S * 0.095))
    for x, y in (handle_to,):
        r = S * 0.0475
        draw.ellipse([x - r, y - r, x + r, y + r], fill=WHITE)

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
    image.putalpha(mask)
    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures", "app_icon.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(path)


if __name__ == "__main__":
    main()
