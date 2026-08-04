#!/usr/bin/env python3
"""Generate the committed, copyright-free test images for vision soak/bench.

Produces 8 deterministic synthetic PNG images of varied resolution + aspect ratio
under tests/fixtures/images/. Synthetic => no licensing concern; varied sizes
exercise the vision preprocessor (patch counts, aspect-ratio handling, the
max_pixels resize path) under concurrency. Each image has distinct geometry +
a size label so the model has describable content and they are visually
distinct. Re-run to regenerate identically (fixed RNG).

Usage:  python3 scripts/gen_test_images.py
"""
import os
from PIL import Image, ImageDraw

# (w, h, name) — small square -> 720p, plus a portrait. Covers the resize/patch paths.
SPECS = [
    (224, 224, "01_square_224"),
    (336, 336, "02_square_336"),
    (512, 384, "03_landscape_512x384"),
    (640, 360, "04_wide_640x360"),
    (768, 768, "05_square_768"),
    (1024, 576, "06_wide_1024x576"),
    (1280, 720, "07_hd_1280x720"),
    (480, 854, "08_portrait_480x854"),
]
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "images")


def gen(w, h, name, idx):
    im = Image.new("RGB", (w, h))
    px = im.load()
    # deterministic two-axis gradient, hue offset per image -> visually distinct
    r0 = (idx * 37) % 256
    for y in range(h):
        for x in range(w):
            px[x, y] = ((r0 + x * 255 // w) % 256, (y * 255 // h), ((x + y + idx * 29) % 256))
    d = ImageDraw.Draw(im)
    # a few distinct shapes
    d.rectangle([w * 0.1, h * 0.1, w * 0.45, h * 0.45], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.ellipse([w * 0.55, h * 0.55, w * 0.9, h * 0.9], fill=(20, 20, 20), outline=(255, 255, 255), width=3)
    d.line([0, h, w, 0], fill=(255, 255, 0), width=3)
    d.text((8, 8), f"{name}  {w}x{h}", fill=(0, 0, 0))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    for idx, (w, h, name) in enumerate(SPECS):
        path = os.path.join(OUT, f"{name}.png")
        gen(w, h, name, idx).save(path, "PNG")
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    print(f"{len(SPECS)} images in {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
