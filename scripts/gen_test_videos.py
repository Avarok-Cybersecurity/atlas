#!/usr/bin/env python3
"""Generate the committed, copyright-free video fixtures for the video bench.

Four clips of solid colors, one color per second. Solid colors are the
point, not a shortcut: the assertion the whole benchmark rests on is "name the
colors in the order they appear", and a scene with any ambiguity in it makes
that answer a judgement call instead of a fact.

  01_colors_fwd.mp4   red, green, blue, yellow      4 s
  02_colors_rev.mp4   yellow, blue, green, red      4 s   (01 reversed)
  03_colors_fwd.gif   red, green, blue, yellow      4 s   (same as 01)
  04_colors_half.mp4  red, green                    2 s   (half of 01)
  05_colors_unit.mp4  red                           1 s   (quarter of 01)

WHY EACH ONE EXISTS:

* 01 vs 02 — the temporal-order assertion. Same geometry, same prompt, frames
  in the opposite order. If the answer does not reverse with them, the model
  is not reading the sequence; it is pattern-matching the question. This pair
  is what caught the 2026-08-14 splice bug, where video pad tokens received no
  encoder rows at all and the model answered "gray, gray, gray..." while every
  token count looked perfect.
* 01 vs 03 — backend parity. The MP4 goes through ffmpeg, the GIF decodes
  in-process in pure Rust. Identical content must produce identical geometry,
  or the two paths disagree.
* 05, 04, 01 — the geometry assertion, at 1x/2x/4x. THREE durations, not two:
  two measurements cannot test proportionality at all, because the implied
  template overhead absorbs any discrepancy and a line always fits two points.
  With three, `t4 - t2 == 2 * (t2 - t1)` is a claim that can fail, and it is
  independent of both the overhead and the tokens-per-group figure — so it
  holds whatever `--video-fps` the server was started with.

224x224 to match the image ladder: at patch 16 / merge 2 that is a 14x14 patch
grid and 49 merged tokens per temporal group, so the arithmetic stays easy to
check by hand.

Usage:  python3 scripts/gen_test_videos.py     (needs ffmpeg for the MP4s)
"""
import os
import shutil
import subprocess
import sys

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "videos")
SIZE = 224
FPS = 8  # source rate; the server resamples to its own --video-fps

# Chosen to be unmistakable to a VLM and unambiguous in English. No "cyan" or
# "magenta": a model may reasonably call those blue or pink, and the assertion
# would then be about vocabulary rather than about what reached the encoder.
FWD = ["red", "green", "blue", "yellow"]
REV = list(reversed(FWD))
HALF = FWD[:2]
UNIT = FWD[:1]

RGB = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def mp4(path, colors):
    """One second of each color, concatenated, H.264 in MP4."""
    inputs, filt = [], ""
    for i, c in enumerate(colors):
        inputs += ["-f", "lavfi", "-i", f"color=c={c}:s={SIZE}x{SIZE}:d=1:r={FPS}"]
        filt += f"[{i}:v]"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", f"{filt}concat=n={len(colors)}:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True,
    )


def gif(path, colors):
    """Same content as the MP4, written with PIL so it is fully deterministic
    and needs no external tool — this is the fixture that proves the pure-Rust
    decode path, so generating it with ffmpeg would be circular.

    TWO frames per color, because the MP4 yields two frames per color once the
    server samples it at 2 fps, and `temporal_patch_size = 2` then pairs them
    into one group per color. A GIF with one frame per color is a DIFFERENT
    clip: its groups straddle color boundaries (red+green, blue+yellow), and a
    model reading it correctly reports "red, blue". That is what the first
    version of this fixture produced, and it read as an engine bug when it was
    a fixture bug.

    The single tweaked corner pixel is what makes the duplicates survive. PIL
    merges byte-identical consecutive frames — with or without `optimize` — so
    eight frames became four with doubled durations. One flipped low bit in the
    red channel of the bottom-right pixel defeats the merge; it is one pixel in
    50176, invisible to the encoder's 16x16 patching after normalization, and
    it does not change the color any observer would name.
    """
    from PIL import Image

    frames = []
    for c in colors:
        for k in range(2):
            im = Image.new("RGB", (SIZE, SIZE), RGB[c])
            if k == 1:
                r, g, b = RGB[c]
                im.putpixel((SIZE - 1, SIZE - 1), (r ^ 1, g, b))
            frames.append(im)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # ms per frame -> 2 fps, matching the MP4 after sampling
        loop=0,
        optimize=False,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found — the MP4 fixtures cannot be regenerated", file=sys.stderr)
        return 1
    specs = [
        ("01_colors_fwd.mp4", lambda p: mp4(p, FWD)),
        ("02_colors_rev.mp4", lambda p: mp4(p, REV)),
        ("03_colors_fwd.gif", lambda p: gif(p, FWD)),
        ("04_colors_half.mp4", lambda p: mp4(p, HALF)),
        ("05_colors_unit.mp4", lambda p: mp4(p, UNIT)),
    ]
    for name, make in specs:
        path = os.path.join(OUT, name)
        make(path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    print(f"{len(specs)} clips in {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
