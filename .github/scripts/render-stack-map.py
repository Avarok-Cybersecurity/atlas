#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render the stack map for one pull request in a stack.

Sibling of `render-certificate.py` and deliberately identical in shape: the
template ships with stable `field-*` ids and per-layer groups, and this script
substitutes over them rather than generating SVG. Anything the template can be
made to look like, a designer can change without touching Python.

The template holds six layer slots. Unused ones are hidden, and exactly one
`here-N` marker is shown -- the layer whose PR this comment is being posted on.

Layer 1 is the BOTTOM of the stack: its base is `main` and it merges first.
Layer N is the top; on GitHub's native stacks merging it lands every layer
below in one operation.
"""
import argparse
import html
import json
import re
import sys

MAX_LAYERS = 6
SLOT_STEP = 96      # layer height 76 + gap 20, must match stack-map.svg
CANVAS_H = 870      # template height with all six slots visible


def set_text(svg: str, id_: str, value: str) -> str:
    """Replace the text content of the element carrying this id."""
    pat = re.compile(r'(<text[^>]*id="%s"[^>]*>)(.*?)(</text>)' % re.escape(id_), re.S)
    if not pat.search(svg):
        raise SystemExit(f"template has no text id={id_}")
    return pat.sub(lambda m: m.group(1) + html.escape(value) + m.group(3), svg, count=1)


def hide(svg: str, group_id: str) -> str:
    """Set display="none", REPLACING any display already present.

    Appending a second attribute yields `display="inline" display="none"`, which
    librsvg refuses outright with "Attribute display redefined" -- it rasterises
    nothing rather than ignoring the duplicate. Learned the hard way on the
    certificate card.
    """
    pat = re.compile(r'<g([^>]*?)id="%s"([^>]*?)>' % re.escape(group_id))
    m = pat.search(svg)
    if not m:
        raise SystemExit(f"template has no group id={group_id}")
    head = re.sub(r'\s*display="[^"]*"', "", m.group(1))
    tail = re.sub(r'\s*display="[^"]*"', "", m.group(2))
    return svg[: m.start()] + f'<g{head}id="{group_id}"{tail} display="none">' + svg[m.end():]


def show(svg: str, group_id: str) -> str:
    """Drop a display="none" the template ships, so a filled slot is visible."""
    pat = re.compile(r'<g([^>]*?)id="%s"([^>]*?)>' % re.escape(group_id))
    m = pat.search(svg)
    if not m:
        raise SystemExit(f"template has no group id={group_id}")
    head = re.sub(r'\s*display="none"', "", m.group(1))
    tail = re.sub(r'\s*display="none"', "", m.group(2))
    return svg[: m.start()] + f'<g{head}id="{group_id}"{tail}>' + svg[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True,
                    help='JSON array, BOTTOM FIRST: [{"num":871,"title":"…","author":"tbraun96"}, …]')
    ap.add_argument("--here", type=int, required=True,
                    help="PR number this map is being posted on; it gets the 'you are here' ring")
    ap.add_argument("--campaign", default="",
                    help='footer line; defaults to "One campaign certifies all N."')
    args = ap.parse_args()

    layers = json.loads(args.layers)
    if not layers:
        raise SystemExit("--layers is empty; a stack needs at least one pull request")
    if len(layers) > MAX_LAYERS:
        # Refuse rather than silently truncate: a map that omits a layer is worse
        # than no map, because it reads as complete.
        raise SystemExit(
            f"{len(layers)} layers but the template holds {MAX_LAYERS}. "
            f"Add slots to stack-map.svg rather than dropping a layer from the picture."
        )

    svg = open(args.template, encoding="utf-8").read()

    here_found = False
    for i in range(1, MAX_LAYERS + 1):
        if i <= len(layers):
            L = layers[i - 1]
            svg = set_text(svg, f"field-layer-{i}-num", f"#{L['num']}")
            svg = set_text(svg, f"field-layer-{i}-title", L.get("title", "")[:78])
            svg = set_text(svg, f"field-layer-{i}-author", "@" + L.get("author", ""))
            if int(L["num"]) == args.here:
                svg = show(svg, f"here-{i}")
                here_found = True
            else:
                svg = hide(svg, f"here-{i}")
        else:
            svg = hide(svg, f"layer-{i}")

    if not here_found:
        # The whole point of the map is "where am I"; a map with no marker is a
        # silent lie about the reader's position.
        raise SystemExit(f"--here {args.here} is not one of the layers: "
                         f"{[l['num'] for l in layers]}")

    footer = args.campaign or f"One campaign certifies all {len(layers)}."
    svg = set_text(svg, "field-campaign", footer)

    # COLLAPSE the unused slots instead of leaving dead space above the stack.
    # The template lays out six slots with layer 1 anchored at the bottom, so a
    # two-layer stack would otherwise render four empty slots of blank canvas.
    # Slide the whole body up by the unused height and shrink the canvas to
    # match -- the header stays put because it lives outside `stack-body`.
    #
    # Caught by LOOKING at the rendered PNG: every assertion passed while the
    # picture had a third of a page of void and a footer cropped off the bottom.
    unused = MAX_LAYERS - len(layers)
    if unused:
        shift = unused * SLOT_STEP
        svg = re.sub(r'<g id="stack-body">',
                     f'<g id="stack-body" transform="translate(0 -{shift})">', svg, count=1)
        h = CANVAS_H - shift
        svg = re.sub(r'viewBox="0 0 1200 \d+"', f'viewBox="0 0 1200 {h}"', svg, count=1)
        svg = re.sub(r'(<svg[^>]*?)height="\d+"', rf'\1height="{h}"', svg, count=1)
        svg = re.sub(r'(<rect id="sm-bg"[^>]*?)height="\d+"', rf'\1height="{h}"', svg, count=1)

    open(args.out, "w", encoding="utf-8").write(svg)
    print(f"wrote {args.out}: {len(layers)} layers, marker on #{args.here}", file=sys.stderr)


if __name__ == "__main__":
    main()
