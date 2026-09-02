#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render a per-PR merge certificate from the committed template.

String substitution over stable ids, exactly as docs/certification-artwork.md
describes -- no SVG parsing, because the template is ours and its ids are a
contract. The ONE exception is the QR: its module matrix changes with the URL,
so that group is regenerated wholesale rather than substituted.

Every optional field hides by setting display="none" on its GROUP, never by
blanking the text: an empty <text> leaves the label stranded beside nothing,
which reads as a rendering bug rather than an absent value.
"""
import argparse
import html
import re
import sys

# The QR must clear the image border by >=38 device px at the 900px embed or
# OpenCV -- and phone cameras -- stop decoding it, even with the modules
# perfectly aligned. 52 SVG px is 39 device px. Measured, not guessed; see
# docs/certification-artwork.md section 4.
QR_MARGIN = 52
QR_PLAQUE = 168
QR_MODULE = 4


def qr_group(url, x, y):
    """Regenerate <g id="qr"> for `url`. Module origin stays on integer pixels
    at BOTH 1200 and 900 widths; a half-pixel origin renders identically and
    silently fails to scan."""
    import segno
    q = segno.make(url, error="m")
    m = [list(r) for r in q.matrix]
    n = len(m)
    px = x + (QR_PLAQUE - n * QR_MODULE) // 2
    py = y + (QR_PLAQUE - n * QR_MODULE) // 2
    d = []
    for r, row in enumerate(m):
        for c, v in enumerate(row):
            if v:
                d.append(f"M{px + c * QR_MODULE} {py + r * QR_MODULE}h{QR_MODULE}v{QR_MODULE}h-{QR_MODULE}z")
    return (
        f'<g id="qr">'
        f'<rect x="{x}" y="{y}" width="{QR_PLAQUE}" height="{QR_PLAQUE}" rx="6" '
        f'fill="#FFFFFF" stroke="#2E353F" stroke-width="1"/>'
        f'<path d="{"".join(d)}" fill="#0B0E12"/>'
        f"</g>"
    )


def set_text(svg, id_, value):
    """Replace the body of the element carrying `id_`."""
    pat = re.compile(r'(<(?:text|tspan)[^>]*id="%s"[^>]*>)(.*?)(</(?:text|tspan)>)' % re.escape(id_), re.S)
    if not pat.search(svg):
        raise SystemExit(f"template has no id={id_} -- the artwork and this script have diverged")
    return pat.sub(lambda m: m.group(1) + html.escape(str(value)) + m.group(3), svg, count=1)


def hide(svg, group_id):
    """Set display="none" on a group, REPLACING any display it already carries.

    The template ships some groups with display="inline" so the intent is
    visible in the source. Appending a second attribute instead of replacing
    produces `display="inline" display="none"`, which is not merely ignored --
    librsvg refuses the whole file with "Attribute display redefined" and
    rasterises nothing.
    """
    pat = re.compile(r'<g([^>]*?)id="%s"([^>]*?)>' % re.escape(group_id))
    m = pat.search(svg)
    if not m:
        raise SystemExit(f"template has no group id={group_id}")
    head = re.sub(r'\s*display="[^"]*"', "", m.group(1))
    tail = re.sub(r'\s*display="[^"]*"', "", m.group(2))
    return svg[: m.start()] + f'<g{head}id="{group_id}"{tail} display="none">' + svg[m.end():]


def show(svg, group_id):
    """Drop a display="none" the template ships, so a filled slot is visible."""
    pat = re.compile(r'<g([^>]*?)id="%s"([^>]*?)>' % re.escape(group_id))
    m = pat.search(svg)
    if not m:
        raise SystemExit(f"template has no group id={group_id}")
    head = re.sub(r'\s*display="none"', "", m.group(1))
    tail = re.sub(r'\s*display="none"', "", m.group(2))
    return svg[: m.start()] + f'<g{head}id="{group_id}"{tail}>' + svg[m.end():]


def ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}TH"
    return f"{n}{ {1: 'ST', 2: 'ND', 3: 'RD'}.get(n % 10, 'TH') }"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--gates", default="")
    ap.add_argument("--authors", default="", help="comma-separated logins, opener first")
    ap.add_argument("--merge-count", default="")
    ap.add_argument("--stamp-user", default="")
    ap.add_argument("--stamp-sha", default="")
    ap.add_argument("--seal-user", default="")
    ap.add_argument("--seal-sha", default="")
    ap.add_argument("--qr-x", type=int, default=0)
    ap.add_argument("--qr-y", type=int, default=0)
    a = ap.parse_args()

    svg = open(a.template, encoding="utf-8").read()

    for id_, val in (
        ("value-cert-pr", f"#{a.pr}"),
        ("value-cert-pr-title", a.title),
        ("value-cert-repo", a.repo),
        ("value-cert-commit", a.commit),
        ("value-cert-date", a.date),
        ("value-cert-gates", a.gates),
    ):
        if val:
            svg = set_text(svg, id_, val)

    authors = [x.strip() for x in a.authors.split(",") if x.strip()]
    for i in range(1, 4):
        if i <= len(authors):
            svg = set_text(svg, f"value-cert-author-{i}", authors[i - 1])
            # The template ships slots 2 and 3 hidden so a single-author
            # certificate is the default. Filling the text is not enough --
            # without this the second author is silently dropped.
            svg = show(svg, f"field-cert-author-{i}")
        else:
            svg = hide(svg, f"field-cert-author-{i}")

    # The count belongs to the OPENER, not to every co-author. Naming them in
    # the line is what keeps that unambiguous on a multi-author certificate.
    if a.merge_count and str(a.merge_count).isdigit() and authors:
        svg = set_text(svg, "value-cert-merge-count",
                       f"THE {ordinal(a.merge_count)} MERGED INTO THIS REPOSITORY BY")
        svg = set_text(svg, "value-cert-merge-author", authors[0])
    else:
        svg = hide(svg, "field-cert-merge-count")

    for user, sha, group, uid, sid in (
        (a.stamp_user, a.stamp_sha, "field-stamp", "value-stamp-user", "value-stamp-sha"),
        (a.seal_user, a.seal_sha, "field-seal", "value-seal-user", "value-seal-sha"),
    ):
        if user:
            svg = set_text(svg, uid, user)
            svg = set_text(svg, sid, sha or "")
        else:
            svg = hide(svg, group)

    if a.qr_x and a.qr_y:
        new = qr_group(a.url, a.qr_x, a.qr_y)
        svg = re.sub(r'<g id="qr">.*?</g>', lambda _: new, svg, count=1, flags=re.S)

    open(a.out, "w", encoding="utf-8").write(svg)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
