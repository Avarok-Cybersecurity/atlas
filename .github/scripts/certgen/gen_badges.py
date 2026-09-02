#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate Atlas PR-certification masthead components (header band, QR plaque,
silver stamp, gold seal) and inject them into copies of the state diagrams.

All badge geometry is original artwork drawn programmatically (arcs, scallops,
guilloche sine rings). QR module data comes from segno (BSD-3-Clause) at
generation time; only path data derived from the URL is embedded.

Layout (1200-wide canvas, body translated down by 112, height 928 -> 1040):
  - title block band: x 64..952, y 92..176
  - QR plaque: x 968..1132, y 8..172 (164px; 4px/module incl. 4-module quiet
    zone; 3.0 px/module after the 0.75x downscale to 900px, integer-aligned)
  - gold seal center (706,110) rotate 3.5deg; silver stamp center (866,116)
    rotate -5deg; both stay left of x=962, clear of the plaque.
"""
import math, os, re, subprocess

import segno

SRC = "/workspace/stackA/docs/diagrams/states"
OUT = "/workspace/tmp/cert-badges"
os.makedirs(OUT, exist_ok=True)

F = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")

def pt(r, deg):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)  # y-down SVG coords

def arc_path(r, a0, a1, sweep, large):
    x0, y0 = pt(r, a0); x1, y1 = pt(r, a1)
    return f"M {F(x0)} {F(y0)} A {F(r)} {F(r)} 0 {large} {sweep} {F(x1)} {F(y1)}"

def scallop_path(n, rb, bump=0.62, sweep=1):
    """Closed rosette silhouette: n outward arc bumps on circle radius rb."""
    chord = 2 * rb * math.sin(math.pi / n)
    ra = chord * bump
    pts = [pt(rb, i * 360.0 / n) for i in range(n)]
    d = [f"M {F(pts[0][0])} {F(pts[0][1])}"]
    for i in range(1, n + 1):
        x, y = pts[i % n]
        d.append(f"A {F(ra)} {F(ra)} 0 0 {sweep} {F(x)} {F(y)}")
    d.append("Z")
    return " ".join(d)

def ring_text(txt, r, fs, fill, mode="top", tracking=2.0, weight=700):
    """Per-glyph circular text (librsvg has no textPath support).
    mode=top: centered on 12 o'clock, glyphs upright facing outward.
    mode=bottom: centered on 6 o'clock, glyphs upright facing inward."""
    adv = 0.602 * fs + tracking
    dphi = math.degrees(adv / r)
    n = len(txt)
    out = [f'<g fill="{fill}" font-size="{fs}" font-weight="{weight}" '
           f'font-family="{MONO}" text-anchor="middle">']
    for i, ch in enumerate(txt):
        if ch == " ":
            continue
        c = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}.get(ch, ch)
        off = (i - (n - 1) / 2) * dphi
        if mode == "top":
            out.append(f'<text transform="rotate({F(off)}) translate(0 {F(-r)})">{c}</text>')
        else:
            out.append(f'<text transform="rotate({F(-off)}) translate(0 {F(r)})">{c}</text>')
    out.append("</g>")
    return "\n    ".join(out)

def sine_ring(rc, amp, n, phase=0.0, step=2):
    d = []
    for i in range(0, 361, step):
        th = math.radians(i)
        r = rc + amp * math.sin(n * th + phase)
        d.append(("M" if i == 0 else "L") + f" {F(r*math.cos(th))} {F(r*math.sin(th))}")
    d.append("Z")
    return " ".join(d)

# ------------------------------------------------------------------ QR ------
QR_URL = "https://github.com/Avarok-Cybersecurity/atlas/pull/840"
QR_X, QR_Y, QR_M = 968, 8, 4  # plaque origin, module size

def qr_group(url=QR_URL, x=QR_X, y=QR_Y, m=QR_M, side=None):
    """side=None: plaque = modules + exact 4-module quiet zone, module origin at
    x+4m (so x,y must be multiples of m for 900px integer alignment).
    side=<px>: plaque is that big and the module origin snaps to the nearest
    multiple of m independently of the plaque origin — lets the plaque sit at
    ANY integer x,y (e.g. equidistant corner margins) while module edges still
    land on integer device pixels at both 1200 and 900 widths. Quiet zone must
    stay >= 4 modules on every side (asserted)."""
    q = segno.make(url, error="m")
    rows = [bytearray(r) for r in q.matrix]
    n = len(rows)
    assert n == 33 and all(len(r) == n for r in rows), "expected version-4 QR"
    if side is None:
        side = (n + 8) * m  # 4-module quiet zone each side
        ox, oy = x + 4 * m, y + 4 * m
    else:
        ox = m * round((x + (side - n * m) / 2) / m)
        oy = m * round((y + (side - n * m) / 2) / m)
        for lo, o, hi in ((x, ox, x + side), (y, oy, y + side)):
            assert o - lo >= 4 * m and hi - (o + n * m) >= 4 * m, \
                f"quiet zone < 4 modules (plaque {lo}, modules {o}, side {side})"
    d = []
    for j, row in enumerate(rows):
        i = 0
        while i < n:
            if row[i]:
                run = 1
                while i + run < n and row[i + run]:
                    run += 1
                d.append(f"M{F(ox + i*m)} {F(oy + j*m)}h{F(run*m)}v{F(m)}h-{F(run*m)}z")
                i += run
            else:
                i += 1
    return (f'  <g id="qr">\n'
            f'    <rect x="{x}" y="{y}" width="{side}" height="{side}" rx="6" '
            f'fill="#FFFFFF" stroke="#2E353F" stroke-width="1"/>\n'
            f'    <path fill="#0F1216" d="{"".join(d)}"/>\n'
            f'  </g>\n'), side

# ---------------------------------------------------------------- defs ------
DEFS = f"""  <defs id="cert-defs">
    <linearGradient id="cert-silverRim" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#F7F9FC"/><stop offset="0.18" stop-color="#C9CFD8"/>
      <stop offset="0.38" stop-color="#8A919C"/><stop offset="0.52" stop-color="#E2E6EC"/>
      <stop offset="0.68" stop-color="#757C87"/><stop offset="0.85" stop-color="#B9BFC9"/>
      <stop offset="1" stop-color="#626976"/>
    </linearGradient>
    <radialGradient id="cert-silverPlate" cx="0.42" cy="0.38" r="0.78">
      <stop offset="0" stop-color="#E6EAF0"/><stop offset="0.55" stop-color="#C4CAD3"/>
      <stop offset="1" stop-color="#99A0AC"/>
    </radialGradient>
    <linearGradient id="cert-goldRim" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFF3C8"/><stop offset="0.2" stop-color="#F2C95F"/>
      <stop offset="0.4" stop-color="#B97F1C"/><stop offset="0.55" stop-color="#FFE08A"/>
      <stop offset="0.72" stop-color="#9C6B12"/><stop offset="0.88" stop-color="#EFB338"/>
      <stop offset="1" stop-color="#7A5210"/>
    </linearGradient>
    <radialGradient id="cert-goldField" cx="0.42" cy="0.36" r="0.8">
      <stop offset="0" stop-color="#FBE7A9"/><stop offset="0.5" stop-color="#F0C662"/>
      <stop offset="1" stop-color="#C88E1E"/>
    </radialGradient>
    <radialGradient id="cert-shadow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#000000" stop-opacity="0.55"/>
      <stop offset="0.72" stop-color="#000000" stop-opacity="0.3"/>
      <stop offset="1" stop-color="#000000" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="cert-fadeHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#171B21" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#171B21" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="cert-fadeRibAg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#59606B" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#59606B" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="cert-fadeRibAu" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#9C6B12" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#9C6B12" stop-opacity="1"/>
    </linearGradient>
    <clipPath id="cert-clip-pr"><rect x="300" y="122" width="122" height="38"/></clipPath>
    <clipPath id="cert-clip-author"><rect x="460" y="126" width="134" height="32"/></clipPath>
    <clipPath id="cert-clip-stamp-user"><rect x="-68" y="-10" width="122" height="22"/></clipPath>
    <clipPath id="cert-clip-stamp-sha"><rect x="-30" y="20" width="60" height="14"/></clipPath>
    <clipPath id="cert-clip-seal-user"><rect x="-80" y="-10" width="140" height="22"/></clipPath>
    <clipPath id="cert-clip-seal-sha"><rect x="-30" y="20" width="60" height="14"/></clipPath>
  </defs>
"""

MONO = "ui-mono, ui-monospace, SFMono-Regular, Menlo, monospace"

# -------------------------------------------------------------- header ------
def header_group(pr="#0000", author="author-login"):
    return f"""  <g id="field-header">
    <rect x="64" y="92" width="888" height="84" rx="3" fill="#171B21" stroke="#2E353F" stroke-width="1.4"/>
    <rect x="67.5" y="95.5" width="881" height="77" fill="none" stroke="#22282F" stroke-width="1"/>
    <line x1="280" y1="92" x2="280" y2="176" stroke="#2E353F" stroke-width="1"/>
    <line x1="440" y1="92" x2="440" y2="176" stroke="#2E353F" stroke-width="1"/>
    <line x1="620" y1="92" x2="620" y2="176" stroke="#2E353F" stroke-width="1"/>
    <text x="84" y="117" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">DOCUMENT</text>
    <text x="84" y="150" font-size="15" fill="#C9CCD4" font-weight="650">PR Certification Record</text>
    <text x="300" y="117" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">PULL REQUEST</text>
    <text x="300" y="154" font-size="26" fill="#C9CCD4" font-weight="700" clip-path="url(#cert-clip-pr)" id="value-pr-number">{pr}</text>
    <rect x="394" y="124" width="28" height="34" fill="url(#cert-fadeHdr)"/>
    <text x="460" y="117" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">AUTHOR</text>
    <text x="460" y="150" font-size="13.5" fill="#C9CCD4" font-family="{MONO}" clip-path="url(#cert-clip-author)" id="value-pr-author">{author}</text>
    <rect x="566" y="127" width="28" height="30" fill="url(#cert-fadeHdr)"/>
    <text x="640" y="117" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">APPROVALS</text>
    <g id="cert-approvals-placeholders" stroke="#2E353F" fill="none">
      <circle cx="700" cy="134" r="30" stroke-dasharray="3 4"/>
      <circle cx="872" cy="134" r="30" stroke-dasharray="3 4"/>
      <path d="M 630 102 h 8 M 634 98 v 8 M 934 102 h 8 M 938 98 v 8 M 630 166 h 8 M 634 162 v 8 M 934 166 h 8 M 938 162 v 8" stroke="#2E353F" stroke-width="1"/>
    </g>
    <text x="700" y="131" font-size="8" fill="#4A525C" font-weight="600" letter-spacing="1.5" text-anchor="middle">SEAL</text>
    <text x="700" y="142" font-size="6.5" fill="#3C434D" text-anchor="middle">codeowner /seal</text>
    <text x="872" y="131" font-size="8" fill="#4A525C" font-weight="600" letter-spacing="1.5" text-anchor="middle">STAMP</text>
    <text x="872" y="142" font-size="6.5" fill="#3C434D" text-anchor="middle">/stamp</text>
  </g>
"""

# ------------------------------------------------------------ silver --------
def stamp_group(user="stamper-login", sha="0000000000",
                transform='translate(872 116) rotate(-5)', hidden=False):
    disp = ' display="none"' if hidden else ""
    spec = arc_path(64, 185, 250, 1, 0)
    return f"""  <g id="field-stamp" transform="{transform}"{disp}>
    <circle cx="4" cy="6" r="76" fill="url(#cert-shadow)"/>
    <circle r="70" fill="url(#cert-silverRim)" stroke="#4A505A" stroke-width="1"/>
    <circle r="67.5" fill="none" stroke="#3E444E" stroke-width="1" stroke-dasharray="2 2.2"/>
    <circle r="58" fill="url(#cert-silverPlate)" stroke="#6E747E" stroke-width="1"/>
    <circle r="55.5" fill="none" stroke="#FFFFFF" stroke-width="0.9" opacity="0.55"/>
    <circle r="54.5" fill="none" stroke="#40454E" stroke-width="0.9" opacity="0.45"/>
    <path d="{spec}" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.4"/>
    {ring_text("PR CERTIFICATION", 45, 8.5, "#454B55", "top")}
    {ring_text("\u25c6 ATLAS /stamp \u25c6", 55, 8.5, "#454B55", "bottom")}
    <circle r="40.5" fill="none" stroke="#8A9099" stroke-width="0.8" opacity="0.7"/>
    <text y="-22" font-size="7.5" fill="#454B55" font-weight="700" letter-spacing="1.6" text-anchor="middle">STAMPED BY</text>
    <rect x="58" y="-11" width="9" height="24" fill="#000000" opacity="0.18"/>
    <rect x="-67" y="-11" width="9" height="24" fill="#000000" opacity="0.18"/>
    <path d="M -80 -11 L 80 -11 L 72 1 L 80 13 L -80 13 L -72 1 Z" fill="#59606B" stroke="#3A3F48" stroke-width="1"/>
    <line x1="-76" y1="-9.4" x2="76" y2="-9.4" stroke="#C7CCD4" stroke-width="0.8" opacity="0.5"/>
    <line x1="-76" y1="11.4" x2="76" y2="11.4" stroke="#23272E" stroke-width="0.8" opacity="0.6"/>
    <text x="-68" y="5.5" font-size="12.5" font-weight="600" fill="#EFF2F6" font-family="{MONO}" clip-path="url(#cert-clip-stamp-user)" id="value-stamp-user">{user}</text>
    <rect x="26" y="-9" width="28" height="20" fill="url(#cert-fadeRibAg)"/>
    <rect x="-33" y="19.5" width="66" height="15.5" rx="2" fill="#AEB4BE" stroke="#6E747E" stroke-width="0.8"/>
    <text x="-28.5" y="31" font-size="9.5" fill="#232830" font-family="{MONO}" clip-path="url(#cert-clip-stamp-sha)" id="value-stamp-sha">{sha}</text>
  </g>
"""

# -------------------------------------------------------------- gold --------
def seal_group(user="sealer-login", sha="0000000000",
               transform='translate(700 110) rotate(3.5)', hidden=False):
    disp = ' display="none"' if hidden else ""
    spec = arc_path(71, 185, 250, 1, 0)
    g1 = sine_ring(42, 3.2, 18, 0.0)
    g2 = sine_ring(42, 3.2, 18, math.pi)
    sc = scallop_path(26, 74)
    return f"""  <g id="field-seal" transform="{transform}"{disp}>
    <circle cx="4" cy="6" r="84" fill="url(#cert-shadow)"/>
    <path d="{sc}" fill="url(#cert-goldRim)" stroke="#7A5210" stroke-width="1"/>
    <circle r="69" fill="none" stroke="#7A5210" stroke-width="1" stroke-dasharray="0.6 4.2" opacity="0.8"/>
    <circle r="64" fill="url(#cert-goldField)" stroke="#8A5E12" stroke-width="1.2"/>
    <circle r="61" fill="none" stroke="#FFF0C0" stroke-width="0.9" opacity="0.5"/>
    <circle r="60" fill="none" stroke="#6B4A0E" stroke-width="0.9" opacity="0.45"/>
    <path d="{spec}" fill="none" stroke="#FFF8DC" stroke-width="4" stroke-linecap="round" opacity="0.45"/>
    {ring_text("CODEOWNER SEAL", 49, 9, "#4A3407", "top")}
    {ring_text("\u25c6 ATLAS /seal \u25c6", 58, 9, "#4A3407", "bottom")}
    <path d="{g1}" fill="none" stroke="#A66B14" stroke-width="0.8" opacity="0.75"/>
    <path d="{g2}" fill="none" stroke="#A66B14" stroke-width="0.8" opacity="0.75"/>
    <circle r="37" fill="none" stroke="#A66B14" stroke-width="0.7" opacity="0.6"/>
    <text y="-22" font-size="7.5" fill="#6B4A0E" font-weight="700" letter-spacing="1.6" text-anchor="middle">SEALED BY</text>
    <rect x="64" y="-11" width="9" height="24" fill="#000000" opacity="0.18"/>
    <rect x="-73" y="-11" width="9" height="24" fill="#000000" opacity="0.18"/>
    <path d="M -94 -11 L 94 -11 L 86 1 L 94 13 L -94 13 L -86 1 Z" fill="#9C6B12" stroke="#6B4A0E" stroke-width="1"/>
    <line x1="-90" y1="-9.4" x2="90" y2="-9.4" stroke="#FFE9A8" stroke-width="0.8" opacity="0.5"/>
    <line x1="-90" y1="11.4" x2="90" y2="11.4" stroke="#4A3407" stroke-width="0.8" opacity="0.6"/>
    <text x="-80" y="5.5" font-size="12.5" font-weight="600" fill="#FFF3D0" font-family="{MONO}" clip-path="url(#cert-clip-seal-user)" id="value-seal-user">{user}</text>
    <rect x="32" y="-9" width="28" height="20" fill="url(#cert-fadeRibAu)"/>
    <rect x="-33" y="19.5" width="66" height="15.5" rx="2" fill="#F3D488" stroke="#8A5E12" stroke-width="0.8"/>
    <text x="-28.5" y="31" font-size="9.5" fill="#4A3407" font-family="{MONO}" clip-path="url(#cert-clip-seal-sha)" id="value-seal-sha">{sha}</text>
  </g>
"""

# ----------------------------------------------------------- injection ------
VB_OLD = 'viewBox="0 0 1200 928" width="1200" height="928"'
VB_NEW = 'viewBox="0 0 1200 1040" width="1200" height="1040"'
BG_OLD = '<rect width="1200" height="928" fill="#0F1216"/>'
BG_NEW = '<rect width="1200" height="1040" fill="#0F1216"/>'
SUBTITLE_RE = re.compile(r'( {2}<text x="156" y="66"[^\n]*</text>\n)')

def inject(src_text, pr, author, stamp=None, seal=None, url=QR_URL):
    assert src_text.count(VB_OLD) == 1, "viewBox anchor"
    assert src_text.count(BG_OLD) == 1, "bg anchor"
    assert src_text.count("</svg>") == 1, "closing anchor"
    assert len(SUBTITLE_RE.findall(src_text)) == 1, "subtitle anchor"
    s = src_text.replace(VB_OLD, VB_NEW).replace(BG_OLD, BG_NEW)
    stamp_g = stamp_group(*stamp) if stamp else stamp_group(hidden=True)
    seal_g = seal_group(*seal) if seal else seal_group(hidden=True)
    qr, _ = qr_group(url)
    block = (DEFS + header_group(pr, author) + qr + stamp_g + seal_g
             + '  <g id="cert-body" transform="translate(0 112)">\n')
    s = SUBTITLE_RE.sub(lambda m: m.group(1) + block, s, count=1)
    s = s.replace("</svg>", "  </g>\n</svg>")
    return s

LOGO = """  <g transform="translate(64 22) scale(0.06918)">
    <g fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round">
      <path d="M38 38L358 318L38 598" stroke="#BE9DF8"/>
      <path d="M318 38L638 318L318 598" stroke="#49C3DB"/>
      <path d="M598 38L918 318L598 598" stroke="url(#cert-atlasGoldCut)"/>
    </g>
  </g>
"""

def standalone_header(pr, author):
    qr, _ = qr_group()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 1200 208" width="1200" height="208" '
            f'font-family="Inter, ui-sans-serif, system-ui, Segoe UI, Helvetica, Arial, sans-serif" '
            f'role="img" aria-label="Atlas PR certification header band">\n'
            f'  <title>Atlas — PR certification header band</title>\n'
            + DEFS
            + '  <linearGradient id="cert-atlasGoldCut" x1="0" y1="0" x2="0" y2="1">'
              '<stop offset="0.5" stop-color="#12B981"/><stop offset="0.5" stop-color="#EFB338"/></linearGradient>\n'
            + '  <rect width="1200" height="208" fill="#0F1216"/>\n'
            + LOGO
            + '  <text x="156" y="44" font-size="23" fill="#C9CCD4" font-weight="650">PR Certification</text>\n'
            + '  <text x="156" y="66" font-size="13.5" fill="#82868F">How a change gets from opened to merged, and what knocks it back.</text>\n'
            + header_group(pr, author)
            + qr
            + '</svg>\n')

def standalone_badge(kind):
    if kind == "stamp":
        body = stamp_group("m-ferraro", "3f9c2d81ab", transform="translate(150 108) rotate(-5)")
        w, h, label = 300, 216, "Atlas silver certification stamp"
    else:
        body = seal_group("a-hoffmann", "3f9c2d81ab", transform="translate(160 112) rotate(3.5)")
        w, h, label = 320, 232, "Atlas gold codeowner seal"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="Inter, ui-sans-serif, system-ui, Segoe UI, Helvetica, Arial, sans-serif" '
            f'role="img" aria-label="{label}">\n  <title>{label}</title>\n'
            + DEFS + f'  <rect width="{w}" height="{h}" fill="#0F1216"/>\n' + body + '</svg>\n')

def standalone_qr():
    qr, side = qr_group(x=18, y=18)
    w = side + 36
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w}" width="{w}" height="{w}" '
            f'role="img" aria-label="QR code linking to the pull request">\n'
            f'  <title>Atlas PR link QR plaque</title>\n'
            f'  <rect width="{w}" height="{w}" fill="#0F1216"/>\n' + qr + '</svg>\n')

# ------------------------------------------------------------- build --------
def read(p):
    with open(p) as f: return f.read()

def write(p, s):
    with open(p, "w") as f: f.write(s)
    print("wrote", p)

stage1 = read(f"{SRC}/pr-certification-stage-1.svg")
stage2 = read(f"{SRC}/pr-certification-stage-2-both.svg")
stage3 = read(f"{SRC}/pr-certification-stage-3.svg")

write(f"{OUT}/header.svg", standalone_header("#840", "tbraun96"))
write(f"{OUT}/stamp-silver.svg", standalone_badge("stamp"))
write(f"{OUT}/seal-gold.svg", standalone_badge("seal"))
write(f"{OUT}/qr-demo.svg", standalone_qr())

write(f"{OUT}/preview-none.svg",
      inject(stage1, "#840", "tbraun96"))
write(f"{OUT}/preview-stamped.svg",
      inject(stage2, "#840", "tbraun96", stamp=("m-ferraro", "3f9c2d81ab")))
write(f"{OUT}/preview-sealed.svg",
      inject(stage3, "#840", "tbraun96",
             stamp=("m-ferraro", "3f9c2d81ab"), seal=("a-hoffmann", "3f9c2d81ab")))

LONG = "an-extraordinarily-long-github-username"
assert len(LONG) == 39, len(LONG)
write(f"{OUT}/preview-longnames.svg",
      inject(stage3, "#840", LONG, stamp=(LONG, "3f9c2d81ab"), seal=(LONG, "3f9c2d81ab")))

for name in ["preview-none", "preview-stamped", "preview-sealed", "preview-longnames",
             "header", "stamp-silver", "seal-gold", "qr-demo"]:
    subprocess.run(["rsvg-convert", "-w", "900", f"{OUT}/{name}.svg",
                    "-o", f"{OUT}/{name}.png"], check=True)
    print("rendered", name)

subprocess.run(["cp", f"{OUT}/preview-sealed.png", f"{OUT}/qr-scan-test.png"], check=True)
print("done")
