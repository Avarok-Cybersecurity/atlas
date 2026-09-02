#!/usr/bin/env python3
"""Merged-state certificate (landscape 1200x675 + square 1200x1200).
Imports the badge/QR components from gen_badges (which also rebuilds the
state-diagram previews as a side effect of import — idempotent)."""
import math, subprocess

import gen_badges as G

OUT = G.OUT
F, MONO, DEFS = G.F, G.MONO, G.DEFS


def wave_band(x0, x1, y, amp=6.5, wl=46.0, phase=0.0, step=6):
    d, x = [], x0
    while x <= x1:
        yy = y + amp * math.sin((x - x0) / wl * 2 * math.pi + phase)
        d.append(("M" if x == x0 else "L") + f" {F(x)} {F(yy)}")
        x += step
    return " ".join(d)


def corners(pts):
    out = []
    for x, y, sx, sy in pts:
        out.append(
            f'  <g transform="translate({x} {y}) scale({sx} {sy})" stroke="#EFB338" fill="none">'
            f'<path d="M 0 24 L 0 0 L 24 0" stroke-width="1.5"/>'
            f'<path d="M 0 34 L 0 28 M 28 0 L 34 0" stroke-width="1" opacity="0.7"/></g>')
    return "\n".join(out)


EXTRA_DEFS = """    <clipPath id="cert-clip-a1"><rect x="64" y="272" width="780" height="36"/></clipPath>
    <clipPath id="cert-clip-a2"><rect x="64" y="316" width="780" height="36"/></clipPath>
    <clipPath id="cert-clip-a3"><rect x="64" y="360" width="780" height="36"/></clipPath>
    <clipPath id="cert-clip-prline"><rect x="64" y="440" width="784" height="27"/></clipPath>
    <clipPath id="cert-clip-repo"><rect x="80" y="530" width="270" height="32"/></clipPath>
    <linearGradient id="cert-fadeBg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#0F1216" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#0F1216" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="cert-fadeCell" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#000000" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#000000" stop-opacity="1"/>
    </linearGradient>
"""

LOGO = """  <g transform="translate({x} {y}) scale({s})">
    <g fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round">
      <path d="M38 38L358 318L38 598" stroke="#BE9DF8"/>
      <path d="M318 38L638 318L318 598" stroke="#49C3DB"/>
      <path d="M598 38L918 318L598 598" stroke="url(#cert-atlasGoldCut)"/>
    </g>
  </g>
"""

GOLDCUT = ('  <linearGradient id="cert-atlasGoldCut" x1="0" y1="0" x2="0" y2="1">'
           '<stop offset="0.5" stop-color="#12B981"/>'
           '<stop offset="0.5" stop-color="#EFB338"/></linearGradient>\n')


def cell(x, y, w, label, value, vid, clip=None, vfill="#C9CCD4", fs=13.5):
    clipattr = f' clip-path="url(#{clip})"' if clip else ""
    fade = (f'  <rect x="{x + w - 42}" y="{y + 1}" width="30" height="82" fill="url(#cert-fadeCell)"/>\n'
            if clip else "")
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="84" rx="4" fill="#000000" stroke="#2E353F" stroke-width="1"/>\n'
        f'  <rect x="{x}" y="{y}" width="3" height="84" fill="#EFB338"/>\n'
        f'  <text x="{x+16}" y="{y+24}" font-size="10" fill="#82868F" font-weight="600" letter-spacing="1.5">{label}</text>\n'
        f'  <text x="{x+16}" y="{y+58}" font-size="{fs}" fill="{vfill}" font-family="{MONO}"{clipattr} id="{vid}">{value}</text>\n'
        + fade)


def author_slots(authors, x, y0, dy, fs, clip_prefix, fade_x, anchor="start"):
    a = list(authors) + ["", ""]
    out = []
    anch = f' text-anchor="{anchor}"' if anchor != "start" else ""
    for i in range(3):
        hide = "" if i < len(authors) else ' display="none"'
        yy = y0 + dy * i
        out.append(
            f'  <g id="field-cert-author-{i+1}"{hide}>\n'
            f'    <text x="{x}" y="{yy}" font-size="{fs}" font-weight="700" fill="#EDF0F4"'
            f'{anch} font-family="{MONO}" clip-path="url(#{clip_prefix}{i+1})" '
            f'id="value-cert-author-{i+1}">{a[i]}</text>\n'
            f'    <rect x="{fade_x}" y="{yy-27}" width="36" height="36" fill="url(#cert-fadeBg)"/>\n'
            f'  </g>\n')
    return "".join(out)


SAMPLE = dict(pr="#840",
              pr_title="Fuse the GDN spine epilogue into the decode kernel",
              repo="Avarok-Cybersecurity/atlas", commit="9d4e1f07c2",
              date="2026-09-02", gates="11 / 11 CERTIFIED",
              stamp=("m-ferraro", "3f9c2d81ab"), seal=("a-hoffmann", "3f9c2d81ab"),
              merge_count=263)  # gh search/issues total_count for the OPENER


def ordinal(n):
    suf = "TH" if 10 <= n % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{n}{suf}"


def merge_count_row(count, opener, x, y, anchor="end", fs=10):
    """Opener's lifetime merged-PR ordinal. count=None -> display='none'
    (search API 422 on vanished logins / 30 req-min rate limit); the row it
    sits on keeps its own label, so nothing looks gappy when hidden."""
    hide = ' display="none"' if count is None else ""
    # hidden groups keep placeholder content, same convention as stamp/seal
    phrase = f"THE {ordinal(count if count is not None else SAMPLE['merge_count'])} MERGED INTO THIS REPOSITORY BY"
    anch = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'  <g id="field-cert-merge-count"{hide}>\n'
        f'    <text x="{x}" y="{y}" font-size="{fs}" fill="#EFB338" font-weight="600" '
        f'letter-spacing="1.5"{anch}>'
        f'<tspan id="value-cert-merge-count">{phrase}</tspan>'
        f'<tspan dx="6" letter-spacing="0" font-family="{MONO}" '
        f'id="value-cert-merge-author">{opener}</tspan></text>\n'
        f'  </g>\n')


def certificate(authors, **kw):
    o = dict(SAMPLE, **kw)
    # QR bottom-right, equidistant: 52px right margin == 52px bottom margin
    # (39 device px at the 900px render — OpenCV's detector needs ~38px of
    # image-edge clearance there; 38/31 SVG px measurably failed detection).
    # 168px plaque; module grid snaps to multiples of 4 inside it (see qr_group).
    qr, _ = G.qr_group(G.QR_URL, x=980, y=455, side=168)
    defs = DEFS.replace("  </defs>", EXTRA_DEFS + "  </defs>")
    cells = (cell(64, 500, 300, "REPOSITORY", o["repo"], "value-cert-repo", clip="cert-clip-repo", fs=12.5)
             + cell(378, 500, 160, "MERGE COMMIT", o["commit"], "value-cert-commit")
             + cell(552, 500, 150, "MERGED ON", o["date"], "value-cert-date")
             + cell(716, 500, 170, "GATES", o["gates"], "value-cert-gates", vfill="#12B981"))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'viewBox="0 0 1200 675" width="1200" height="675" '
        'font-family="Inter, ui-sans-serif, system-ui, Segoe UI, Helvetica, Arial, sans-serif" '
        'role="img" aria-label="Atlas certificate of certified merge">\n'
        '  <title>Atlas — Certificate of Certified Merge</title>\n'
        + defs + GOLDCUT
        + '  <rect width="1200" height="675" fill="#0F1216"/>\n'
        + '  <rect x="14" y="14" width="1172" height="647" fill="none" stroke="#2E353F" stroke-width="1.5"/>\n'
        + '  <rect x="24" y="24" width="1152" height="627" fill="none" stroke="#EFB338" stroke-width="0.8" opacity="0.55"/>\n'
        + '  <g opacity="0.16" stroke="#EFB338" stroke-width="1" fill="none">\n'
        + f'    <path d="{wave_band(34, 1166, 47, 6.5, 46, 0)}"/>\n'
        + f'    <path d="{wave_band(34, 1166, 47, 6.5, 46, math.pi)}"/>\n'
        + '  </g>\n'
        + corners([(34, 34, 1, 1), (1166, 34, -1, 1), (34, 641, 1, -1), (1166, 641, -1, -1)]) + "\n"
        + LOGO.format(x=64, y=66, s=0.052)
        + '  <text x="126" y="88" font-size="11" fill="#82868F" font-weight="600" letter-spacing="3.5">ATLAS ENGINEERING &#183; PULL REQUEST CERTIFICATION</text>\n'
        + '  <text x="64" y="156" font-size="38" fill="#EFB338" font-weight="800" letter-spacing="4">CERTIFIED MERGE</text>\n'
        + '  <line x1="64" y1="172" x2="474" y2="172" stroke="#EFB338" stroke-width="1" opacity="0.5"/>\n'
        + '  <text x="64" y="204" font-size="13.5" fill="#C9CCD4">Verified, stamped, sealed by a codeowner, and merged to main &#8212;</text>\n'
        + '  <text x="64" y="224" font-size="13.5" fill="#C9CCD4">with <tspan fill="#EFB338" font-weight="650">all eleven benchmark gates certified</tspan> on GB10 hardware.</text>\n'
        + '  <text x="64" y="274" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">AUTHORED BY</text>\n'
        + author_slots(authors, 64, 300, 44, 32, "cert-clip-a", 808)
        + '  <text x="64" y="436" font-size="10" fill="#82868F" font-weight="600" letter-spacing="2">PULL REQUEST</text>\n'
        + merge_count_row(o["merge_count"], authors[0], 848, 436)
        + f'  <text x="64" y="460" font-size="17" clip-path="url(#cert-clip-prline)">'
          f'<tspan fill="#EFB338" font-weight="700" id="value-cert-pr">{o["pr"]}</tspan>'
          f'<tspan dx="10" fill="#C9CCD4" id="value-cert-pr-title">{o["pr_title"]}</tspan></text>\n'
        + '  <rect x="812" y="441" width="36" height="26" fill="url(#cert-fadeBg)"/>\n'
        + cells
        + '  <text x="964" y="596" font-size="9.5" fill="#82868F" font-weight="600" letter-spacing="1.5" text-anchor="end">SCAN TO READ</text>\n'
        + '  <text x="964" y="612" font-size="9.5" fill="#82868F" font-weight="600" letter-spacing="1.5" text-anchor="end">THE MERGED PR</text>\n'
        + G.stamp_group(*o["stamp"], transform="translate(1078 352) rotate(-5) scale(1.15)")
        + G.seal_group(*o["seal"], transform="translate(952 150) rotate(3.5) scale(1.25)")
        + '  <text x="520" y="642" font-size="10.5" fill="#82868F" text-anchor="middle">Issued by Atlas Cybernetics Corp &#183; <tspan fill="#49C3DB">atlasinference.io</tspan></text>\n'
        + qr
        + '</svg>\n')


# ------------------------------------------------------------- square -------
SQ_DEFS = """    <clipPath id="certsq-clip-a1"><rect x="150" y="460" width="900" height="42"/></clipPath>
    <clipPath id="certsq-clip-a2"><rect x="150" y="512" width="900" height="42"/></clipPath>
    <clipPath id="certsq-clip-a3"><rect x="150" y="564" width="900" height="42"/></clipPath>
    <clipPath id="certsq-clip-prline"><rect x="150" y="628" width="900" height="28"/></clipPath>
    <clipPath id="certsq-clip-repo"><rect x="96" y="1000" width="240" height="32"/></clipPath>
    <linearGradient id="cert-fadeBg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#0F1216" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#0F1216" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="cert-fadeCell" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#000000" stop-opacity="0"/>
      <stop offset="0.85" stop-color="#000000" stop-opacity="1"/>
    </linearGradient>
"""


def certificate_square(authors, **kw):
    o = dict(SAMPLE, **kw)
    # QR bottom-right, same 52px right == bottom margin as the landscape.
    qr, _ = G.qr_group(G.QR_URL, x=980, y=980, side=168)
    defs = DEFS.replace("  </defs>", SQ_DEFS + "  </defs>")
    slots = author_slots(authors, 600, 490, 52, 36, "certsq-clip-a", 1014, anchor="middle")
    cells = (cell(80, 970, 280, "REPOSITORY", o["repo"], "value-cert-repo", clip="certsq-clip-repo", fs=12.5)
             + cell(374, 970, 190, "MERGE COMMIT", o["commit"], "value-cert-commit")
             + cell(578, 970, 180, "MERGED ON", o["date"], "value-cert-date")
             + cell(772, 970, 180, "GATES", o["gates"], "value-cert-gates", vfill="#12B981"))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'viewBox="0 0 1200 1200" width="1200" height="1200" '
        'font-family="Inter, ui-sans-serif, system-ui, Segoe UI, Helvetica, Arial, sans-serif" '
        'role="img" aria-label="Atlas certificate of certified merge">\n'
        '  <title>Atlas — Certificate of Certified Merge</title>\n'
        + defs + GOLDCUT
        + '  <rect width="1200" height="1200" fill="#0F1216"/>\n'
        + '  <rect x="14" y="14" width="1172" height="1172" fill="none" stroke="#2E353F" stroke-width="1.5"/>\n'
        + '  <rect x="24" y="24" width="1152" height="1152" fill="none" stroke="#EFB338" stroke-width="0.8" opacity="0.55"/>\n'
        + '  <g opacity="0.16" stroke="#EFB338" stroke-width="1" fill="none">\n'
        + f'    <path d="{wave_band(34, 1166, 47, 6.5, 46, 0)}"/>\n'
        + f'    <path d="{wave_band(34, 1166, 47, 6.5, 46, math.pi)}"/>\n'
        + f'    <path d="{wave_band(34, 1166, 1153, 6.5, 46, 0)}"/>\n'
        + f'    <path d="{wave_band(34, 1166, 1153, 6.5, 46, math.pi)}"/>\n'
        + '  </g>\n'
        + corners([(34, 34, 1, 1), (1166, 34, -1, 1), (34, 1166, 1, -1), (1166, 1166, -1, -1)]) + "\n"
        + LOGO.format(x=576, y=88, s=0.052)
        + '  <text x="600" y="180" font-size="11" fill="#82868F" font-weight="600" letter-spacing="3.5" text-anchor="middle">ATLAS ENGINEERING &#183; PULL REQUEST CERTIFICATION</text>\n'
        + '  <text x="600" y="252" font-size="44" fill="#EFB338" font-weight="800" letter-spacing="5" text-anchor="middle">CERTIFIED MERGE</text>\n'
        + '  <line x1="380" y1="272" x2="820" y2="272" stroke="#EFB338" stroke-width="1" opacity="0.5"/>\n'
        + '  <text x="600" y="320" font-size="14.5" fill="#C9CCD4" text-anchor="middle">Verified, stamped, sealed by a codeowner, and merged to main &#8212;</text>\n'
        + '  <text x="600" y="342" font-size="14.5" fill="#C9CCD4" text-anchor="middle">with <tspan fill="#EFB338" font-weight="650">all eleven benchmark gates certified</tspan> on GB10 hardware.</text>\n'
        + '  <text x="600" y="438" font-size="10.5" fill="#82868F" font-weight="600" letter-spacing="2.5" text-anchor="middle">AUTHORED BY</text>\n'
        + slots
        + '  <text x="600" y="652" font-size="18" text-anchor="middle" clip-path="url(#certsq-clip-prline)">'
          f'<tspan fill="#EFB338" font-weight="700" id="value-cert-pr">{o["pr"]}</tspan>'
          f'<tspan dx="10" fill="#C9CCD4" id="value-cert-pr-title">{o["pr_title"]}</tspan></text>\n'
        + merge_count_row(o["merge_count"], authors[0], 600, 688, anchor="middle", fs=10.5)
        + G.seal_group(*o["seal"], transform="translate(420 810) rotate(3.5) scale(1.3)")
        + G.stamp_group(*o["stamp"], transform="translate(780 812) rotate(-5) scale(1.25)")
        + cells
        + '  <text x="1148" y="968" font-size="9.5" fill="#82868F" font-weight="600" letter-spacing="1.5" text-anchor="end">SCAN TO READ THE MERGED PR</text>\n'
        + '  <text x="600" y="1132" font-size="10.5" fill="#82868F" text-anchor="middle">Issued by Atlas Cybernetics Corp &#183; <tspan fill="#49C3DB">atlasinference.io</tspan></text>\n'
        + qr
        + '</svg>\n')


def write(p, s):
    with open(p, "w") as f:
        f.write(s)
    print("wrote", p)


write(f"{OUT}/certificate-merged.svg", certificate(["tbraun96"]))
write(f"{OUT}/certificate-merged-2authors.svg", certificate(["tbraun96", "m-ferraro"]))
write(f"{OUT}/certificate-merged-3authors.svg", certificate(["tbraun96", "m-ferraro", "j-okonkwo"]))
write(f"{OUT}/certificate-merged-square.svg", certificate_square(["tbraun96"]))
# merge-count fetch can fail (422 vanished login / search rate limit) — prove
# the layout still reads as deliberate with the field hidden:
write(f"{OUT}/certificate-merged-nocount.svg", certificate(["tbraun96"], merge_count=None))

for name, w in [("certificate-merged", 1200), ("certificate-merged-2authors", 1200),
                ("certificate-merged-3authors", 1200), ("certificate-merged-square", 1200),
                ("certificate-merged-nocount", 1200)]:
    subprocess.run(["rsvg-convert", "-w", str(w), f"{OUT}/{name}.svg",
                    "-o", f"{OUT}/{name}.png"], check=True)
    subprocess.run(["rsvg-convert", "-w", "900", f"{OUT}/{name}.svg",
                    "-o", f"{OUT}/{name}-900px.png"], check=True)
    print("rendered", name)
print("done")
