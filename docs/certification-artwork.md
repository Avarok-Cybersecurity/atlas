# Integrating the certification masthead into the 13 state diagrams

Components: **header band** (title block), **QR plaque**, **silver /stamp badge**,
**gold /seal badge** — plus the standalone **merged certificate**.
Reference implementations: `tools/gen_badges.py` (masthead + injection) and
`tools/cert_gen.py` (certificate). Both are deterministic; `python3 tools/cert_gen.py`
rebuilds every SVG/PNG in this directory (needs `segno`, `rsvg-convert`).

## 1. Canvas change — the body translates down by 112px

There is no usable whitespace at the top of the existing diagrams that fits a
scannable QR (164px) plus two legible badges without covering the stage banner
or the legend box, so the diagram body moves down instead:

| edit | from | to |
|---|---|---|
| root attrs | `viewBox="0 0 1200 928" width="1200" height="928"` | `viewBox="0 0 1200 1040" width="1200" height="1040"` |
| bg rect | `<rect width="1200" height="928" fill="#0F1216"/>` | `<rect width="1200" height="1040" fill="#0F1216"/>` |
| body wrap | — | `<g id="cert-body" transform="translate(0 112)">` … `</g>` |

**Translate offset: exactly 112px.** Bottom margin is preserved (footer 900 → 1012
inside 1040, same 28px as before).

**Injection anchor** (verified unique in all 13 files, including `overview.svg`,
which has no stage banner): the subtitle line

```
  <text x="156" y="66" font-size="13.5" fill="#82868F">How a change gets from opened to merged, and what knocks it back.</text>
```

Immediately AFTER that line insert, in order: the `<defs id="cert-defs">` block,
`<g id="field-header">`, `<g id="field-stamp">`, `<g id="field-seal">`,
`<g id="qr">`, then the opening `<g id="cert-body" transform="translate(0 112)">`.
Insert the closing `</g>` immediately before `</svg>`. Assert exactly one match
for every anchor before writing (see `inject()` in `tools/gen_badges.py`).

All defs ids are prefixed `cert-` and collide with nothing in the state SVGs
(they only define `atlasGoldCut` and the `tip*` markers). Logo, title and
subtitle stay where they are; everything from the stage banner down moves.

## 2. Masthead geometry (do not rearrange piecemeal)

- Title block band: x 64–952, y 92–176. Cells: DOCUMENT (static) | PULL REQUEST |
  AUTHOR | APPROVALS.
- QR plaque: x 968–1132, y 8–172 (see §4).
- Gold seal: `transform="translate(700 110) rotate(3.5)"` — over the APPROVALS cell.
- Silver stamp: `transform="translate(872 116) rotate(-5)"` — right of the seal,
  max extent x≈953, clear of the plaque.
- The APPROVALS cell contains dashed placeholder circles + SEAL/STAMP micro-labels
  (`<g id="cert-approvals-placeholders">`). They are fully covered by a visible
  badge, so they never need toggling; when a badge is hidden they read as the
  empty slot awaiting it.

## 3. Substitution keys (state diagrams)

Same conventions as `assets/cards/result-card.svg`: replace text content by `id`,
hide a whole component by setting `display="none"` on its `field-*` group,
overflow is clip-path + a fade-out rect so long strings degrade in place.

| id | kind | content | overflow (visible before fade) |
|---|---|---|---|
| `field-header` | group | whole title block | — |
| `value-pr-number` | text | `#840` | ~7 chars @26px |
| `value-pr-author` | text | author login | ~16 chars @13.5px mono, fades (39-char logins proven in `preview-longnames.png`) |
| `field-stamp` | group | whole silver badge; `display="none"` until /stamp | — |
| `value-stamp-user` | text | stamper login | ~16 chars @12.5px mono on ribbon |
| `value-stamp-sha` | text | 10-char sha | exactly 10 (clip only, no fade) |
| `field-seal` | group | whole gold badge; `display="none"` until /seal | — |
| `value-seal-user` | text | sealer login | ~18 chars @12.5px mono on ribbon |
| `value-seal-sha` | text | 10-char sha | exactly 10 |
| `qr` | group | **regenerated wholesale per PR** (see §4) | — |

Login text is start-anchored (result-card convention); short names sit left of
ribbon center by design.

## 4. QR — regenerated, never string-substituted

The module matrix changes with the URL, so the entire contents of `<g id="qr">`
are re-emitted per PR. Exact producer (same code as `qr_group()` in
`tools/gen_badges.py`):

```python
import segno  # v1.6.6, BSD-3-Clause

def qr_group(url, x=968, y=8, m=4):
    q = segno.make(url, error="m")
    rows = [bytearray(r) for r in q.matrix]          # 33x33 for a version-4 URL
    n = len(rows)
    assert n == 33, f"URL produced version {(n-17)//4} — replan plaque size"
    side = (n + 8) * m                               # 4-module quiet zone each side
    ox, oy = x + 4*m, y + 4*m
    d = []
    for j, row in enumerate(rows):
        i = 0
        while i < n:
            if row[i]:
                run = 1
                while i + run < n and row[i+run]:
                    run += 1
                d.append(f"M{ox+i*m} {oy+j*m}h{run*m}v{m}h-{run*m}z")
                i += run
            else:
                i += 1
    return (f'<g id="qr">'
            f'<rect x="{x}" y="{y}" width="{side}" height="{side}" rx="6" '
            f'fill="#FFFFFF" stroke="#2E353F" stroke-width="1"/>'
            f'<path fill="#0F1216" d="{"".join(d)}"/></g>')
```

Scannability decisions (hard constraints, all verified):

- **Size: 164px plaque on the 1200 canvas = 4px/module → exactly 3.0px/module
  at the 900px embed.** Module edges land on integer device pixels at BOTH
  widths because plaque x/y and the module size are multiples of 4
  (`968*0.75=726`, `8*0.75=6`, `4*0.75=3`). Keep any new plaque origin a
  multiple of 4 — a half-pixel origin measurably breaks decoding at 900px
  (found and fixed on the square certificate: x=518 failed, x=520 decodes).
  When the plaque origin CAN'T be a multiple of 4 (the certificates' corner
  placement), pass `side=` to `qr_group`: the plaque grows to that size and
  the module grid snaps to the nearest multiple of 4 independently of the
  plaque, keeping module edges integer at both widths (quiet zone stays ≥ 4
  modules on every side, asserted).
- **Edge clearance for detectors:** OpenCV's `QRCodeDetector` fails on a
  QR closer than ~38 *device* px to the image border at the 900px render even
  though the modules are perfectly aligned (measured 2026-09-02: 38/31 SVG-px
  margins failed on the landscape 900px PNG; ≥52 SVG px = 39 device px on both
  sides decodes). Keep certificate corner margins ≥ 52px.
- White plaque, near-black modules (`#0F1216`), no rotation, no gradient, no
  shadow; the 4-module quiet zone lives inside the plaque and nothing may
  encroach (in the certificate the QR is painted last so badge shadows can
  never touch it).
- Verified: OpenCV `QRCodeDetector` decodes the URL from every rendered PNG in
  this directory, including all 900px renders (`qr-scan-test.png`).
- A URL longer than ~62 chars at error level M spills into version 5 (37
  modules) and the assert fires: then grow the plaque to `(37+8)*4 = 180px`
  and re-verify the layout (it still fits left of x=1132 if y stays 8, but
  re-run the decode check).

## 5. Merged certificate (`certificate-merged.svg`, 1200x675; square variant 1200x1200)

Landscape 16:9 is the primary (X/LinkedIn link-preview crop); the square is for
feed posts. Additional substitution ids beyond the badge/QR ids above:

| id | content | notes |
|---|---|---|
| `field-cert-author-1..3` / `value-cert-author-1..3` | recipient logins | groups hidden per author count; 32px mono (36px centered on square) |
| `value-cert-pr` | `#840` | tspan (gold) |
| `value-cert-pr-title` | PR title | tspan flowing after the number; whole line clipped at 784px with fade. On the SQUARE variant the line is center-anchored, so keep title ≤ ~60 chars there |
| `value-cert-repo` | `owner/repo` | clipped 270px @12.5 mono (~33 chars) |
| `value-cert-commit` | merge sha (10) | |
| `value-cert-date` | `2026-09-02` | |
| `value-cert-gates` | `11 / 11 CERTIFIED` | green `#12B981`; text is data, not hardcoded |
| `field-cert-merge-count` | whole merge-count credential (group) | `display="none"` when the count can't be fetched (see below); the row keeps its own label so nothing looks gappy |
| `value-cert-merge-count` | `THE 263RD MERGED INTO THIS REPOSITORY BY` (tspan) | ordinal of the opener's total merged PRs *including this one* |
| `value-cert-merge-author` | opener login (tspan, mono) | always slot-1's login: with co-recipients the count belongs to the PR's OPENER, and naming the login in the line is what keeps it unambiguous |

Author slot 1 is the PR's opener by convention; slots 2–3 are co-authors.

Author-count degradation: 1–3 names get equal slots. Past 3, keep slots 1–2 as
the first two authors and put `and N others` in `value-cert-author-3` — do not
shrink the font or add a fourth line.

### Merge-count fetch (populates `field-cert-merge-count`)

```bash
count=$(gh api -X GET search/issues \
  -f q="repo:$OWNER/$REPO type:pr is:merged author:$OPENER" \
  --jq .total_count) || count=""
# Fails with HTTP 422 when the login no longer exists, and the search API
# allows only 30 requests/min (HTTP 403). On ANY failure or empty result:
# set display="none" on <g id="field-cert-merge-count"> and move on — the
# row it sits on keeps its own label, so the layout stays intact.
# On success render "THE ${count}(ST|ND|RD|TH) MERGED INTO THIS REPOSITORY BY"
# into value-cert-merge-count and the opener login into value-cert-merge-author.
```

`certificate-merged-nocount.{svg,png}` shows the hidden-count layout.

### Certificate geometry (2026-09-02 layout)

QR sits in the bottom-right corner of both variants, **equidistant from the
right and bottom edges: uniform 52px margin** (39 device px at the 900px
render — the OpenCV edge-clearance floor, see §4). Plaque is **168px** with
the module grid snapped to multiples of 4 inside it (`qr_group(..., side=168)`):

- landscape: plaque `x=980, y=455` (modules at 1000, 472)
- square: plaque `x=980, y=980` (modules at 1000, 1000)

Badges (landscape): gold seal `translate(952 150) rotate(3.5) scale(1.25)`,
silver stamp `translate(1078 352) rotate(-5) scale(1.15)` — stacked in the
right column above the QR; every pairwise bbox gap (stamp/seal/QR, drop
shadows included at full radius) is positive (~8.3px stamp↔seal, ~8.6px
stamp↔QR). Badges (square): seal `translate(420 810) … scale(1.3)`, stamp
`translate(780 812) … scale(1.25)` — a centered pair, since the QR no longer
sits between them. The QR group is painted LAST in document order in both
variants; the SCAN caption stays beside it (landscape, left of the plaque)
or above it (square).

## 6. Caveats

- Rendering uses the font stack's fallback (DejaVu on the CI boxes — Inter is
  declared first but not embedded). Metrics differ a little between the two;
  every dynamic field is clipped+faded, so nothing can escape its box either way.
- The per-stage hint text in the stage banner (right-aligned at x=1120) moves
  down with the body and stays fully visible — nothing overlaps it.
- The badges intentionally overhang the title-block band (and cover the
  APPROVALS label when present); that is the applied-stamp look, not a bug.
- `<title>`/`aria-label` of the diagrams are left untouched by injection.
- Curved badge lettering is per-glyph rotated text, NOT `textPath` — librsvg
  (which renders these to PNG) does not implement `textPath`. Keep it that way.
