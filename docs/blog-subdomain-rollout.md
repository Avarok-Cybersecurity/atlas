# blog.atlasinference.io — rollout record

Append-only. One entry per wave: what was found, the evidence, what changed, and
what the negative control proved. Newest entries at the bottom.

## The gate

`/workspace/atlas-blog/.gate.sh` (untracked; mirrors what `.github/workflows/site.yml`
runs in CI) — for each of `site/` and `blog/`:

1. `bun test src/lib` — unit tests
2. `bun x --bun vite build` — the SvelteKit build, with `ATLAS_RECIPES_ROOT`
   pointed at a local `atlas-recipes` checkout
3. the chevron-field contrast check (`.contrast-check.mjs`), which re-derives the
   field's luminance budget against the ground it is actually painted on

Nothing is committed unless the gate is green.

## The baseline (before any change on this branch)

Branch `feat/blog-subdomain`, cut from `origin/main` @ `29591e0a2`.

| leg | result |
|---|---|
| `site` unit tests | **483 pass / 0 fail** across 37 files |
| `site` build | **ok**, 9.25 s, adapter-static wrote `build/` |
| `blog` | absent — not yet created |
| contrast check | not yet written |

Green. Any red after this point belongs to this work until proven otherwise.

## Wave 1 — reconnaissance and the two things the brief got wrong about the repo

**Found.** Two premises needed correcting before any code was written.

1. *The working tree was stale.* `/workspace/atlas` sits on
   `fix/ssm-rollback-hardening` and its `site/` is the old **light** "warm
   workshop" system (`--bg: #f4f0e8`, copper `#b5622f`). `origin/main` has since
   moved to the **deep violet workstation** system (`--bg: #14111f`, accent
   `#9271f4`) and already names the four chevron colours as first-class tokens —
   `--ch-violet #BE9DF8`, `--ch-cyan #49C3DB`, `--ch-green #12B981`,
   `--ch-gold #EFB338`. Those are *byte-identical* to the four chevron constants
   in the supplied scaffold. All work happens in a clean worktree at
   `/workspace/atlas-blog` cut from `origin/main`, never in `/workspace/atlas`.

2. *Consequently the brief's ambiguity dissolves.* "Use the look and feel of the
   main website" and "the same color scheme" are already 90% satisfied by the
   scaffold; the only real divergence is the ground (`#0F1216` scaffold vs
   `#14111f` main). The blog adopts **main's** tokens; both properties gain the
   WebGL field.

**Evidence.** `git show origin/main:site/src/app.css`, lines 9–60.

**Consequence for the field.** `FIELD-NOTES.md` derives the field's amplitude
from a contrast budget solved against `#0F1216`. `#14111f` is a *different*
ground (violet-tinted, and slightly lighter), so those measured ratios —
13.41:1 / 10.35:1 / 4.56:1 — **do not transfer**. They have to be re-derived, and
the tightest of them (metadata gray) is only 0.06 above AA on the original
ground, so this is not a rounding concern. That re-derivation is the
`.contrast-check.mjs` gate leg.

## Wave 2 — the origin vhost

**Changed.** `blog/deploy/nginx/blog.atlasinference.io.conf` is the SSOT for the
vhost; it is installed on the avarok origin as
`/etc/nginx/sites-available/00-blog.atlasinference.io.conf` and symlinked into
`sites-enabled`. Docroot `/var/www/blog.atlasinference.io/html`, owned
`ubuntu:ubuntu` — the same user the marketing-site deploy already rsyncs as
(established below), so the blog needs no new SSH identity.

**How the deploy identity was established** — the values live in GitHub
environment secrets and cannot be read back, so they were derived from the
origin. `sshd`'s journal shows `Accepted publickey for ubuntu from 52.162.9.240`
(Azure) at `Aug 29 00:56:38`, and `/var/www/atlasinference.io/build` has mtime
`Aug 29 00:56` and owner `ubuntu`. So `DEPLOY_SSH_USER=ubuntu` and
`DEPLOY_PATH=/var/www/atlasinference.io/build`.

**Proved** — origin-direct (`--resolve … 127.0.0.1`) and again through
Cloudflare on the public name:

| check | result |
|---|---|
| `GET /` through Cloudflare | **200**, serves the docroot |
| `/` cache-control | `public, max-age=300` |
| `/_app/immutable/probe.js` | `public, max-age=31536000, immutable` |
| security headers on **HTML** | `nosniff` + `SAMEORIGIN` + `strict-origin-when-cross-origin` all present |
| `/404` (extensionless clean URL) | 200 via `$uri.html` |
| `/.env` | 404 |
| `http://…/x` | 301 → `https://blog.atlasinference.io/x` |
| `nginx -t` | ok |

**The negative control, and what it caught.** The claim being tested is that
computing Cache-Control through a `map` — instead of in a `location` — is what
keeps the security headers on HTML. The control is the docs vhost, which still
has the location-level form:

```
$ curl -D- https://docs.atlasinference.io/index.html
cache-control: public, max-age=300
# …and nothing else. No x-frame-options, no nosniff, no referrer-policy.
```

The control is **red**, as required: the defect reproduces on a live vhost, so
the check is measuring something real. The blog vhost, same request shape,
returns all four headers.

> **Open, and deliberately not acted on:** `docs.atlasinference.io` is serving
> every HTML document with no `X-Content-Type-Options`, `X-Frame-Options` or
> `Referrer-Policy`. The fix is the same `map` used here. It is a different
> property from the one this work was authorised for, so it is reported rather
> than applied.

**One thing that looked wrong and was not.** The first `curl` after
`systemctl reload nginx` came back with a 3724-byte body, `last-modified
Aug 6`, and an HSTS header this vhost never sets — i.e. some *other* server
block answered. Cause: `reload` is asynchronous, and an old worker served that
connection before the new configuration was live. Re-running after the reload
settled gives the table above every time. Recorded because "a header set I did
not write" is exactly the shape of a real misconfiguration, and it would have
been easy to go debugging server_name matching for an hour.

## Wave 3 — the blog application

**Changed.** `blog/` is a SvelteKit app built the same way `site/` is
(adapter-static, bun, Vite 8, Svelte 5 runes), prerendered to static files.

**Design system: one, not two.** The `:root` token block moved out of
`site/src/app.css` into `web-shared/atlas-tokens.css`, which both apps now
import. This is the SSOT the brief implies when it says "the same colour
scheme" — with two copies, "the same" survives exactly until the first edit.
`blog/src/app.css` defines only editorial structure (reading column, chevron
rail, TOC, code blocks, footnotes) and aliases every colour it needs onto a
shared token or a `color-mix` of one. It introduces exactly one value of its
own, `--bg-sunk`, because the marketing site has no recessed surface to borrow.

*Control for the extraction:* the emitted stylesheet was diffed before and
after. Same length to the byte, and the only difference is that `:root` now
precedes the `*` reset instead of following it — disjoint selectors, disjoint
properties, no cascade consequence. A pure refactor, proved rather than
asserted.

**Renderer: raw WebGL2, as instructed.** `blog/src/lib/gl/` is the supplied
runtime, not the three.js variant. Two changes to it:

- The five colours lost their defaults. They were hardcoded `#0F1216` /
  `#BE9DF8` / … in `DEFAULTS`, which is a second source of truth for values
  that live in the token file — and the canvas paints the **ground itself**, so
  a drifted value shows up as a visible seam between canvas and page with
  nothing failing. The component now reads them off the cascade with
  `getComputedStyle` and the runtime refuses to build without them.
- `density` dropped from 1.0 to 0.45. See below; this is the wave's real finding.

**Posts are Svelte components, not markdown.** These posts carry measured
tables, annotated code and callouts; markdown reaches that only through mdsvex
plus a parser plus a highlighter, which would outweigh the entire WebGL
background by more than an order of magnitude. Front matter is a `meta` export
from `<script module>`. Dropping a file into `blog/src/lib/posts/` puts it on
the index, its tag page, its author page, the RSS feed, the sitemap and the
prev/next chain with nothing else to register.

`postindex.js` holds the rules and `posts.js` holds the `import.meta.glob` —
split on the SBIO line, so the rules can be tested without a bundler.

### The contrast finding

`FIELD-NOTES.md` derives the field's amplitude from a budget solved against
ground `#0F1216`, by sampling 14 frames and reporting the worst seen. Neither
half of that transfers here. The ground is `#14111f`, and sampling cannot see
the case where all three depth layers land on one pixel at the sweep's peak —
rare, but exactly the case that puts metadata gray under AA.

`.contrast-check.mjs` therefore computes the **analytic bound**: the most
luminance the shader can add to any pixel, `DIM_SUM (2.28) × AMT_MAX (0.05) ×
density`, per hue, normalised the way the shader normalises. No sampling, so no
case to miss. It reads the ground and the text tokens from the token file and
the density from the runtime, and it asserts four exact lines of the shader
still exist — otherwise the gate would keep passing against a field it no
longer describes.

| density | tightest ratio (`--t3 #8a83af`) | |
|---|---|---|
| 1.00 (as supplied) | 3.75 | below AA |
| 0.60 | 4.35 | below AA |
| **0.5109** | **4.50** | AA exactly |
| **0.45 (shipped)** | **4.60** | AA with margin |

*Control:* the gate was run at 0.6 and watched go **red** at 4.35:1 before 0.45
was chosen. It is not a check that cannot fail.

### Tests, and the three controls that prove they work

`blog/src/lib/postindex.test.js` — 18 tests, 26 assertions. Not coverage
theatre: each rejection test names the silent failure it prevents (an
undeclared tag renders an uncoloured dot and drops the post out of its category
page; an unparseable date sorts it to the top of the index forever). One test
exists purely to prove the validations are not blanket-rejecting.

Three negative controls, each a real defect reintroduced:

| defect reintroduced | result |
|---|---|
| tag validation removed | 17 pass / **1 fail** |
| sort reversed to oldest-first | 15 pass / **3 fail** |
| `findIndex` `-1` guard removed from `neighboursOf` | 17 pass / **1 fail** |
| restored | **18 pass / 0 fail** |

`blog/e2e/check-headers.mjs` is the live check the vhost comment promises: four
headers across three response classes nginx routes differently. Green on the
blog (18/18); pointed at `docs.atlasinference.io`, which still has the
location-level form, it goes **red on exactly the six security-header
assertions**. That is the control, and it runs against a real server.

**Deployed.** `blog.atlasinference.io` now serves the built site through
Cloudflare — rsynced by hand this once, with the same flags the workflow will
use, so the header check had something true to test.

## Wave 4 — deploy on merge, inside the existing job

**Changed.** `.github/workflows/site.yml` now builds and deploys both
properties. Not a second workflow: the brief asked for the same job, and one
job is also the correct shape — it means a single SSH agent, a single host-key
pin, and no way for the two properties to deploy from different commits.

| where | added |
|---|---|
| `on.push.paths` | `blog/**`, and **`web-shared/**`** |
| `unit` job | blog unit tests, and the contrast budget |
| `build` job | blog install → build → per-route title check → artifact |
| `deploy` job | `rsync` to `DEPLOY_BLOG_PATH`, then the live header check |

**"If a diff exists" is already how rsync behaves.** `rsync -az --delete-delay`
transfers nothing and deletes nothing when the built tree matches the deployed
one, so a push that only touched `site/` costs one no-op sync rather than a
redeploy. No content hashing of our own was needed.

**`web-shared/**` in the path filter is the non-obvious one.** The tokens file
is imported by both apps. Without that line, editing a colour would restyle
neither property until something unrelated happened to touch `site/`, and the
two would sit at different versions of "the same colour scheme" in the
meantime — which is precisely the failure the shared file exists to prevent.

**`DEPLOY_BLOG_PATH`** was added to the `production-site` environment
(`/var/www/blog.atlasinference.io/html`). The blog deploy step **fails hard** if
it is absent, rather than taking the soft skip the job takes when
`DEPLOY_SSH_PRIVATE_KEY` is missing. Those are different situations: the soft
skip is for an environment with no deploy configured at all, this would be an
environment that deploys with one target forgotten — and skipping quietly there
means the blog stops updating while every run stays green.

**Post-deploy verification runs against the live origin**, not against the
artifact. `blog/e2e/check-headers.mjs` is the only instrument that observes the
`add_header` defect, and it also catches a deploy that landed the wrong tree —
the 404 assertion fails if `404.html` is not in the docroot.

## Wave 5 — the reference is the palette, and the field was invisible

**Correction from the user, and it was right.** The reference in
`/workspace/etc/site-blog` is authoritative for both the palette and the
artwork. Two things were wrong:

1. The blog was on `#14111f` — the marketing site's ramp — not the reference's
   `#0F1216`.
2. The blog's header and footer used `favicon.svg` plus the word "Atlas" set in
   the UI font. The reference uses the **real lockup**: the mark, the wordmark
   outlines including the Avarok signature "A" with its arrow shaft, and the
   tagline. Confirmed by the user: *"the Atlas 'A' does not have an arrow on the
   current blog, yet the inputted reference does use it."*

**Decided with the user:** both properties move onto the reference ramp, so
there is one palette rather than two.

### The palette move

`web-shared/atlas-tokens.css` now holds the reference ramp under the marketing
site's token names, so nothing downstream had to be renamed. The work that was
not a hex swap:

| what | count | why it mattered |
|---|---|---|
| `rgba(124, 92, 255, x)` → `color-mix(… var(--accent) …)` | 13 | hand-written tints of the *old* violet; they would have been the only violet-tinted things left on the page |
| `rgba(251, 191, 36, x)` → `--amber` | 17 | same, for the gold |
| `rgba(58, 48, 84, x)` → `--border-strong` | 6 | card hairlines |
| `#0d0a16` → `--sunk`, `#ddd8f0`/`#e6e2f5` → `--t2`/`--t1`, `#a78bfa` → `--accent`, `#7ba7d4` → `--ch-cyan`, `#f87171` → `--red`, two gradients | 15 | literals duplicating a token's value |

Verified by grep: **zero** old-palette literals survive in either built
stylesheet. What remains hardcoded is legitimately not brand — Discord's
`#5865f2`, and the three macOS traffic-light colours in the terminal mock.

**One bug found on the way.** `.btn-primary:hover` and `.nav-star-btn:hover`
set `background: var(--accent-deep)` while the rule above them sets
`color: #fff`. `--accent-deep` is the *light* violet — white on it was already
about 2:1 before this work, and the reference ramp makes it worse. The token
file already carries `--accent-fill-hover`, documented as *"deepens on hover,
so white gains contrast rather than losing it"*, and the hovers now use it.

### The artwork

`web-shared/components/AtlasLockup.svelte` carries the reference `<defs>`
verbatim — mark, wordmark, tagline — with one substitution: the literal brand
greys and chevron hues became the tokens holding those same values, so the
lockup follows the palette instead of pinning a second copy of it. Both
properties render `kind="defs"` once per document and `<use>` it from the nav
(`horizontal`) and the footer (`full`). Sizing is by width, per the guidelines'
minimums, with clear space as CSS margin rather than viewBox padding.

### The finding: the contrast bound had deleted the background

The field was live — `cf-on` was on the canvas, WebGL2 was up under SwiftShader
— and invisible. Measured, in a text-free gutter against ground `#0F1216`:

```
brightest gutter pixel (21, 24, 28)     # ground (15, 18, 22)
                                        # +6/+6/+6, and neutral, not tinted
```

Six of 255 on every channel equally: at that amplitude the chevron hues round
to grey in 8-bit. The bound was arithmetically correct and the result was
useless — the amplitude of the whole field was being set by its rarest
accident, three depth layers landing on one pixel (2.28 layers of luma).

**The fix is in the shader, not the density.** Two lines clamp accumulated luma
to one layer's worth. Each hue is already unit-luma, so it is a uniform scale on
the colour vector: hue is preserved exactly and only overlapping pixels are
touched.

```glsl
float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
col /= max(1.0, lum);
```

The bound drops from 2.28 to 1.00, which buys back 2.28× the amplitude for the
same guarantee.

| | AA boundary density | shipped | tightest ratio | brightest gutter pixel |
|---|---|---|---|---|
| unclamped | 0.4438 | 0.38 | 4.60 | `(21,24,28)` — neutral |
| **clamped** | **1.0119** | **0.85** | **4.61** | **`(23,25,33)` — violet** |

The gate asserts the clamp line still exists in the shader, so removing it
cannot leave the check describing a field that no longer exists.

### Two defects the screenshots caught

- **`/posts/<slug>.html` 404'd.** adapter-static writes that file and nginx
  serves both it and the extensionless URL, but on the `.html` one the client
  router hands `load` a slug of `"foo.html"`, which matches no post: the page
  server-rendered correctly and then 404'd on hydration. `cleanSlug` strips it
  in all three dynamic routes and in the canonical. Two tests, and the control
  (reverting `cleanSlug` to the identity) takes 20 pass → **18 pass / 2 fail**.
- **No `+error.svelte`.** SvelteKit's default rendered unstyled, flush to the
  viewport edge, inside the site's chrome. There is now a shared `NotFound`
  component behind both the prerendered `404.html` and the runtime error page.
- **The nav's current-page underline floated** three pixels above the header
  hairline, because it was offset from a shrink-wrapped link by a hand-measured
  number. The links are full bar height now and the underline is pinned to
  `bottom: 0`, so it cannot drift when the bar height or the font size moves.

## Wave 6 — the chart palette, and two things that should not have been in the PR

**The chart series palette was pinned, not changed.** `site/src/lib/gates.js`
carries three hand-derived series colours whose comment asks, in as many words,
for re-derivation if the palette is ever revisited. It was, so they were
re-measured. They did not need to move — the ground got *darker*, so every ratio
rose and the ≥3:1 floor gained margin:

| series | on `--bg` | on `--card` | before |
|---|---|---|---|
| copper `#ee6f2f` | 6.21 | 5.53 | 6.15 / 5.51 |
| steel `#2f88ee` | 5.25 | 4.68 | 5.20 / 4.66 |
| teal `#51cdb0` | 9.58 | 8.52 | 9.48 / 8.49 |
| fallback `#6f6a8d` | 3.69 | 3.28 | — |

The pairwise CIEDE2000 separations the palette was optimised for do not depend
on the background at all, so they carry over unchanged.

A comment is not a guard, so `site/src/lib/series-contrast.test.js` now measures
this on every run, reading the surfaces from the token file rather than
retyping them — the whole point being that the two cannot drift. The palette
moved to `series-colors.js` to make it importable: `gates.js` imports
`$lib/gates.generated.json`, and `$lib` is a Vite alias that does not exist
under `bun test`, so nothing importing `gates.js` was testable at all.

Two controls, both fired:

| defect | result |
|---|---|
| a near-ground grey `#1a1d22` put in the series | 8 pass / **2 fail** |
| `--card` lightened in the token file | 6 pass / **4 fail** |
| restored | **493 pass / 0 fail** |

**Feed content types.** `/rss.xml` and `/sitemap.xml` were being served as
`text/xml` — nginx typing them by extension. They now carry
`application/rss+xml` and `application/xml`, via `location =` blocks that set
*only* a type. That is precisely the shape that reintroduces the `add_header`
defect if anyone adds a Cache-Control line to one, so the live check was
extended to assert all four headers on both, and it is green at 28/28.

**Two things were in the PR that should not have been.**

- **145 MB of bun cache.** `bun install` refused the system temp directory
  (`AccessDenied`), so `HOME` and `TMPDIR` were pointed at `.tmp/` inside the
  worktree — and `git add -A` swept 2,768 cache files into all four commits.
  Caught by counting lines per file in the branch diff before opening the PR.
  Stripped with `git filter-branch --index-filter` across the range; the branch
  diff went from 2,837 files to 69. `.tmp/` is in `.git/info/exclude` now.
- **`site/src/lib/gates.generated.json`**, regenerated by the build on every
  run and swept up the same way. Restored to `origin/main`'s copy, so the PR
  diff shows it unchanged.

The lesson is narrow and worth keeping: `git add -A` in a worktree that also
holds a build cache commits the cache, and nothing about the commit output says
so — the file count only shows up if you go looking for it.

## Wave 7 — phone widths, and a link with no name

**Found by screenshotting at 390px**, which is the only way either of these
shows up.

**The categories disappeared below 900px.** The scaffold hides the nav there and
offers nothing in its place. On the index that is survivable — the chip row
carries the same links — but on an *article* a phone visitor had no route to any
category at all. They now wrap to a second, horizontally scrolling row inside
the same sticky bar: one wrapped flex line, no JavaScript, every link the
desktop has. `scroll-padding-top` doubles under that breakpoint, because the bar
is two rows tall there and an anchor jump has to clear both.

**A link with no accessible name.** `@media (max-width: 460px)` sets
`.btn-ghost .lbl { display: none }`, which removes the label from the
accessibility tree as well as from the page — so under 460px the
`atlasinference.io` button was an anchor containing one `aria-hidden` arrow and
nothing else. Screen readers get "link"; Lighthouse's accessibility gate, which
this repo pins at **minScore 1**, fails it outright. Fixed with an explicit
`aria-label`, which survives the label being hidden. The GitHub button next to
it already had one, which is why only one of the two was broken — the kind of
asymmetry that a visual check never surfaces.

The marketing site needed nothing here: its hamburger drawer already carries
every link, and the lockup's negative margin puts the mark on the same optical
left edge as the hero text at 390px.

## Wave 8 — the accessibility audit, and a false positive of my own

**Lighthouse could not be run locally.** `headless_shell` under SwiftShader is
the only browser on this box, and `lhci` dies in it with *"Waiting for DevTools
protocol response has exceeded the allotted time (Method:
Network.setUserAgentOverride)"*. **There is no local Lighthouse score for this
change** — CI's real Chrome has to produce it. That matters because
`site/lighthouse/lighthouserc.json` asserts `minScore: 1` on performance,
accessibility, best-practices and SEO for `index.html`, which is the tightest
gate in the repo and the one most likely to reject this work.

So the categories most at risk were checked directly instead, with a DOM-walking
auditor (`blog/e2e/audit-a11y.js`) run against every route of both properties.

| page | contrast | names | heading order |
|---|---|---|---|
| marketing index | pass | pass | pass |
| blog index | pass | pass | pass |
| both posts | pass | pass | **h2 → h4** |
| both tag pages | pass | pass | **h1 → h3** |
| author page | pass | pass | **h1 → h3** |
| blog 404 | pass | pass | **h1 → h4** |

**Contrast passes everywhere**, which is the load-bearing result: the palette
move did not cost a single text pair its AA margin.

**The heading skips were real.** Entry titles were `h3` and footer column
headings were `h4`. On a tag page the entry list follows the band's `h1` with
nothing between it, and on the 404 page the footer is the only heading after the
`h1` — so both skipped levels. That is a navigation defect for anyone moving by
heading, and a Lighthouse `heading-order` failure. Both are `h2` now, styled
down rather than marked down. All seven blog routes re-audit clean.

**One finding was mine, not the site's.** The first run reported the NVIDIA
Inception link in the marketing footer as having no accessible name. It does
have one — from the `alt` on the image it wraps, which my auditor was not
reading. Fixed the auditor; the link was always correct. Recording it because a
tool that invents findings costs more than one that misses them, and the only
reason it was caught is that the "defect" looked implausible enough to check.

**The auditor was then proved able to fail**, which is the point of having it.
With four defects planted in a real page — body copy at 1.77:1, metadata at
3.78:1, an anchor wrapping only an `aria-hidden` svg, and an `h5` after an
`h2` — it reports all four.

**Open:** the blog is not covered by `lighthouse.yml`. Extending it there is the
right end state, but the budget has to come from a first observed score rather
than be declared in the same change that first measures one, so it is a
follow-up rather than part of this PR.

## Wave 9 — CI confirms the new steps ran, and the nav had the same bug the chips did not

**The CI steps executed, which is not the same as the job being green.** Pulled
the log rather than trusting the tick:

| step | evidence |
|---|---|
| Install blog dependencies | `52 packages installed` |
| Run blog unit tests | 20 named tests, each printed |
| Chevron-field contrast budget | `ground #0F1216 density 0.85 … PASS: tightest is 4.61:1` |
| Build blog | `✔ done`, `Wrote site to "build"` |
| Every blog route kept its own title | five files checked, real titles printed |
| Upload blog artifact | `blog-site` uploaded |

`Site unit tests`, `Build SvelteKit site`, `≤500 LoC`, `SPDX`, `cargo deny`,
`GDN PINS` and the merge-ancestry guards all pass. The deploy job correctly
reports **skipping** — it is push-only. **Lighthouse is still queued**, and it
remains the one gate that could reject this.

**The nav had the `.html` bug the chip row did not.** `current()` compared
`page.url.pathname` to the href directly, so on `/tags/engineering.html` the
chip lit and the nav entry did not — the chip's state comes from the load
function, which strips the extension, and the nav's did not. Same defect class
as wave 5's, in the one place that had its own copy of the comparison.

The rule moved to `navCurrent(pathname, href)` in `content.js` so it could be
tested at all, and it now also treats a trailing slash as the same page. Eight
assertions; two controls, both fired:

| defect reintroduced | result |
|---|---|
| raw-pathname comparison (the original) | 20 pass / **1 fail** |
| "Latest" matches every path | 20 pass / **1 fail** |
| restored | **21 pass / 0 fail** |

**Tag, empty-tag and author pages reviewed at desktop width.** The empty state
(`Releases`, no posts yet) reads correctly rather than as a broken page, and the
category chevron takes its tag's colour. No changes needed.

## Wave 10 — Lighthouse failed, and it was right to

**`Lighthouse audit` → fail. `Runtime error: Lighthouse was unable to reliably
load the URL you requested because the page stopped responding.`** Every audit
in the run reports `Caught exception: PAGE_HUNG`. This is the gate flagged in
wave 8 as the one that could reject the work, and it did.

**It is not a CI quirk.** The cause is the chevron field, and the defect is real
for anyone browsing without hardware acceleration — a GPU on the browser's
blocklist, a VM, a locked-down corporate image. A fullscreen SDF fragment
shader is nearly free on any GPU and ruinous on none.

**Measured, on two cores at Lighthouse's desktop viewport** (`taskset -c 0,1`,
which is what a GitHub-hosted runner has):

| | wall time |
|---|---|
| field enabled | **11.2 s / 11.1 s** |
| field disabled (factory returns null) | 5.9 s / 7.1 s |
| **after the fix** | **6.6 s** |

**Two instruments were wrong before one was right**, which is worth recording
because the first two looked plausible:

1. An in-page rAF counter reported `measuring` forever and never finished —
   under `--virtual-time-budget`, virtual time does not advance while a rAF
   loop keeps requesting frames, so the probe could not terminate *whatever*
   the page did. It measured nothing.
2. Forcing `prefers-reduced-motion` (one frozen frame, no loop) *also* hung the
   same probe — which briefly looked like evidence that the loop was innocent.
   It was the same instrument defect, not a finding.

Only the wall-clock A/B against a build with the factory stubbed out
discriminated, and only once the core count was constrained to match the
runner: at full core count the difference was 4.2 s vs 3.1 s, easy to dismiss.

**The fix.** The runtime reads `WEBGL_debug_renderer_info` and refuses to start
on a CPU rasteriser (`swiftshader`, `llvmpipe`, `softpipe`, `software`,
`basic render`, `microsoft basic`), releasing the context it took rather than
leaking it toward the browser's 16-context cap. The CSS dot field — already the
design's fallback for no-WebGL — stays visible. Verified: the canvas now carries
`cf` without `cf-on`, and `cf-dots` without `cf-muted`, on both properties.

Two other changes came with it:

- **The draw rate is capped at 30fps.** The field drifts at 0.02 units/second;
  thirty frames of that is indistinguishable from sixty and costs half the fill.
  rAF still runs at the display rate, and the skip happens before the draw.
  *This is reviewed, not measured* — see below.
- **A context leak was fixed.** When `build()` failed the runtime returned
  `null` without releasing the GL context, so a shader-compile failure leaked
  one per mount.

**A watchdog was written and removed.** It would have frozen the field on a
renderer that reports hardware but cannot keep up. It could not be verified in
any instrument on this box: under virtual time `performance.now()` advances in
jumps so a wall-clock threshold never trips, and without virtual time the
headless browser screenshots before the loop has run. Its failure mode is
disabling the background on a machine that was merely busy for a moment. An
unverifiable guard with a user-visible false-positive is worse than no guard,
and the measured case is already covered. The reasoning is in the source where
the code would have been.

**What is verified and what is not**, stated plainly:

| | status |
|---|---|
| software-renderer bail | **measured** — 11.2 s → 6.6 s, canvas inert, dots visible |
| context leak on build failure | reviewed; the failure path is not reachable on demand here |
| 30fps cap | **reviewed, not measured** — every instrument available defeats it. Its worst case is the previous 60fps behaviour |
| Lighthouse passing | **unknown** — CI's real Chrome is the only instrument, and it has not re-run yet |

**Consequence for the screenshots in earlier waves:** they were captured under
SwiftShader, which is now exactly the configuration where the field does not
run. They remain valid evidence of what a GPU visitor sees, but they can no
longer be reproduced without defeating the check — the procedure for doing that
is documented beside it in the source.

## Wave 11 — two kilobytes hiding in a string literal

**Checking the post's own claim caught a real payload defect.** The dek said
"the animated field behind this page is 2.5 KB" — a figure taken from
`FIELD-NOTES.md`, i.e. a measurement of somebody else's build. Measured on what
this repo actually ships:

| | minified | gzip | brotli |
|---|---|---|---|
| as written | 11,192 | 4.91 KB | **4.25 KB** |
| comments stripped at build | 6,393 | 2.79 KB | **2.52 KB** |

The shader is imported with `?raw`, so it ships inside a JavaScript string
literal — and a minifier cannot touch string contents. Every byte of the
commentary explaining the SDF geometry, the falloff and the luma clamp was going
to every visitor. The shader source is 7,254 bytes of which 2,481 are code:
**comments were 66% of the file and 43% of the shipped component**, and almost
exactly the gap between the claimed number and the real one.

`web-shared/glsl-strip.js` is a Vite plugin, used by both apps, that strips
comments from `.glsl?raw` at build time. It refuses to emit a shader that lost
its `#version` directive or its `main()` — a shader that fails to compile does
not throw, it silently drops to the CSS dot field, which is the exact failure
mode this component exists to avoid.

With it, the field measures **2.52 KB brotli** — which is the field notes'
figure to two decimal places. Their number was right; it was measured on a build
that stripped shader comments, and this one now does too.

**Proved, not assumed:**

| check | result |
|---|---|
| comments absent from the shipped bundle | grep: 0 occurrences |
| stripped shader still compiles and links | canvas reaches `cf-on` with the renderer bail bypassed — a shader that failed to compile could not |
| `stripGlsl` unit tests | 6 tests, 27 total across the blog |
| control: also strip newlines (welds `#version` to the next line) | 2 pass / **4 fail** |
| control: make the strip a no-op | 2 pass / **4 fail** |

**The post was corrected**, not just the code: the three.js comparison table is
now attributed to the pre-existing comparison rather than presented as this
page's measurement, a new section documents the string-literal finding with the
numbers above, and the dek quotes 2.52 KB. A post whose thesis is measured
honesty cannot cite someone else's measurement as its own.
