# Avarok website branding

The brand is Avarok. It was Atlas until September 2026. The mark, the palette,
the clear space and the minimum sizes did not change. The wordmark did.

The full guidelines are `assets/brand/BRAND-GUIDELINES.md`. This file says how
the website applies them.

## The lockup

One component draws every lockup on the site, the blog and the product mockup:
`web-shared/components/AtlasLockup.svelte`. The file keeps its old name so the
imports across two apps did not have to move in the same change. It renders
Avarok.

- It is inline SVG whose two greys are `--logo-word` and `--logo-tagline`, so
  the theme swap costs no second file and no second request.
- **Header**: the horizontal lockup, 152 px wide, 132 px below 520 px.
- **Footer**: the full lockup, "Avarok" over "Inference Engine", 244 px.
- **Below 220 px of room**: the horizontal lockup, as the guidelines prescribe.
  The tagline stops being legible under that width.
- The header and footer supply the clear space themselves, so the lockup's own
  margin is set to zero there. Anywhere else, leave the margin alone: it is one
  chevron gap, which is the rule.

The arrow A is the brand kit's hand corrected path, unchanged. The letters
"varok" were set in the kit's wordmark typeface and fitted to the approved
artwork. `assets/brand/src/wordmark-paths.json` holds the outlines and the
measurements, and the masters in `assets/brand/` are generated from it.
`src/lib/lockup-artwork.test.js` fails if the component and the masters drift.

Never generate the logo with an image model, never redraw it, never recolour
it. `media-brief/README.md` repeats this for anyone making imagery.

## Colour

`web-shared/avarok-tokens.css` is the single source, for this site and the
blog. Four chevron hues: lavender `#BE9DF8`, cyan `#49C3DB`, green `#12B981`,
gold `#EFB338`. Two grounds: `#0F1216` dark, `#FFFFFF` light. `data-theme` is
set before first paint from `localStorage` (`avarok-theme`) or the system
preference. The contrast gate (`.contrast-check.mjs`) runs in CI.

On the marketing pages each of the three promises has a hue, used consistently:
speed is lavender, security is cyan, governance is green. Gold is the fourth
accent, for everything that is none of the three.

## Type

IBM Plex Sans for text, IBM Plex Mono for labels, numbers and code. Both are
self hosted from `static/fonts/` with their licences beside them, because the
site makes no third party requests. They were chosen because the brand kit's
slide and letterhead templates use them.

## Imagery

Product footage is recorded from the product mockup and always carries its
"Demo data" chip. Ambient footage is procedural, drawn from the palette by
`src/lib/broll/scenes.js`. Generated imagery follows `media-brief/`. Logos of
other companies appear only on the prior roles wall, under the note that says
they are not customers, and `static/logos/README.md` records each source.

The social card, `static/og-image.png`, is rendered by `scripts/media/og.mjs`
from the vector master and the front page headline.
