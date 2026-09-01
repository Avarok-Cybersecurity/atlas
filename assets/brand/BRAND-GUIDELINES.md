# Atlas — brand guidelines

The same rules as `Atlas-Brand-Guidelines.pdf`, in text, for pasting into a wiki
or a contributor doc.

## Which lockup

- **`logo-full`** — the default, wherever the tagline has room to be read: 220 px
  wide and up on screen, 45 mm and up in print.
- **`logo-horizontal`** — navigation bars, tight headers, anywhere narrower than
  220 px. Minimum 120 px wide.
- **`wordmark`** — pages where the mark already appears elsewhere.
- **`mark`** — every square context: avatars, favicons, app icons. Minimum 16 px;
  below 48 px use `mark-compact`.

Pick the variant that matches the background: the plain files for light grounds,
the `-ondark` files for dark. Don't recolor one into the other — the grays
differ deliberately.

## Clear space

One `x` on every side, where `x` is the gap between two chevrons. It's a feature
of the mark itself, so it can be measured off the artwork at any size — there's
no ratio to remember. Nothing else enters that space: no type, no rules, no
other logos, no edge of the page.

## Minimum sizes

| Asset | Screen | Print |
| --- | --- | --- |
| `logo-full` | 220 px wide | 45 mm |
| `logo-horizontal` | 120 px wide | 25 mm |
| `mark` | 16 px | 5 mm |

220 px puts the tagline's x-height at 8 px, which is the floor for comfortable
reading on screen. These are legibility limits, not preferences.

## Color

| Role | Hex | Contrast |
| --- | --- | --- |
| Chevron 1 | `#BE9DF8` | 2.24:1 on white, 8.38:1 on `#0F1216` |
| Chevron 2 | `#49C3DB` | 2.08:1 / 9.04:1 |
| Chevron 3, upper | `#12B981` | 2.54:1 / 7.41:1 |
| Chevron 3, lower | `#EFB338` | 1.88:1 / 9.99:1 |
| Wordmark on light | `#9397A0` | 2.93:1 on white |
| Tagline on light | `#BDC0C5` | 1.82:1 on white |
| Wordmark on dark | `#C9CCD4` | 11.69:1 on `#0F1216` |
| Tagline on dark | `#82868F` | 5.15:1 on `#0F1216` |
| UI gray | `#73767D` | 4.55:1 on white |

**The light-ground grays are logo colors, not text colors.** At 2.93:1 and
1.82:1 they sit below the 3:1 bar for graphical objects and nowhere near the
4.5:1 needed for body copy. A logo is exempt from those rules; a paragraph is
not. Use `#73767D` for gray type on light backgrounds.

The palette reads much better on the dark ground — every value clears 5:1 there.
If a layout can go either way, dark is the stronger choice for this mark.

## Backgrounds

- Light ground: `#FFFFFF`. Off-whites up to about `#F7F7F9` are fine.
- Dark ground: `#0F1216`. This is the only dark background — don't substitute
  pure black or a brand-tinted dark.
- Mid-tones are the failure case. The tagline gray disappears against anything
  in the `#888`–`#BBB` range; put the logo on a solid light or dark panel.
- Photographs: use a solid panel behind the logo rather than a shadow or an
  outline.

## What not to do

- Don't stretch, squash, or scale the axes independently.
- Don't recolor the chevrons, including "just for this one campaign".
- Don't rotate or tilt it.
- Don't add shadows, glows, outlines, or bevels.
- Don't rebuild the wordmark in a different typeface — the outlines are the
  artwork.
- Don't put the light-ground version on dark, or the dark-ground version on
  light.
- Don't box the mark in a circle or a rounded square — platforms crop it for you,
  and the assets are already sized for that crop.
- Don't reorder or recolor the chevron sequence. Purple leads, left to right.

## The gold cut

The third chevron changes from `#12B981` to `#EFB338` exactly at its elbow. It's
one path with a hard-stop gradient, not two strokes — the round join belongs to
a single path, so there's no seam to drift as the logo scales. If you rebuild
the mark in another tool, keep it that way.

## Files and tokens

`tokens/` carries the palette as `brand.css` (custom properties, with a
`prefers-color-scheme` block), `brand.scss`, `brand.json`, and a Tailwind color
fragment. Import from there rather than pasting hexes, so a palette change
propagates.

`gen.js` regenerates every export from the constants at the top of the file.
Run `node gen.js` after any change, then re-cut `favicon.ico` from
`favicon-48.png`.
