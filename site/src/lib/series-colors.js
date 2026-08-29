// =============================================================================
// series-colors.js — the benchmark chart's series palette, and nothing else.
//
// Split out of gates.js so it can be measured directly. gates.js imports
// `$lib/gates.generated.json`, and `$lib` is a Vite alias that does not exist
// outside a Vite build, so anything importing gates.js cannot be unit-tested.
// The palette is the part that needs testing — see series-contrast.test.js.
// =============================================================================

// Series color follows the MODEL (the entity), never the tab or verdict.
//
// Re-derived 2026-08-25 when the canvas moved from paper to the deep-violet
// dark theme. The previous trio (copper #b5622f, steel #1f6a9e, teal #1c7a6b)
// was validated against #f4f0e8/#fbf9f3 and does not survive the inversion:
// steel falls to 2.87:1 on the card surface, under the >=3:1 floor this palette
// has always held to. The old comment asked for all three to be re-derived
// together if the palette were ever revisited, so they were.
//
// Same three hue families, lifted for a near-black canvas. Measured with
// CIEDE2000 under Vienot dichromat simulation. The method reproduces the
// previous comment's figures exactly on the old surfaces (43.8 / 16.3 / 27.1,
// 26.1 / 27.5 / 20.3), so these numbers ARE continuous with the 2026-08-14 set
// and can be compared directly.
//
// Re-measured 2026-08-29 when the surfaces moved to the brand reference ramp
// (--bg #14111f -> #0F1216, --card #201b30 -> #191E27). The hues are unchanged
// and did not need to change: the ground got DARKER, so every ratio rose and
// the >=3:1 floor gained margin rather than losing it. The pairwise CIEDE2000
// separations below do not depend on the background at all.
//   copper #ee6f2f  6.21:1 / 5.53:1   (was 6.15 / 5.51)
//   steel  #2f88ee  5.25:1 / 4.68:1   (was 5.20 / 4.66)
//   teal   #51cdb0  9.58:1 / 8.52:1   (was 9.48 / 8.49)
// series-contrast.test.js pins this, so the next palette move fails the build
// here instead of quietly dropping a series under the floor.
// pairwise, normal / protan / deutan:
//   copper vs steel  49.6 / 60.2 / 68.3
//   copper vs teal   55.3 / 25.0 / 26.4
//   steel  vs teal   39.9 / 42.4 / 32.6   <- the load-bearing comparison
// Worst case 25.0, against 16.3 before; the steel-teal pair, which is the whole
// point of the series (3.8 vs 3.6-27B: same architecture, same draw, read as a
// generation-over-generation delta), improves from 20.3 to 32.6.
//
// Lightness was capped during the search. An unconstrained optimum scored 31.4
// but put teal at 14.4:1 — a near-white cyan that no longer reads as teal, and
// glares on a dark canvas. Separation is not worth spending the hue identity on.
// Exported so series-contrast.test.js can measure it against the token file
// rather than re-typing the hexes into the test — the point of the test is that
// the two cannot drift.
export const MODEL_COLORS = {
  'Qwen/Qwen3.6-35B-A3B-FP8': '#ee6f2f',
  'unsloth/Qwen3.6-27B-NVFP4': '#2f88ee',
  'unsloth/Qwen3.8-27B-NVFP4': '#51cdb0'
};
// The fallback is a series colour too: an unrecognised model still gets drawn.
export const UNKNOWN_MODEL_COLOR = '#6f6a8d';
export const colorFor = (model) => MODEL_COLORS[model] ?? UNKNOWN_MODEL_COLOR;
