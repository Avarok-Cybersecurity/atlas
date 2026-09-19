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
//
// Extended 2026-08-30, when the per-model split landed. Three checkpoints were
// already being charted with no entry here, so all three fell through to the
// single fallback grey and rendered identically — and because the chart used
// to colour a whole series from its FIRST point's model, they were in practice
// drawn in copper. That is what made Gemma's legitimate 23,484 ms cold start
// look like an absurd Qwen outlier on ttft-cold-gate.
//
// Searched with the SAME method as the trio above (CIEDE2000 under Vienot
// dichromat simulation), which reproduces this file's existing normal-vision
// figures exactly (49.6 / 55.3 / 39.9), so the numbers below are continuous
// with them. The simulated-deficiency figures differ by ~1-3 from the 2026-08
// set, so treat those as re-measured rather than identical.
//
// The search fixed the shipped trio, required >=4:1 on both surfaces, kept
// clear of the UI accent #BE9DF8 (a series must not read as a link) and
// reserved the green band (a series must not read as a PASS verdict), then
// maximised the worst pair under normal/protan/deutan vision:
//   rose   #cd517a   4.52:1 / 4.02:1
//   citron #d5e88a  14.07:1 / 12.53:1
//   sky    #a1e0f7  12.98:1 / 11.56:1
// Worst pair over all 15, normal/protan/deutan: 16.4 (teal-sky).
// Worst against the fallback grey: 11.7 (rose).
//
// ASSIGNMENT IS NOT ARBITRARY. sky goes to nvidia's NVFP4 re-quant because
// copper is the FP8 flagship of the same family, and FP8-vs-NVFP4 is the one
// comparison a reader must never misread: copper vs sky scores 46.8 at worst,
// where copper vs citron would have been 16.6.
//
// KNOWN WEAKNESS, recorded rather than hidden: the worst tritan pair is 8.5
// (teal-sky), below the >=15 these hues reach for normal vision. Tritanopia is
// ~3 orders of magnitude rarer than protan/deutan, and pushing it higher costs
// the common-vision worst case, which would trade ~8% of male readers for
// ~0.003%. Identity is double-encoded anyway: every series carries a coloured
// end label and a legend entry, and every marker is shape-coded.
//
// Split by theme 2026-09-19. The six hexes above were one set shared by both
// themes, and on the light theme's white ground three of them are under the
// 3:1 floor: teal 1.96, citron 1.33, sky 1.45 (copper 3.02 passes white but
// reads 2.65 on --card-2). One hex cannot serve both grounds — >=3:1 on white
// needs a luminance <=0.30, >=3:1 on the dark --card needs >=0.136, and the
// pale teal/citron/sky are nowhere near that band — so each theme now declares
// its own `--series-<slug>` tokens in web-shared/avarok-tokens.css and
// colorFor() resolves through the token, with the dark hex as the var()
// fallback (the idiom ConcurrencyLadder.svelte already uses for its baselines).
// The DARK set is unchanged, by value and by assignment.
//
// The LIGHT set was searched with a re-implementation of this file's method
// (CIEDE2000 under Vienot 1999 dichromat simulation), because the original
// search script was never committed. Calibration: it reproduces the trio's
// documented normal-vision figures exactly (49.6 / 55.3 / 39.9) and the
// protan/deutan ones within 0.3 (60.1 / 68.2, 25.0 / 26.5, 42.5 / 32.3 against
// 60.2 / 68.3, 25.0 / 26.4, 42.4 / 32.6), so its figures are continuous with
// the ones above to about half a unit — but they are that implementation's,
// not the original's. By the same implementation the dark set's worst pair
// over normal/protan/deutan is 16.2 and its worst tritan pair 8.7.
//
// Constraints: >=4.5:1 on white and >=4.0:1 on --card-2 (the 2026-08-30 bar);
// each slug within 15 degrees of its dark hue, copper only toward ochre and
// teal only away from the light --green (a series must not read as a PASS);
// L* >= 30 and C* >= 20 so no series reads as ink or as the fallback grey; and
// against the light --accent, --green, --t2, --t3 and the fallback, at least
// the separation the dark set holds against its own (9.5 / 9.2 / 8.4 / 11.7 /
// 6.3). Then maximise the worst pair. The result:
//   copper #a2672c   4.65 / 4.27 / 4.08   (white / --bg2 / --card-2)
//   steel  #466dc1   4.98 / 4.56 / 4.36
//   teal   #2d5b4c   7.74 / 7.10 / 6.78
//   rose   #832837   9.04 / 8.28 / 7.92
//   citron #565200   8.07 / 7.40 / 7.07
//   sky    #005977   7.79 / 7.14 / 6.82
// Worst pair over all 15, normal/protan/deutan: 10.0 (teal-rose protan).
// The pair the concurrency tabs rest on, copper-teal: 35.9 / 17.4 / 25.7
// (dark: 55.3 / 25.0 / 26.4). teal vs the light --t2/--t3 ink the vLLM
// baselines are drawn in: 10.9 / 11.9 (dark: 8.4 / 15.9); copper: 32.7 / 25.9.
// Worst tritan pair 5.2 (rose-citron), against the dark set's 8.7 — the same
// documented trade as above. A light ground cannot use the lightness spread
// the dark set relies on (every series must sit at L* <= 49 to clear white),
// so 10.0 is the regime, not a slip: the plan's first candidate, lightness
// lowered under normal vision only, scored 15.4 normal but 4.9 deutan
// (copper-citron) and 0.5 tritan (copper-rose) by this implementation.
const SERIES = [
  // model, slug, dark hex, light hex
  ['Qwen/Qwen3.6-35B-A3B-FP8', 'copper', '#ee6f2f', '#a2672c'],
  ['unsloth/Qwen3.6-27B-NVFP4', 'steel', '#2f88ee', '#466dc1'],
  ['unsloth/Qwen3.8-27B-NVFP4', 'teal', '#51cdb0', '#2d5b4c'],
  ['bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16', 'rose', '#cd517a', '#832837'],
  ['ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4', 'citron', '#d5e88a', '#565200'],
  ['nvidia/Qwen3.6-35B-A3B-NVFP4', 'sky', '#a1e0f7', '#005977']
];
/** model -> the `--series-<slug>` token suffix. */
export const MODEL_SLUGS = Object.fromEntries(SERIES.map(([model, slug]) => [model, slug]));
/** model -> dark-theme hex; also the var() fallback colorFor() emits. */
export const MODEL_COLORS = Object.fromEntries(SERIES.map(([model, , dark]) => [model, dark]));
/** model -> light-theme hex, mirrored by the `[data-theme="light"]` block. */
export const MODEL_COLORS_LIGHT = Object.fromEntries(SERIES.map(([model, , , light]) => [model, light]));
// The fallback is a series colour too: an unrecognised model still gets drawn.
// One hex serves both themes: 3.69:1 / 3.28:1 on the dark grounds, 5.09:1 /
// 4.67:1 / 4.46:1 on the light ones.
export const UNKNOWN_MODEL_COLOR = '#6f6a8d';
/**
 * The colour a series is drawn in, as a CSS value: the theme's token with the
 * dark hex as fallback, so an SVG rendered without the token file (a snapshot,
 * an embed) still gets the palette the chart was designed on.
 */
export const colorFor = (model) => {
  const slug = MODEL_SLUGS[model];
  return slug ? `var(--series-${slug}, ${MODEL_COLORS[model]})` : UNKNOWN_MODEL_COLOR;
};

/**
 * The human-facing part of a checkpoint id: everything after the last `/`.
 *
 * Lives here rather than in `gates.js` because that module imports
 * `$lib/gates.generated.json`, which nothing under `bun test` can resolve —
 * anything importing it stops being unit-testable. `gates.js` re-exports this.
 *
 * The quant suffix is deliberately kept: `Qwen3.6-35B-A3B-FP8` and
 * `Qwen3.6-35B-A3B-NVFP4` are different subjects, and a chart that shortened
 * both to `Qwen3.6-35B-A3B` would make the one comparison that matters most
 * impossible to read.
 */
export const shortModel = (model) => (model || '').split('/').pop() || model;
