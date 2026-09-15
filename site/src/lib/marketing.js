// SPDX-License-Identifier: AGPL-3.0-only
import palette from "../../../assets/brand/tokens/brand.json";

export const ENGINE = "/engine.html";
const { chevron, ground, ui, wordmark, tagline } = palette.color;
export const brandStyle = Object.entries({
  purple: chevron.one,
  cyan: chevron.two,
  green: chevron.threeUpper,
  gold: chevron.threeLower,
  ink: ground.dark,
  paper: ground.light,
  "gray-text": ui.grayText,
  "wordmark-dark": wordmark.onDark,
  "tagline-dark": tagline.onDark,
})
  .map(([name, value]) => `--atlas-${name}:${value}`)
  .join(";");

export const legacySections = [
  { id: "proof", label: "Project milestones" },
  { id: "news", label: "Atlas news" },
  { id: "hardware", label: "Verified hardware" },
  { id: "community", label: "Community" },
  { id: "contribute", label: "Contribute" },
  { id: "roadmap", label: "Roadmap" },
  { id: "mission", label: "Mission" },
  { id: "faq", label: "Frequently asked questions" },
  { id: "reach", label: "Contact Atlas" },
];

export function legacyEngineDestination(hash, search) {
  let id;
  try {
    id = decodeURIComponent(hash.replace(/^#/, ""));
  } catch {
    return null;
  }
  return legacySections.some((section) => section.id === id) ? `${ENGINE}${search}#${id}` : null;
}

// Derived from the generated ladder. A missing baseline fails the build rather
// than leaving an unsubstantiated performance claim on the front page.
export function benchmarkHighlight(ladder) {
  if (!Array.isArray(ladder.rows) || ladder.rows.length === 0)
    throw new Error("Missing benchmark rows");
  const row = [...ladder.rows].sort((a, b) => b.c - a.c)[0];
  const baseline = row.baselines?.find((item) => item.id === row.best_baseline_id);
  if (![row.c, row.atlas, baseline?.tok_s].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("Invalid benchmark evidence");
  }
  const max = Math.max(row.atlas, baseline.tok_s);
  return {
    concurrency: row.c,
    atlas: row.atlas,
    baseline: baseline.tok_s,
    baselineLabel: baseline.label,
    ratio: row.atlas / baseline.tok_s,
    improved: row.atlas > baseline.tok_s,
    atlasWidth: (row.atlas / max) * 100,
    baselineWidth: (baseline.tok_s / max) * 100,
  };
}
