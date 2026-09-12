// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "bun:test";
import { benchmarkHighlight, legacyEngineDestination } from "./marketing.js";

const data = {
  rows: [
    {
      c: 8,
      atlas: 60,
      best_baseline_id: "fast",
      baselines: [
        { id: "slow", tok_s: 30 },
        { id: "fast", tok_s: 50, label: "Baseline fast" },
      ],
    },
    {
      c: 1,
      atlas: 10,
      best_baseline_id: "slow",
      baselines: [{ id: "slow", tok_s: 8, label: "Baseline slow" }],
    },
  ],
};
test("the highlight uses the highest measured concurrency and its fastest published baseline", () => {
  const result = benchmarkHighlight(data);
  expect(result.concurrency).toBe(8);
  expect(result.baseline).toBe(50);
  expect(result.ratio).toBe(1.2);
  expect(result.atlasWidth).toBe(100);
  expect(result.baselineWidth).toBeCloseTo(83.333);
  expect(result.improved).toBe(true);
});
test("a future slower run is represented honestly and bars stay in range", () => {
  const result = benchmarkHighlight({
    rows: [
      {
        c: 16,
        atlas: 40,
        best_baseline_id: "base",
        baselines: [{ id: "base", tok_s: 50, label: "Baseline" }],
      },
    ],
  });
  expect(result.improved).toBe(false);
  expect(result.ratio).toBe(0.8);
  expect(result.atlasWidth).toBe(80);
  expect(result.baselineWidth).toBe(100);
});
test("missing or invalid evidence cannot turn into a marketing claim", () => {
  for (const fixture of [
    { rows: [] },
    { rows: [{ c: 8, atlas: 10, best_baseline_id: "missing", baselines: [] }] },
    {
      rows: [{ c: 8, atlas: 10, best_baseline_id: "zero", baselines: [{ id: "zero", tok_s: 0 }] }],
    },
  ]) {
    expect(() => benchmarkHighlight(fixture)).toThrow();
  }
});
test("legacy technical fragments keep their precise destination and query", () => {
  expect(legacyEngineDestination("#faq", "?ref=docs")).toBe("/engine.html?ref=docs#faq");
  expect(legacyEngineDestination("#hardware", "")).toBe("/engine.html#hardware");
  expect(legacyEngineDestination("#%66aq", "")).toBe("/engine.html#faq");
  for (const hash of [
    "#verified",
    "#models",
    "#run",
    "#why-atlas",
    "#not-a-section",
    "#%E0%A4%A",
  ]) {
    expect(legacyEngineDestination(hash, "")).toBeNull();
  }
});
