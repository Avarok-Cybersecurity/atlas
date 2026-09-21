// =============================================================================
// gate-variants.js — which gate ids are two views of ONE instrument.
//
// The dashboard's default shape is one benchmark id per chart. That is right
// almost everywhere, and wrong for the concurrency ladder: `concurrency-sweep`
// and `concurrency-sweep-dflash2` run the same fixture at the same rungs on the
// same checkpoint, and differ only in whether the served engine speculates.
// Drawn as two panels they answer "how did each move over time"; drawn as two
// lines on one panel they answer "what does speculation buy at each rung",
// which is the question the gate pair exists for.
//
// Pure and dependency-free ON PURPOSE. `gates.js` imports
// `$lib/gates.generated.json`, and the `$lib` Vite alias does not resolve
// outside a Vite build, so anything importing it cannot be unit-tested under
// bun. The policy lives here so it can be measured; the components render what
// it says and add nothing of their own. Same split, same reason, as
// series-colors.js.
// =============================================================================

/// A group of gate ids that share one chart. `primary` decides the panel specs
/// and the section's headline tiles; `members` is draw order.
///
/// The dash is the ONLY visual difference between members, deliberately.
/// Series colour follows the MODEL everywhere else in this dashboard, and both
/// members serve the same checkpoint — giving DFlash2 its own hue would say
/// "different model" in the one vocabulary this palette has. A dash says
/// "same subject, different configuration", which is exactly the claim, and it
/// is the idiom ConcurrencyLadder already uses for `vllm-nospec`.
export const VARIANT_GROUPS = [
  {
    tab: 'concurrency',
    primary: 'concurrency-sweep',
    members: [
      { bench: 'concurrency-sweep', label: 'no drafter', dash: null },
      { bench: 'concurrency-sweep-dflash2', label: 'DFlash2', dash: '5 4' }
    ]
  }
];

const MEMBER = new Map(
  VARIANT_GROUPS.flatMap((g) => g.members.map((m) => [m.bench, { ...m, group: g }]))
);

/// The group a benchmark id belongs to, or `null` for the ordinary one-id-one-
/// chart case.
export const groupFor = (benchId) => MEMBER.get(benchId)?.group ?? null;

/// Every benchmark id that is drawn inside some group. Used to keep grouped
/// members out of the ungrouped section list so they are not drawn twice.
export const groupedBenches = new Set(MEMBER.keys());

/// The dash pattern for a record's variant — `null` for a solid line, which is
/// also the answer for any record outside a group.
export const dashFor = (benchId) => MEMBER.get(benchId)?.dash ?? null;

/// The human label for a variant, e.g. "DFlash2". `null` outside a group.
export const variantLabel = (benchId) => MEMBER.get(benchId)?.label ?? null;

/// Records of every member of `group`, merged into one chronological list.
///
/// Chronological across variants is right for a chart whose x-axis is time,
/// and harmless for the ladder, which re-keys by rung. What must NOT happen is
/// a line joining two variants' points — that is prevented at the series
/// level by `splitByVariant`, not here.
export function groupRecords(group, recordsFor) {
  return group.members
    .flatMap((m) => recordsFor(m.bench))
    .sort((a, b) => a.recorded_at - b.recorded_at);
}

/// Split records into one bucket per variant, in the group's draw order,
/// dropping variants with nothing to draw.
///
/// This is the function that keeps a DFlash2 point from being joined to a
/// no-drafter point. Two variants interleaved in one polyline would read as a
/// regression and recovery on every alternation, which is the single most
/// misleading thing this chart could render.
export function splitByVariant(records) {
  const buckets = [];
  for (const [bench, m] of MEMBER) {
    const rs = records.filter((r) => r.benchmark_id === bench);
    if (rs.length > 0) buckets.push({ bench, label: m.label, dash: m.dash, records: rs });
  }
  // Anything not in a group stays one unlabelled bucket, so a plain benchmark
  // renders exactly as it did before this module existed.
  const rest = records.filter((r) => !MEMBER.has(r.benchmark_id));
  if (rest.length > 0) buckets.push({ bench: null, label: null, dash: null, records: rest });
  return buckets;
}

/// Whether `record` is the newest of its own variant within `records`.
///
/// The ladder chart fades every run but the newest. With two variants that
/// cannot be "the last element": the newest DFlash2 run is usually not the
/// newest run overall, so a single global latest would leave one whole
/// variant permanently faded and read as stale.
export function isLatestOfVariant(record, records) {
  const mine = records.filter((r) => r.benchmark_id === record.benchmark_id);
  return mine.length > 0 && mine[mine.length - 1].recorded_at <= record.recorded_at;
}
