// SPDX-License-Identifier: AGPL-3.0-only

// A chart edge is a claim: the two points are successive observations of one
// comparable code line. Recorded time alone cannot establish that claim when
// the dashboard contains receipts from every branch.

const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => [k, canonical(v)])
    );
  return value;
};

const instrumentKey = (record) =>
  JSON.stringify(
    canonical({
      target_model: record.target_model ?? '',
      served_by: record.served_by ?? '',
      hardware: {
        gpu: record.hardware?.gpu ?? '',
        driver: record.hardware?.driver ?? '',
        perf_class: record.perf_class ?? ''
      },
      params: record.params ?? {},
      serve_overrides: record.serve_overrides ?? {}
    })
  );

/**
 * Assign the nearest earlier comparable ancestor to each time-sorted record.
 *
 * Unknown ancestry fails closed. Mutates the records because the generator
 * serializes this field into gates.generated.json.
 */
const recordKey = (record) =>
  JSON.stringify([
    record.benchmark_id ?? '',
    record.target_model ?? '',
    record.served_by ?? '',
    record.git_sha,
    record.recorded_at
  ]);

export function assignTrendPredecessors(records, isAncestor) {
  const instruments = records.map(instrumentKey);
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    let predecessor;
    for (let prior = index - 1; prior >= 0; prior -= 1) {
      const candidate = records[prior];
      if (
        instruments[prior] === instruments[index] &&
        isAncestor(candidate.git_sha, record.git_sha)
      ) {
        predecessor = candidate;
        break;
      }
    }
    record.trend_predecessor = predecessor ? recordKey(predecessor) : '';
  }
  return records;
}

/** Return only generator-proven edges for records or `{ rec, ... }` points. */
export function trendEdges(items) {
  const byKey = new Map(items.map((item) => [recordKey(item.rec ?? item), item]));
  return items.flatMap((item) => {
    const record = item.rec ?? item;
    const predecessor = byKey.get(record.trend_predecessor);
    return predecessor ? [[predecessor, item]] : [];
  });
}
