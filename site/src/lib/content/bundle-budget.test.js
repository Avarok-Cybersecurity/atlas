// SPDX-License-Identifier: AGPL-3.0-only
//
// gates.generated.json is a megabyte. The developer pages chart it and pay for
// it knowingly. A marketing page must never load it: the first facelift build
// imported it from content/live.js to print one integer, and that put a 1 MB
// chunk on the front page and eleven others while every test stayed green.
//
// The rule is on imports, which is where the mistake is made: nothing a
// marketing page loads may import the gate record set or anything that does.
// The counts those pages print come from live.generated.json (scripts/gen-live.mjs).
import { expect, test } from 'bun:test';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = join(import.meta.dir, '..', '..');
const walk = (dir) =>
  readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });

const MARKETING = [join(SRC, 'lib', 'content'), join(SRC, 'lib', 'components', 'avarok'), join(SRC, 'routes', '(marketing)'), join(SRC, 'lib', 'broll')];
const HEAVY = ['gates.generated.json', '$lib/gates.js', '$lib/deck/content.js'];
// The one marketing page that exists to chart the gate records, the way
// /engine does. It pays for the data knowingly and nothing else may.
const ALLOWED = [join('routes', '(marketing)', 'benchmarks', '+page.svelte')];

test('nothing a marketing page loads imports the gate record set', () => {
  const offenders = [];
  for (const file of MARKETING.flatMap(walk).filter((f) => /\.(svelte|js)$/.test(f) && !f.endsWith('.test.js') && !ALLOWED.some((a) => f.endsWith(a)))) {
    const text = readFileSync(file, 'utf8');
    for (const line of text.split('\n')) {
      if (!/^\s*import\b/.test(line)) continue;
      const hit = HEAVY.find((h) => line.includes(h));
      if (hit) offenders.push(`${relative(SRC, file)} imports ${hit}`);
    }
  }
  expect(offenders).toEqual([]);
});

test('the counts file the marketing pages read stays small', () => {
  const bytes = statSync(join(SRC, 'lib', 'live.generated.json')).size;
  expect(bytes).toBeLessThan(2048);
});

test('live.generated.json carries the counts live.js prints', () => {
  const counts = JSON.parse(readFileSync(join(SRC, 'lib', 'live.generated.json'), 'utf8'));
  expect(counts.recipes).toBeGreaterThan(0);
  expect(counts.gates.concurrencyRecords).toBeGreaterThan(0);
  expect(counts.gates.concurrencyPass).toBeLessThanOrEqual(counts.gates.concurrencyRecords);
});
