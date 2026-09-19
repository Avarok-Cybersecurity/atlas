// SPDX-License-Identifier: AGPL-3.0-only
//
// The guide's memory must match the site. guide/ledger.json records every
// tracked name, address, link, licence line and asset with a revision and a
// date. If one of them changes and the ledger is not told, the map of the site
// is wrong and nobody knows when the value moved. So this fails, and says what
// to run. scripts/guide/facts.mjs defines what is tracked.
import { expect, test } from 'bun:test';
import { collectAssets, collectFacts, diffLedger, readLedger } from '../../../scripts/guide/facts.mjs';

const HINT = 'Run `bun x --bun vite build` then `bun run guide -- --note "what changed"` in site/, and commit guide/ and SITE-GUIDE.md.';

test('every tracked fact matches the ledger', async () => {
  const diff = diffLedger(readLedger(), await collectFacts(), collectAssets());
  expect({ changed: diff.changedFacts, removed: diff.removedFacts, hint: diff.changedFacts.length + diff.removedFacts.length ? HINT : '' }).toEqual({ changed: [], removed: [], hint: '' });
});

test('every tracked asset matches the ledger', async () => {
  const diff = diffLedger(readLedger(), await collectFacts(), collectAssets());
  expect({ changed: diff.changedAssets, removed: diff.removedAssets, hint: diff.changedAssets.length + diff.removedAssets.length ? HINT : '' }).toEqual({ changed: [], removed: [], hint: '' });
});

test('the ledger has a revision, a date and a change log entry for everything in it', () => {
  const ledger = readLedger();
  expect(ledger.changes.length).toBeGreaterThan(0);
  for (const [key, entry] of [...Object.entries(ledger.facts), ...Object.entries(ledger.assets)]) {
    expect(entry.rev, key).toBeGreaterThan(0);
    expect(entry.date, key).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  }
  const revs = ledger.changes.map((c) => c.rev);
  expect(revs).toEqual(revs.map((_, i) => i + 1));
});
