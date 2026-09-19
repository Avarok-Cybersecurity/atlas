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

// static/brand/*.svg are git symlinks. Linux reads through one, a Windows checkout
// without symlink support leaves a text file holding the target's path. The ledger
// was once written on one and failed on the other. Both must hash the same bytes.
test('a symlink and its Windows stand-in read as the same file', async () => {
  const { readAsset } = await import('../../../scripts/guide/facts.mjs');
  const { mkdtempSync, writeFileSync, rmSync, mkdirSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const dir = mkdtempSync(join(tmpdir(), 'ledger-'));
  try {
    mkdirSync(join(dir, 'masters'));
    mkdirSync(join(dir, 'served'));
    writeFileSync(join(dir, 'masters', 'mark.svg'), '<svg>the real artwork</svg>');
    writeFileSync(join(dir, 'served', 'mark.svg'), '../masters/mark.svg'); // what Windows checks out
    writeFileSync(join(dir, 'served', 'note.svg'), '<svg/>'); // short, but not a path
    writeFileSync(join(dir, 'served', 'dangling.svg'), '../masters/gone.svg'); // a path to nothing
    expect(readAsset(join(dir, 'served', 'mark.svg')).toString()).toBe('<svg>the real artwork</svg>');
    expect(readAsset(join(dir, 'served', 'note.svg')).toString()).toBe('<svg/>');
    expect(readAsset(join(dir, 'served', 'dangling.svg')).toString()).toBe('../masters/gone.svg');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
