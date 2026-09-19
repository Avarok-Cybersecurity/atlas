// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import {
  RUNG_ALL,
  formatDashboardHash,
  isDeepLink,
  parseDashboardHash,
  resolveRung,
  resolveSubject,
  resolveTab
} from './dashboard-link.js';

// The shape the dashboard passes: outer tab ids, subject ids in the owner's
// order, and the union of rungs the subjects declare.
const known = {
  tabIds: ['agentic', 'bfcl', 'ttft', 'concurrency'],
  subjectIds: ['qwen38-27b', 'qwen36-35b-a3b', 'qwen38-27b-dflash'],
  rungs: [1, 2, 4, 8, 16, 32, 64, 128]
};

describe('parseDashboardHash', () => {
  test('reads the documented link, with or without its #', () => {
    const want = { tab: 'concurrency', subject: 'qwen36-35b-a3b', c: 64 };
    expect(parseDashboardHash('#bench=concurrency&subject=qwen36-35b-a3b&c=64', known)).toEqual(want);
    expect(parseDashboardHash('bench=concurrency&subject=qwen36-35b-a3b&c=64', known)).toEqual(want);
  });

  test('no hash is not a deep link, and still carries the defaults', () => {
    for (const empty of ['', '#', undefined, null]) {
      const link = parseDashboardHash(empty, known);
      expect(link).toEqual({ tab: null, subject: 'qwen38-27b', c: RUNG_ALL });
      expect(isDeepLink(link)).toBe(false);
    }
  });

  test('an unknown tab is not a deep link; unknown subject and rung fall to their defaults', () => {
    expect(parseDashboardHash('#bench=decode', known).tab).toBeNull();
    expect(parseDashboardHash('#bench=concurrency&subject=nvidia-35b', known).subject).toBe('qwen38-27b');
    expect(parseDashboardHash('#bench=concurrency&c=3', known).c).toBe(RUNG_ALL);
    expect(isDeepLink(parseDashboardHash('#bench=ttft', known))).toBe(true);
  });

  test('only the exact decimal spelling of a declared rung selects it', () => {
    for (const bad of ['064', '64.0', ' 64', '64 ', '0', '-4', '1e2', 'C=64', '', 'ALL']) {
      expect(parseDashboardHash(`#bench=concurrency&c=${encodeURIComponent(bad)}`, known).c).toBe(RUNG_ALL);
    }
    expect(parseDashboardHash('#bench=concurrency&c=128', known).c).toBe(128);
    expect(parseDashboardHash('#bench=concurrency&c=all', known).c).toBe(RUNG_ALL);
  });

  test('never throws on hostile input', () => {
    for (const junk of ['#%E0%A4%A', '#bench=%00', '#=&&=&bench', '#bench=concurrency&bench=ttft', 42, {}, '#'.repeat(3)]) {
      expect(() => parseDashboardHash(junk, known)).not.toThrow();
    }
    // A repeated key takes the first value, so a link cannot be "amended" by appending.
    expect(parseDashboardHash('#bench=concurrency&bench=ttft', known).tab).toBe('concurrency');
  });

  test('a stale link to a tab or subject that no longer exists is not honoured', () => {
    const fewer = { tabIds: ['ttft'], subjectIds: ['qwen38-27b'], rungs: [1] };
    const link = parseDashboardHash('#bench=concurrency&subject=qwen36-35b-a3b&c=64', fewer);
    expect(link).toEqual({ tab: null, subject: 'qwen38-27b', c: RUNG_ALL });
  });

  test('refuses to run without the known lists rather than treating everything as unknown', () => {
    expect(() => parseDashboardHash('#bench=ttft', {})).toThrow(/known\.tabIds/);
    expect(() => parseDashboardHash('#bench=ttft', { tabIds: [], subjectIds: [] })).toThrow(/known\.rungs/);
  });
});

describe('the resolvers', () => {
  test('resolveSubject is null only when there are no subjects at all', () => {
    expect(resolveSubject('anything', [])).toBeNull();
    expect(resolveSubject(null, ['a', 'b'])).toBe('a');
    expect(resolveSubject('b', ['a', 'b'])).toBe('b');
  });

  test('resolveTab and resolveRung do not coerce', () => {
    expect(resolveTab(undefined, ['undefined'])).toBeNull();
    expect(resolveRung(64, [64])).toBe(RUNG_ALL); // a number is not a hash value
    expect(resolveRung('64', [64])).toBe(64);
  });
});

describe('formatDashboardHash', () => {
  test('round-trips every valid state through parse', () => {
    for (const tab of known.tabIds) {
      for (const subject of known.subjectIds) {
        for (const c of [...known.rungs, RUNG_ALL]) {
          const state = { tab, subject, c };
          expect(parseDashboardHash('#' + formatDashboardHash(state), known)).toEqual(state);
        }
      }
    }
  });

  test('writes the documented spelling', () => {
    expect(formatDashboardHash({ tab: 'concurrency', subject: 'qwen36-35b-a3b', c: 64 })).toBe(
      'bench=concurrency&subject=qwen36-35b-a3b&c=64'
    );
    expect(formatDashboardHash({ tab: 'concurrency', subject: 'qwen38-27b', c: RUNG_ALL })).toBe(
      'bench=concurrency&subject=qwen38-27b&c=all'
    );
  });

  test('a non-concurrency link carries no subject or rung, and no tab means no link', () => {
    expect(formatDashboardHash({ tab: 'ttft' })).toBe('bench=ttft');
    expect(formatDashboardHash({ tab: null, subject: 'qwen38-27b', c: 64 })).toBe('');
  });
});
