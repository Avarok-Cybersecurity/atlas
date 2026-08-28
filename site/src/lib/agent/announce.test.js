// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { ANNOUNCE_DEBOUNCE_MS, announcement } from './announce.js';

const warn = { severity: 'warning', nodeName: 'dgx2', kind: 'disk_low', detail: 'cache fs at 94%' };
const crit = { severity: 'critical', nodeName: 'dgx3', kind: 'sm_clock_clamped', detail: '' };

describe('the single live region speaks only on severity transitions', () => {
  test('an unchanged worst severity says nothing, however often it renders', () => {
    expect(announcement('warning', [warn])).toBeNull();
    expect(announcement('warning', [warn, warn])).toBeNull();
    expect(announcement(null, [])).toBeNull();
  });

  test('the first alert is announced, verbatim from the alert', () => {
    expect(announcement(null, [warn])).toEqual({
      severity: 'warning',
      text: 'warning: dgx2: cache fs at 94%'
    });
  });

  test('escalation and de-escalation both speak', () => {
    expect(announcement('warning', [crit, warn])?.severity).toBe('critical');
    expect(announcement('critical', [warn])?.severity).toBe('warning');
  });

  test('all-clear is a transition too, not silence', () => {
    expect(announcement('warning', [])).toEqual({ severity: null, text: 'All alerts cleared.' });
  });

  test('an empty detail falls back to the kind, never to an empty sentence', () => {
    expect(announcement(null, [crit])?.text).toBe('critical: dgx3: sm clock clamped');
  });

  test('a nameless, kindless alert still forms a sentence', () => {
    expect(announcement(null, [{ severity: 'info' }])?.text).toBe('info: a machine: alert');
  });

  test('unknown severities are refused rather than voiced', () => {
    // A hostile or future severity must not reach the class attribute or the
    // announcement; the worst KNOWN state is what matters, and here there is
    // none — which reads as all-clear against a previously announced warning.
    expect(announcement('warning', [{ severity: 'apocalyptic' }])).toEqual({
      severity: null,
      text: 'All alerts cleared.'
    });
    expect(announcement(null, [{ severity: 'apocalyptic' }])).toBeNull();
  });

  test('detail is sanitized before it reaches a screen reader', () => {
    const hostile = { severity: 'info', nodeName: 'dgx2', detail: 'ok‮gpj.exe' };
    expect(announcement(null, [hostile])?.text).not.toContain('‮');
  });

  test('the debounce constant is a settling window, not a render throttle', () => {
    expect(ANNOUNCE_DEBOUNCE_MS).toBeGreaterThan(0);
    expect(ANNOUNCE_DEBOUNCE_MS).toBeLessThanOrEqual(3000);
  });
});
