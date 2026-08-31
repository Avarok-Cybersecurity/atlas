// SPDX-License-Identifier: AGPL-3.0-only
//
// Lighthouse's `link-text` SEO audit fails a link whose visible text is a
// non-descriptive phrase ("here", "more", "start", ...), and the site gate
// asserts a PERFECT SEO score, so one such label reds the whole PR.
//
// This is a real trap rather than a hypothetical: renaming the "Get running"
// section to "Start" on 2026-08-31 made the nav link, the drawer link and the
// hero CTA all read exactly "Start", dropping SEO 1.00 -> 0.92 on both audited
// pages. An `aria-label` does NOT rescue it — lighthouse 13.4.1
// (core/audits/seo/link-text.js) tests `link.text`, the anchor's innerText,
// never the accessible name. Only the visible text counts.
//
// The list below is the English set from that audit, copied verbatim. It is a
// duplicate of upstream by necessity (the audit is not importable here), so if
// a future Lighthouse adds a phrase this test will lag it — which is why it
// asserts against OUR link labels rather than pretending to be the audit.
import { test, expect } from 'bun:test';
import { nav, hero } from './data.js';

const NON_DESCRIPTIVE = new Set([
  'click here', 'click this', 'go', 'here', 'information', 'learn more', 'more',
  'more info', 'more information', 'right here', 'read more', 'see more', 'start', 'this'
]);

test('no nav link label is non-descriptive link text', () => {
  for (const l of nav.links) {
    expect(
      NON_DESCRIPTIVE.has(l.text.trim().toLowerCase()),
      `nav link "${l.text}" is on Lighthouse's non-descriptive list — it fails the SEO gate`
    ).toBe(false);
  }
});

test('no hero call-to-action is non-descriptive link text', () => {
  for (const key of ['primaryCta', 'secondaryCta', 'discordCta']) {
    const label = hero[key];
    if (!label) continue;
    expect(
      NON_DESCRIPTIVE.has(label.trim().toLowerCase()),
      `hero.${key} = "${label}" is on Lighthouse's non-descriptive list — it fails the SEO gate`
    ).toBe(false);
  }
});
