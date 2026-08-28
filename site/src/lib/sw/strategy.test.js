// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, it } from 'bun:test';
import { NEVER_CACHED, shouldPrecache, strategyFor } from './strategy.js';

describe('strategyFor', () => {
  // The rule with a consequence outside the browser: a cached installer is a
  // copy of a shell script the server may have replaced, handed to someone who
  // is about to pipe it into `sh`.
  it('never caches an install script', () => {
    for (const p of NEVER_CACHED) {
      expect(strategyFor(p)).toBe('bypass');
    }
  });

  // And the list is exact. A prefix or substring rule would bypass
  // `/install.sh.html` or `/docs/install.sh-notes`, quietly turning ordinary
  // pages into uncacheable ones.
  it('matches install scripts exactly, not loosely', () => {
    expect(strategyFor('/install.sh.html')).toBe('network-first');
    expect(strategyFor('/docs/install.sh')).toBe('network-first');
    expect(strategyFor('/install.shx')).toBe('network-first');
  });

  it('serves content-hashed assets from cache first', () => {
    expect(strategyFor('/_app/immutable/chunks/abc123.js')).toBe('cache-first');
    expect(strategyFor('/_app/immutable/assets/0.C4O6L5-m.css')).toBe('cache-first');
  });

  // `/_app/` without `immutable/` is NOT content-hashed — version.json lives
  // there and is how the client notices a deploy. Cache-first would make the
  // page unable to see that it is out of date.
  it('does not treat every /_app/ path as immutable', () => {
    expect(strategyFor('/_app/version.json')).toBe('network-first');
  });

  it('puts documents and static files on network-first', () => {
    for (const p of ['/', '/control', '/index.html', '/logo.svg', '/site.webmanifest']) {
      expect(strategyFor(p)).toBe('network-first');
    }
  });
});

describe('shouldPrecache', () => {
  // Precaching an install script would store it at install time, which is the
  // same leak by a different door — the fetch handler never gets a say.
  it('keeps install scripts out of the precache', () => {
    for (const p of NEVER_CACHED) {
      expect(shouldPrecache(p)).toBe(false);
    }
  });

  it('leaves out what only a scraper or a feature needs', () => {
    expect(shouldPrecache('/og-image.png')).toBe(false);
    expect(shouldPrecache('/lattice/lattice_server_bg.wasm')).toBe(false);
  });

  it('keeps what a visitor needs on the first paint', () => {
    for (const p of ['/logo.svg', '/favicon.ico', '/site.webmanifest', '/llms.txt']) {
      expect(shouldPrecache(p)).toBe(true);
    }
  });
});
