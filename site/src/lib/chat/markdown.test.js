// SPDX-License-Identifier: AGPL-3.0-only

// renderMarkdown's output goes to {@html} in ChatMessage.svelte, and its input
// is the reply of a remote model that a page author does not control. The
// module's header claims it is "XSS-safe by construction"; until now nothing
// enforced that claim, so an edit to renderSegment could retire it silently.
//
// The assertion below is deliberately about *emitted markup*, not about
// substrings: escaped text legitimately contains "onerror=" and "<script>", and
// a test that greps for those reports a break where the renderer did its job.

import { expect, test } from 'bun:test';
import { renderMarkdown } from './markdown.js';

// Every tag the documented feature set can produce. Anything else reaching the
// DOM is a breakout, so the set is closed and the helper fails on a newcomer.
const ALLOWED = new Set(['p', 'br', 'a', 'strong', 'code', 'pre', 'ol', 'ul', 'li', 'sup']);

function liveMarkup(html) {
  const tags = [...html.matchAll(/<\/?([a-zA-Z][\w-]*)/g)].map((m) => m[1].toLowerCase());
  return {
    rogueTags: tags.filter((t) => !ALLOWED.has(t)),
    // An on*= handler inside a real tag, as opposed to inside escaped text.
    handlers: /<[a-z][^>]*\son[a-z]+\s*=/i.test(html),
    // The only href the renderer may emit is one it validated as http(s).
    foreignHref: /<a[^>]+href="(?!https?:)/i.test(html),
  };
}

function expectInert(src) {
  const { rogueTags, handlers, foreignHref } = liveMarkup(renderMarkdown(src));
  expect(rogueTags).toEqual([]);
  expect(handlers).toBe(false);
  expect(foreignHref).toBe(false);
}

test('a model reply full of markup reaches the DOM as text, not as elements', () => {
  for (const attack of [
    '<img src=x onerror=alert(1)>',
    '<script>alert(1)</script>',
    '<iframe src=//evil></iframe>',
    '<svg onload=alert(1)>',
    '**<svg onload=alert(1)>**',
    '`<img src=x onerror=alert(1)>`',
    '- <img src=x onerror=1>\n- second',
    '1. <img src=x onerror=1>',
  ]) {
    expectInert(attack);
  }
});

test('a link cannot smuggle a scheme or break out of its own href', () => {
  for (const attack of [
    '[click](javascript:alert(1))',
    '[click](data:text/html,<script>alert(1)</script>)',
    '[x](https://a.com/" onmouseover="alert(1))',
    '["><img src=x onerror=alert(1)>](https://a.com)',
    '[x](vbscript:alert(1))',
  ]) {
    expectInert(attack);
  }
});

// The fence language becomes a class attribute, which is the one place a raw
// capture is interpolated into markup rather than escaped.
test('a fence language cannot open a second attribute', () => {
  expectInert('```js" onload="alert(1)\nbody\n```');
  expect(renderMarkdown('```js\nx\n```')).toContain('<code class="language-js">');
});

test('a javascript: URL is left as text even when it looks like a link', () => {
  const html = renderMarkdown('[click](javascript:alert(1))');
  expect(html).not.toContain('<a ');
  expect(html).toContain('javascript:alert(1)');
});

// Correctness of the documented features — a renderer that escaped everything
// into oblivion would pass the tests above and be useless.
test('the documented markup still renders', () => {
  expect(renderMarkdown('**bold**')).toBe('<p><strong>bold</strong></p>');
  expect(renderMarkdown('`code`')).toBe('<p><code>code</code></p>');
  expect(renderMarkdown('- a\n- b')).toBe('<ul><li>a</li><li>b</li></ul>');
  expect(renderMarkdown('1. a\n2. b')).toBe('<ol><li>a</li><li>b</li></ol>');
  expect(renderMarkdown('See [1] there')).toContain('<sup class="cc-cite">[1]</sup>');
  const link = renderMarkdown('[t](https://a.com)');
  expect(link).toContain('href="https://a.com"');
  expect(link).toContain('rel="noopener nofollow"');
});

test('an ampersand in a URL survives as one character, not a double escape', () => {
  expect(renderMarkdown('[t](https://a.com/?a=1&b=2)')).toContain('href="https://a.com/?a=1&amp;b=2"');
});

test('empty and non-string input render nothing rather than throwing', () => {
  expect(renderMarkdown('')).toBe('');
  expect(renderMarkdown(undefined)).toBe('');
  expect(renderMarkdown(null)).toBe('');
  expect(renderMarkdown(42)).toBe('');
});

// The header claims "the only attributes ever emitted are a fixed rel/target
// pair and an https?-validated href". That was untrue in the DOM: the bold and
// citation passes run over a string already containing the emitted anchor, so a
// URL carrying `[`, `]` or `*` had its own href rewritten — closing the
// attribute early and dropping rel and target. The allowlist test above cannot
// see it: the injected attribute is not an `on*` handler and the href still
// begins `https:`.
function anchorAttrs(html) {
  return [...html.matchAll(/<a\b([^>]*)>/g)].map((m) =>
    [...m[1].matchAll(/([a-zA-Z-]+)\s*=/g)].map((a) => a[1]).sort()
  );
}

test('every emitted anchor carries exactly href, rel and target', () => {
  for (const src of [
    '[t](https://a.com)',
    '[**b**](https://a.com)',
    'See [1] and [t](https://a.com)',
    '[a](https://a.com) and [b](https://b.com)',
    '[x](https://e.com/[1]z)',
    '[x](https://e.com/**a) more **b',
    '[x](https://e.com/a[0]b*c)'
  ]) {
    for (const attrs of anchorAttrs(renderMarkdown(src))) {
      expect(attrs).toEqual(['href', 'rel', 'target']);
    }
  }
});

test('a URL carrying the rewrite characters produces no anchor at all', () => {
  // Left as text, which is the safe direction: a link that does not render is
  // visibly wrong, an anchor with its rel silently stripped is not.
  for (const src of ['[x](https://e.com/[1]z)', '[x](https://e.com/**a)']) {
    expect(renderMarkdown(src)).not.toContain('<a ');
  }
  // And the ordinary case still links.
  expect(renderMarkdown('[t](https://a.com/ok)')).toContain('<a href="https://a.com/ok"');
});
