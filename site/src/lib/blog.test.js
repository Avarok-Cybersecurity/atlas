// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { headroom, signed } from './ladder.js';
import real from './ladder.generated.json';
import { faq } from './data.js';
import {
  posts,
  getPost,
  postHref,
  postUrl,
  indexUrl,
  writing,
  sitemapPaths,
  SITE_ORIGIN
} from './blog.js';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '../..');

const FORBIDDEN = /[:\u2014;!]/;
const SKIP_KEYS = new Set(['slug', 'src', 'href', 'date', 'type', 'id']);
const NEW_FAQ = [
  'What does Atlas mean by an AI swarm?',
  'What is a context bus?',
  'What is an AI motherboard?',
  'What is Atlas Enterprise?'
];

function visibleStrings(node, key = '') {
  const out = [];
  const walk = (v, k) => {
    if (v == null) return;
    if (typeof v === 'string') {
      if (k && SKIP_KEYS.has(k)) return;
      out.push(v);
      return;
    }
    if (Array.isArray(v)) {
      v.forEach((item) => walk(item, k));
      return;
    }
    if (typeof v === 'object') {
      for (const [nk, val] of Object.entries(v)) walk(val, nk);
    }
  };
  walk(node, key);
  return out;
}

test('every post has the fields a static host and an answer engine need', () => {
  expect(posts.length).toBeGreaterThan(0);
  for (const post of posts) {
    expect(post.slug).toMatch(/^[a-z0-9-]+$/);
    expect(post.title.length).toBeGreaterThan(8);
    expect(post.description.length).toBeGreaterThan(20);
    expect(post.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(post.blocks.some((b) => b.type === 'p')).toBe(true);
    expect(getPost(post.slug)).toBe(post);
    expect(postHref(post.slug)).toBe(`/blog/${post.slug}.html`);
    expect(postUrl(post.slug)).toBe(`${SITE_ORIGIN}/blog/${post.slug}.html`);
  }
  expect(getPost('not-a-post')).toBeNull();
  expect(writing.indexHref).toBe('/blog.html');
});

test('visible writing copy keeps the site voice', () => {
  for (const s of visibleStrings({ writing, posts })) {
    expect(s).not.toMatch(FORBIDDEN);
  }
});

test('new glossary answers keep the site voice and stay in the FAQ SSOT', () => {
  const questions = faq.items.map((item) => item.q);
  for (const q of NEW_FAQ) {
    expect(questions).toContain(q);
    const item = faq.items.find((i) => i.q === q);
    expect(item.a).not.toMatch(FORBIDDEN);
    expect(q).not.toMatch(FORBIDDEN);
  }
});

test('motherboard copy is labeled vision and does not round up to a SKU', () => {
  const item = faq.items.find((i) => i.q === 'What is an AI motherboard?');
  expect(item.a.toLowerCase()).toContain('vision');
  const fleets = getPost('fleets');
  const motherboard = fleets.blocks
    .filter((b) => b.type === 'p' || b.type === 'caption' || b.type === 'figure')
    .map((b) => `${b.text ?? ''} ${b.caption ?? ''} ${b.alt ?? ''}`)
    .join(' ')
    .toLowerCase();
  expect(motherboard).toContain('vision');
});

test('blog copy does not paste ladder figures', () => {
  const blob = visibleStrings(posts).join('\n');
  expect(blob).not.toMatch(/\d+\.\d+\s*x/i);
  expect(blob).not.toMatch(/\d+\.\d+\s*%/);
  expect(blob.toLowerCase()).not.toContain('tok/s');
});

test('the fleets article still has a live headroom reading to render', () => {
  const h = headroom(real.rows ?? []);
  expect(h).not.toBeNull();
  expect(h.atlas).toBeGreaterThan(0);
  expect(h.atlas).toBeGreaterThan(h.baseline);
  expect(signed(h.atlas)).toMatch(/^[+-]\d+\.\d%$/);
  expect(getPost('fleets').blocks.some((b) => b.type === 'scale')).toBe(true);
});

test('committed sitemap names every public writing URL and skips diligence', () => {
  const xml = readFileSync(resolve(site, 'static/sitemap.xml'), 'utf8');
  for (const path of sitemapPaths()) {
    const loc =
      path === '/' ? `${SITE_ORIGIN}/` : `${SITE_ORIGIN}${path}`;
    expect(xml).toContain(`<loc>${loc}</loc>`);
  }
  expect(xml).not.toContain('diligence');
});

test('committed llms.txt lists the writing URLs from the same SSOT', () => {
  const txt = readFileSync(resolve(site, 'static/llms.txt'), 'utf8');
  expect(txt).toContain('## Writing');
  for (const post of posts) {
    expect(txt).toContain(post.title);
    expect(txt).toContain(postUrl(post.slug));
  }
  expect(txt).toContain(indexUrl);
});
