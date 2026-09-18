// SPDX-License-Identifier: AGPL-3.0-only
//
// The prompt pack is data (media-brief/shots.json) and a generated sheet
// (media-brief/PROMPTS.md). These tests keep the sheet current and keep the
// prompts inside what the research says works and what the brand allows.
import { expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { render, fullPrompt } from '../../../scripts/media/brief.mjs';
import { allClips } from './media.js';
import { pages } from './index.js';

const SITE_DIR = join(import.meta.dir, '..', '..', '..');
const brief = JSON.parse(readFileSync(join(SITE_DIR, 'media-brief', 'shots.json'), 'utf8'));
const ids = new Set(brief.shots.map((s) => s.id));
const words = (s) => s.trim().split(/\s+/).length;

test('PROMPTS.md is what brief.mjs renders from shots.json', () => {
  const sheet = readFileSync(join(SITE_DIR, 'media-brief', 'PROMPTS.md'), 'utf8').replace(/\r\n/g, '\n');
  expect(sheet === render(brief) ? 'current' : 'stale: run node scripts/media/brief.mjs').toBe('current');
});

test('shot ids are unique and every shot is complete', () => {
  expect(ids.size).toBe(brief.shots.length);
  for (const s of brief.shots) {
    expect(['text-to-image', 'image-to-image', 'image-to-video', 'text-to-video'], s.id).toContain(s.mode);
    for (const key of ['slot', 'aspect', 'prompt', 'keep', 'alt']) expect(typeof s[key], `${s.id}.${key}`).toBe('string');
    expect(s.alt.length, `${s.id} alt`).toBeGreaterThan(20);
    expect(brief.limits.image_aspect_ratios, `${s.id} aspect`).toContain(s.aspect);
  }
});

test('every source is an approved shot or a file that exists', () => {
  const missing = [];
  for (const s of brief.shots.filter((x) => x.source)) {
    for (const src of [].concat(s.source)) {
      if (ids.has(src)) continue;
      if (!existsSync(join(SITE_DIR, src))) missing.push(`${s.id}: ${src}`);
    }
  }
  expect(missing).toEqual([]);
});

test('an edit never takes more source images than the model accepts', () => {
  for (const s of brief.shots.filter((x) => x.mode === 'image-to-image')) {
    expect([].concat(s.source).length, s.id).toBeLessThanOrEqual(brief.limits.edit_source_images);
  }
});

test('video shots stay inside the duration limit and lead with motion, then one camera instruction', () => {
  for (const s of brief.shots.filter((x) => x.mode === 'image-to-video')) {
    expect(s.seconds, s.id).toBeLessThanOrEqual(brief.limits.video_seconds_max);
    // Short, because the model can see the still. Long prompts redescribe it.
    expect(words(s.prompt), `${s.id} is ${words(s.prompt)} words`).toBeLessThanOrEqual(40);
    expect((s.prompt.match(/\bcamera\b/gi) ?? []).length, `${s.id} camera instructions`).toBe(1);
    expect(s.prompt.toLowerCase().indexOf('camera'), `${s.id}: motion comes before the camera`).toBeGreaterThan(20);
  }
});

test('text to video shots carry the house style in words, stay short enough to steer, and move one camera', () => {
  const shots = brief.shots.filter((x) => x.mode === 'text-to-video');
  expect(shots.length).toBeGreaterThan(0);
  for (const s of shots) {
    expect(s.seconds, s.id).toBeLessThanOrEqual(brief.limits.video_seconds_max);
    expect(words(s.prompt), `${s.id} is ${words(s.prompt)} words`).toBeLessThanOrEqual(70);
    expect((s.prompt.match(/\bcamera\b/gi) ?? []).length, `${s.id} camera instructions`).toBe(1);
    expect(fullPrompt(brief, s).endsWith(brief.style.video_from_text), s.id).toBe(true);
  }
});

test('no prompt asks for resolution in words, a logo, or the product interface', () => {
  const banned = /\b(4k|8k|ultra[- ]?hd|hyper[- ]?detailed|avarok|atlas|dashboard|user interface|\bui\b)\b/i;
  const offenders = brief.shots.filter((s) => banned.test(s.prompt)).map((s) => s.id);
  // E04 places the recorded console poster on a monitor. It names no interface
  // and generates none: the screen content is the uploaded recording.
  expect(offenders).toEqual([]);
});

test('every still prompt carries the house style, so the set reads as one', () => {
  for (const s of brief.shots.filter((x) => x.mode === 'text-to-image' || x.mode === 'image-to-image')) {
    expect(fullPrompt(brief, s).endsWith(brief.style.still), s.id).toBe(true);
  }
});

test('a video shot fills a slot the site has, or names a new b-roll slot to add', () => {
  const slots = new Set(allClips.map((c) => c.name));
  for (const s of brief.shots.filter((x) => x.mode.endsWith('-to-video'))) {
    expect(slots.has(s.slot) || s.slot.startsWith('broll-'), `${s.id} -> ${s.slot}`).toBe(true);
  }
});

test('every page a still is meant for exists', () => {
  const registry = new Set(pages.map((p) => p.path));
  const dead = brief.shots.flatMap((s) => (s.pages ?? []).filter((p) => p.startsWith('/') && !registry.has(p)).map((p) => `${s.id}: ${p}`));
  expect(dead).toEqual([]);
});

test('the rules that keep generated media legal are written down', () => {
  expect(brief.never.length).toBeGreaterThanOrEqual(5);
  expect(brief.never.join(' ')).toMatch(/logo/i);
  expect(brief.never.join(' ')).toMatch(/UI/);
  expect(brief.sources.length).toBeGreaterThanOrEqual(2);
});
