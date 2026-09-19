// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { crossesGroup, groupOf } from './route-groups.js';

const ORIGIN = 'https://atlascybernetics.ai';

test('every developer page, in both spellings, is one group', () => {
  for (const p of ['/engine', '/engine/', '/engine.html', '/control', '/diligence', '/diligence.html']) expect(groupOf(p)).toBe('developer');
});

test('a marketing page that merely starts with the same letters is not a developer page', () => {
  expect(groupOf('/platform/engine')).toBe('marketing');
  expect(groupOf('/engineering')).toBe('marketing');
  expect(groupOf('/')).toBe('marketing');
});

test('a link between the marketing pages and the developer pages crosses, in both directions', () => {
  expect(crossesGroup('/company', '/engine', ORIGIN)).toBe(true);
  expect(crossesGroup('/company', '/engine#run', ORIGIN)).toBe(true);
  expect(crossesGroup('/engine', '/company', ORIGIN)).toBe(true);
  expect(crossesGroup('/control', `${ORIGIN}/pricing`, ORIGIN)).toBe(true);
});

test('a link inside one group, to another site, or to a mailbox does not', () => {
  expect(crossesGroup('/company', '/pricing', ORIGIN)).toBe(false);
  expect(crossesGroup('/engine', '/diligence', ORIGIN)).toBe(false);
  expect(crossesGroup('/company', '#top', ORIGIN)).toBe(false);
  expect(crossesGroup('/company', 'https://github.com/engine', ORIGIN)).toBe(false);
  expect(crossesGroup('/company', 'mailto:someone@example.com', ORIGIN)).toBe(false);
});
