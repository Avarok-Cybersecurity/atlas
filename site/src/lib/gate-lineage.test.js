// SPDX-License-Identifier: AGPL-3.0-only

import { expect, test } from 'bun:test';
import { assignTrendPredecessors, trendEdges } from './gate-lineage.js';

const rec = (git_sha, recorded_at, target_model = 'model-a') => ({
  git_sha,
  recorded_at,
  target_model
});

test('connects each point to the nearest earlier comparable ancestor', () => {
  const ancestors = new Set(['a>b', 'a>c', 'b>c']);
  const records = [rec('a', 1), rec('b', 2), rec('c', 3)];

  assignTrendPredecessors(records, (a, b) => a === b || ancestors.has(`${a}>${b}`));

  expect(trendEdges(records)).toEqual([
    [records[0], records[1]],
    [records[1], records[2]]
  ]);
});

test('does not draw an edge between divergent branch receipts', () => {
  const ancestors = new Set(['base>left', 'base>right']);
  const records = [rec('base', 1), rec('left', 2), rec('right', 3)];

  assignTrendPredecessors(records, (a, b) => a === b || ancestors.has(`${a}>${b}`));

  expect(trendEdges(records)).toEqual([
    [records[0], records[1]],
    [records[0], records[2]]
  ]);
});

test('resumes an older branch instead of joining the most recent divergence', () => {
  const ancestors = new Set(['base>left1', 'base>right', 'base>left2', 'left1>left2']);
  const records = [rec('base', 1), rec('left1', 2), rec('right', 3), rec('left2', 4)];

  assignTrendPredecessors(records, (a, b) => a === b || ancestors.has(`${a}>${b}`));

  expect(trendEdges(records)).toEqual([
    [records[0], records[1]],
    [records[0], records[2]],
    [records[1], records[3]]
  ]);
});

test('keeps different models and unknown ancestry disconnected', () => {
  const records = [rec('a', 1), rec('b', 2, 'model-b'), rec('missing', 3)];

  assignTrendPredecessors(records, () => false);

  expect(trendEdges(records)).toEqual([]);
});

test('does not connect ancestor commits measured with different instruments', () => {
  const a = { ...rec('a', 1), hardware: { gpu: 'GB10', driver: '580' }, params: { osl: '320' } };
  const b = { ...rec('b', 2), hardware: { gpu: 'GB10', driver: '595' }, params: { osl: '320' } };
  const c = { ...rec('c', 3), hardware: { gpu: 'GB10', driver: '580' }, params: { osl: '512' } };

  assignTrendPredecessors([a, b, c], () => true);

  expect(trendEdges([a, b, c])).toEqual([]);
});

test('uses stable machine identity across changing Spark hostnames', () => {
  const a = {
    ...rec('a', 1),
    hardware: { gpu: 'GB10', driver: '580' },
    machine_id: 'machine-1',
    perf_class: 'gb10@spark-256a'
  };
  const b = {
    ...rec('b', 2),
    hardware: { gpu: 'GB10', driver: '580' },
    machine_id: 'machine-1',
    perf_class: 'gb10@spark-43fa'
  };
  const otherMachine = {
    ...rec('c', 3),
    hardware: { gpu: 'GB10', driver: '580' },
    machine_id: 'machine-2',
    perf_class: 'gb10@spark-43fa'
  };

  assignTrendPredecessors([a, b, otherMachine], () => true);

  expect(trendEdges([a, b, otherMachine])).toEqual([[a, b]]);
});

test('fails closed between machine-id and legacy hostname identities', () => {
  const legacy = { ...rec('a', 1), perf_class: 'gb10@spark-256a' };
  const identified = {
    ...rec('b', 2),
    machine_id: 'machine-1',
    perf_class: 'gb10@spark-256a'
  };

  assignTrendPredecessors([legacy, identified], () => true);

  expect(trendEdges([legacy, identified])).toEqual([]);
});

test('parameter key order does not split an otherwise identical instrument', () => {
  const a = { ...rec('a', 1), params: { isls: '512', osl: '320' } };
  const b = { ...rec('b', 2), params: { osl: '320', isls: '512' } };

  assignTrendPredecessors([a, b], () => true);

  expect(trendEdges([a, b])).toEqual([[a, b]]);
});

test('records without generated lineage produce no invented edges', () => {
  const records = [rec('a', 1), rec('b', 2)];

  expect(trendEdges(records)).toEqual([]);
});
