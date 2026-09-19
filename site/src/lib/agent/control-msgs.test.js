// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as M from './control-msgs.js';

// The fingerprint used throughout: NodeId::from_bytes([0xab; 32]).
const NODE = 'ab'.repeat(32);

// Every fixture below is the output of serde_json::to_string over the Rust
// `ClientMsg` at protocol 4 — generated, not hand-written. If a builder stops
// matching one byte-for-byte, either the builder drifted or the protocol did,
// and both are worth a failing test.
describe('wire shapes are byte-exact against protocol 4 serde output', () => {
  const SETTINGS = { max_seq_len: 4096, kv_dtype: 'fp8' };

  const CASES = [
    [M.listRecipes(1, null), '{"type":"list_recipes","id":1,"on":null}'],
    [M.listRecipes(2, NODE), `{"type":"list_recipes","id":2,"on":"${NODE}"}`],
    [
      M.preview(3, 'qwen3.6-27b-fp8', SETTINGS, NODE),
      `{"type":"preview","id":3,"recipe":"qwen3.6-27b-fp8","settings":{"kv_dtype":"fp8","max_seq_len":4096},"on":"${NODE}"}`
    ],
    [
      M.preview(4, 'qwen3.6-27b-fp8', {}, null),
      '{"type":"preview","id":4,"recipe":"qwen3.6-27b-fp8","settings":{},"on":null}'
    ],
    [
      M.launch(5, 'qwen3.6-27b-fp8', SETTINGS, NODE),
      `{"type":"launch","id":5,"recipe":"qwen3.6-27b-fp8","settings":{"kv_dtype":"fp8","max_seq_len":4096},"on":"${NODE}"}`
    ],
    [
      M.stop(6, 'qwen3.6-27b-fp8', NODE),
      `{"type":"stop","id":6,"recipe":"qwen3.6-27b-fp8","on":"${NODE}"}`
    ],
    [M.status(7, null), '{"type":"status","id":7,"on":null}'],
    [M.status(8, NODE), `{"type":"status","id":8,"on":"${NODE}"}`],
    [
      M.launchStats(9, 'qwen3.6-27b-fp8', NODE),
      `{"type":"launch_stats","id":9,"recipe":"qwen3.6-27b-fp8","on":"${NODE}"}`
    ],
    [
      M.launchLogs(10, 'qwen3.6-27b-fp8', 200, NODE),
      `{"type":"launch_logs","id":10,"recipe":"qwen3.6-27b-fp8","lines":200,"on":"${NODE}"}`
    ],
    [M.mintJoinCode(11, false), '{"type":"mint_join_code","id":11,"allow_control":false}'],
    [M.mintJoinCode(12, true), '{"type":"mint_join_code","id":12,"allow_control":true}'],
    [
      M.confirmPairing(13, NODE, true),
      `{"type":"confirm_pairing","id":13,"node":"${NODE}","allow_control":true}`
    ],
    [
      M.confirmPairing(14, NODE, false),
      `{"type":"confirm_pairing","id":14,"node":"${NODE}","allow_control":false}`
    ]
  ];

  test.each(CASES)('%#', (built, wire) => {
    expect(JSON.stringify(built)).toBe(wire);
  });

  test('settings keys are emitted in BTreeMap order regardless of insertion order', () => {
    const reversed = M.preview(3, 'r1', { zeta: 1, alpha: 2 }, null);
    expect(JSON.stringify(reversed)).toContain('"settings":{"alpha":2,"zeta":1}');
  });
});

describe('`on` must be said: local is an explicit null, never an omission', () => {
  test('a forgotten target throws instead of silently going local', () => {
    expect(() => M.stop(1, 'r1')).toThrow(TypeError);
    expect(() => M.listRecipes(1)).toThrow(TypeError);
  });

  test('a malformed target throws instead of riding the wire', () => {
    for (const bad of ['dgx3', NODE.slice(0, 63), NODE.toUpperCase(), 42, {}, undefined]) {
      expect(() => M.status(1, bad)).toThrow(TypeError);
    }
  });
});

describe('allow_control must be said, not implied', () => {
  test('mint and confirm refuse anything but a literal boolean', () => {
    for (const bad of [undefined, null, 1, 'true', 0]) {
      expect(() => M.mintJoinCode(1, bad)).toThrow(TypeError);
      expect(() => M.confirmPairing(1, NODE, bad)).toThrow(TypeError);
    }
  });

  test('confirm refuses a missing or malformed node', () => {
    expect(() => M.confirmPairing(1, null, true)).toThrow(TypeError);
    expect(() => M.confirmPairing(1, 'dgx3', true)).toThrow(TypeError);
  });
});

describe('inputs the wire cannot carry are refused', () => {
  test('correlation ids are u32', () => {
    for (const bad of [-1, 1.5, 2 ** 32, '7', NaN, null]) {
      expect(() => M.status(bad, null)).toThrow(TypeError);
    }
  });

  test('recipe ids outside the protocol alphabet throw', () => {
    for (const bad of ['', 'UPPER', 'has space', '--flag'.repeat(0) + '-x/../y', 42, null, 'a'.repeat(65)]) {
      expect(() => M.stop(1, bad, null)).toThrow(TypeError);
    }
  });

  test('a setting that would vanish or corrupt in JSON throws instead', () => {
    // undefined disappears from JSON.stringify and NaN becomes null — either
    // way the agent would launch without the setting and nobody would know.
    expect(() => M.launch(1, 'r1', { a: undefined }, null)).toThrow(TypeError);
    expect(() => M.launch(1, 'r1', { a: NaN }, null)).toThrow(TypeError);
    expect(() => M.launch(1, 'r1', { a: {} }, null)).toThrow(TypeError);
    expect(() => M.launch(1, 'r1', [], null)).toThrow(TypeError);
    expect(() => M.launch(1, 'r1', null, null)).toThrow(TypeError);
  });

  test('log line counts are positive u32s', () => {
    for (const bad of [0, -5, 1.5, 2 ** 32, '200']) {
      expect(() => M.launchLogs(1, 'r1', bad, null)).toThrow(TypeError);
    }
  });
});
