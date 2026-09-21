// SPDX-License-Identifier: AGPL-3.0-only
//
// The cost section reads record keys that a Rust producer writes. Nothing in
// the site build checks that the two agree: a renamed key on the producer side
// would leave every cost chart in its "not measured" state FOREVER, rendering
// perfectly and saying nothing, and no test here would go red.
//
// So this reads the producer's own source and proves each string the page
// depends on is emitted there. It is a source-text check, not a round trip —
// it cannot prove the VALUE is what the page thinks it is, only that the NAME
// is the producer's — and that limit is written down here rather than implied.
//
// Reading a repo file from a site test is established practice
// (light-text-contrast.test.js reads ../../../web-shared/avarok-tokens.css).
import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { KEY, MIN_POWER_SAMPLES, RUN_KEY } from './cost.js';

const path = (rel) => fileURLToPath(new URL(rel, import.meta.url));
const ENERGY = path('../../../crates/avarok-plugin/src/hardware/energy.rs');
const SAMPLER = path('../../../crates/avarok-plugin/src/hardware/energy_sampler.rs');

// The site is built from a checkout of the whole repo; if that ever stops
// being true this must fail loudly rather than skip and report green.
describe('the producer source is where the page expects it', () => {
  test('both energy modules are readable from the site tree', () => {
    expect(existsSync(ENERGY)).toBe(true);
    expect(existsSync(SAMPLER)).toBe(true);
  });
});

const energySrc = readFileSync(ENERGY, 'utf8');
const samplerSrc = readFileSync(SAMPLER, 'utf8');

describe('every key cost.js reads is one the producer writes', () => {
  test.each(Object.entries(KEY))('per-window key %s = %s is emitted by EnergyWindow::metrics', (_name, key) => {
    expect(energySrc).toContain(`"${key}"`);
  });

  test.each(Object.entries(RUN_KEY))('run-level key %s = %s is emitted', (_name, key) => {
    expect(`${energySrc}${samplerSrc}`).toContain(`"${key}"`);
  });

  test('the rail is in every name, so no reader sets one beside a discrete card', () => {
    for (const key of [...Object.values(KEY), ...Object.values(RUN_KEY)]) expect(key).toContain('gpu_rail');
  });

  test('the prefix the sweep applies is the c{C}_ the page builds', () => {
    // concurrency.rs: r.instrument_metrics(&format!("c{c}_"), …)
    const conc = readFileSync(path('../../../crates/avarok-plugin/src/benchmarks/concurrency.rs'), 'utf8');
    expect(conc).toContain('instrument_metrics(&format!("c{c}_")');
    expect(conc).toContain('c{c}_aggregate_tok_s');
  });
});

describe('the trust rule is calibrated against the producer cadence, not guessed', () => {
  test('the sampler cadence is a pinned constant and the floor is stated against it', () => {
    const m = /pub const SAMPLE_PERIOD_MS: u64 = (\d+);/.exec(energySrc);
    expect(m).not.toBeNull();
    const periodMs = Number(m[1]);
    // MIN_POWER_SAMPLES readings at that cadence is the least evidence the
    // page will count. If the producer ever slows the sampler this says what
    // the floor then means, rather than letting it silently become minutes.
    expect((MIN_POWER_SAMPLES * periodMs) / 1000).toBeLessThanOrEqual(5);
  });

  test('the producer integrates rather than spot-sampling, which is what J/token assumes', () => {
    expect(energySrc).toContain('power.draw.average');
    expect(energySrc).toMatch(/pub fn integrate\(/);
  });

  test('SW power cap is documented as this box NORMAL state — which is why it does not disqualify', () => {
    expect(energySrc).toContain('SW Power Cap is the NORMAL state');
  });
});

describe('the check can fail', () => {
  test('NEGATIVE CONTROL: a key the producer does not emit is not found', () => {
    // The shape the SPEC proposed before the producer landed. It is NOT in the
    // source, which is exactly what this file exists to notice.
    expect(energySrc).not.toContain('"c{C}_completion_tokens"');
    expect(energySrc).not.toContain('"energy_sample_period_ms"');
  });
});
