// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { CAPS, placeholder, placeholdersFor } from './placeholders.js';

const FLEET = { solo: false };
const SOLO = { solo: true };

describe('the registry itself', () => {
  test('every region honours its cap', () => {
    for (const region of Object.keys(CAPS)) {
      expect(placeholdersFor(region, FLEET).length).toBeLessThanOrEqual(CAPS[region]);
    }
  });

  test('every entry names its missing capability in a sentence, not a shrug', () => {
    for (const region of Object.keys(CAPS)) {
      for (const e of placeholdersFor(region, FLEET)) {
        expect(e.soon.startsWith('Coming soon — ')).toBe(true);
        expect(e.soon.length).toBeGreaterThan(30);
        expect(e.label.length).toBeGreaterThan(0);
      }
    }
  });

  test('the I/O strip promises nothing, and the forbidden verbs never appear', () => {
    // ISL and OSL were the strip's two placeholders until the agent learned to
    // emit `isl_mean`/`osl_mean`. Nothing in the strip is a promise now, and a
    // tile that went real must leave the registry — a "soon" chip on a field
    // that already works is worse than no chip.
    expect(placeholdersFor('iostrip', FLEET)).toEqual([]);
    // The closed-enum doctrine: no remote shell, exec, restart or reboot may
    // appear even as a promise.
    for (const region of Object.keys(CAPS)) {
      for (const e of placeholdersFor(region, FLEET)) {
        expect(e.id).not.toMatch(/shell|exec|restart|reboot/);
      }
    }
  });
});

describe('cap enforcement fails at the source of the creep', () => {
  test('registering one placeholder past a cap throws', () => {
    const bloated = [
      { id: 'a', region: 'dock', label: 'A', soon: 'Coming soon — x.' },
      { id: 'b', region: 'dock', label: 'B', soon: 'Coming soon — y.' }
    ];
    expect(() => placeholdersFor('dock', FLEET, bloated)).toThrow(RangeError);
  });

  test('an unknown region is refused', () => {
    expect(() => placeholdersFor('sidebar', FLEET)).toThrow(TypeError);
  });
});

describe('solo mode collapses the actions chips', () => {
  test('one soon chip carries the collapsed entries', () => {
    const chips = placeholdersFor('actions', SOLO);
    expect(chips.length).toBe(1);
    expect(chips[0].id).toBe('soon-menu');
    expect(chips[0].collapsed.map((e) => e.id)).toEqual(
      placeholdersFor('actions', FLEET).map((e) => e.id)
    );
  });

  test('other regions do not collapse — two dashed tiles are not a roadmap', () => {
    expect(placeholdersFor('iostrip', SOLO)).toEqual(placeholdersFor('iostrip', FLEET));
  });

  test('solo must be said, not assumed', () => {
    expect(() => placeholdersFor('actions', {})).toThrow(TypeError);
    expect(() => placeholdersFor('actions')).toThrow(TypeError);
  });
});

describe('popover lookup', () => {
  test('a registered id resolves to its sentence', () => {
    expect(placeholder('update-agent').soon.length).toBeGreaterThan(0);
  });

  test('an unknown id throws instead of opening an empty popover', () => {
    expect(() => placeholder('warp-drive')).toThrow(TypeError);
  });
});
