import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';
import { stripGlsl } from '../../../web-shared/glsl-strip.js';

/**
 * The shader ships inside a string literal, so minifiers never touch its
 * comments and the build plugin is the only thing that removes them. These
 * assert the two properties that matter: it removes the bytes, and it does not
 * remove anything the compiler needs.
 */

test('removes block and line comments', () => {
  expect(stripGlsl('/* a */\nfloat x = 1.0; // b\n')).toBe('float x = 1.0;\n');
});

test('keeps line structure, because preprocessor directives are line-oriented', () => {
  // Joining these would not compile: #version must be its own first line.
  const out = stripGlsl('#version 300 es\n// c\n#define A 1\nvoid main(){}\n');
  expect(out.split('\n')[0]).toBe('#version 300 es');
  expect(out).toContain('\n#define A 1\n');
});

test('drops blank lines but never the first directive', () => {
  const out = stripGlsl('/* banner\n   spanning lines */\n\n\n#version 300 es\n\nvoid main(){}\n');
  expect(out.startsWith('#version 300 es')).toBe(true);
});

test('a multi-line block comment between statements does not weld them together', () => {
  const out = stripGlsl('float a=1.0;\n/* long\n   comment */\nfloat b=2.0;\n');
  expect(out).toBe('float a=1.0;\nfloat b=2.0;\n');
});

test('leaves code that contains no comments byte-identical apart from the trailing newline', () => {
  const src = '#version 300 es\nprecision highp float;\nvoid main(){}';
  expect(stripGlsl(src)).toBe(src + '\n');
});

test('the real shader survives: directives, uniforms, functions and main all intact', () => {
  const src = readFileSync(new URL('../../../web-shared/gl/chevron-field.glsl', import.meta.url), 'utf8');
  const out = stripGlsl(src);

  expect(out.startsWith('#version 300 es\n')).toBe(true);
  expect(out).toMatch(/\bvoid\s+main\s*\(/);
  expect(out).not.toContain('/*');
  expect(out).not.toContain('//');

  // Every uniform, constant and function the runtime binds by name, and the
  // luma clamp the contrast gate is derived from. A strip that ate any of
  // these would leave the field silently falling back to the CSS dots.
  for (const needed of [
    'uniform vec2  u_res', 'u_time', 'u_scroll', 'u_density',
    'u_c1', 'u_c2', 'u_c3u', 'u_c3l', 'u_ground',
    'float sdSeg(', 'float sdChevron(', 'float sdMark(', 'vec2 layer(',
    'col /= max(1.0, lum);', 'fragColor'
  ]) {
    expect(out, `stripping removed ${needed}`).toContain(needed);
  }

  // It has to actually be worth doing. The claim in the post is that comments
  // were most of the payload; below 40% that claim stops being true.
  expect(1 - out.length / src.length).toBeGreaterThan(0.4);
});
