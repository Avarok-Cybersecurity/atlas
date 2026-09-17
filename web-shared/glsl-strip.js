/**
 * Strip comments from GLSL before it is bundled.
 *
 * The shader is imported with `?raw`, so every byte of it — including the
 * commentary that explains the geometry, the falloff and the luma clamp —
 * ships to the browser inside a string literal. Minifiers do not touch string
 * contents, so nothing else removes it.
 *
 * Measured on this shader: 7,254 bytes of source becomes 2,481, and the
 * bundled, minified, gzipped field drops from 4.90 KB to 2.80 KB. That is 43%
 * of the payload of a component whose entire argument is that it costs 2.5 KB
 * instead of 151 KB, so it is not a rounding error — it was most of the gap
 * between what the field notes measured and what this repo was shipping.
 *
 * The comments stay in the file, which is where they are useful.
 */

/** @param {string} src @returns {string} */
export function stripGlsl(src) {
  let out = src.replace(/\/\*[\s\S]*?\*\//g, '');   // block comments
  out = out.replace(/\/\/[^\n]*/g, '');              // line comments
  // Drop blank lines and trailing whitespace. Newlines BETWEEN code lines are
  // kept: GLSL preprocessor directives (`#version`, `#define`) are
  // line-oriented, and joining them would not compile.
  return out.split('\n').map((l) => l.trimEnd()).filter((l) => l.trim()).join('\n') + '\n';
}

/**
 * Vite plugin. `enforce: 'pre'` so this handles `.glsl?raw` before Vite's own
 * raw loader does.
 */
export function glslStrip() {
  return {
    name: 'atlas-glsl-strip',
    enforce: 'pre',
    async load(id) {
      const m = /^(.*\.glsl)\?raw$/.exec(id);
      if (!m) return null;
      const { readFile } = await import('node:fs/promises');
      const src = await readFile(m[1], 'utf8');
      const out = stripGlsl(src);
      // A shader that lost its #version directive compiles to nothing and the
      // field silently falls back to the CSS dots — exactly the failure this
      // whole component is built to make loud. Fail the build instead.
      if (!out.startsWith('#version ')) {
        this.error(`${m[1]}: stripping removed the #version directive`);
      }
      if (!/\bvoid\s+main\s*\(/.test(out)) {
        this.error(`${m[1]}: stripping removed main()`);
      }
      return `export default ${JSON.stringify(out)};`;
    }
  };
}
