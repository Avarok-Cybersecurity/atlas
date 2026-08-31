// SPDX-License-Identifier: AGPL-3.0-only

// preprocess.js — the hook that lets `.md` be a post format.
//
// Registered as a Svelte MARKUP PREPROCESSOR rather than a raw Vite plugin, so
// vite-plugin-svelte (already in the build) claims `.md` through
// `extensions: ['.svelte', '.md']` and we inherit SSR/client dual compilation,
// HMR and sourcemaps instead of reimplementing them around `svelte.compile`.
//
// The transform is in memory. No `.svelte` file is ever written for a markdown
// post: one article, one file, one source of truth.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { imageSize } from 'image-size';
import { authors, tags } from '../content.js';
import { compileMarkdown } from './compile.js';
import { makeHighlighter } from './highlight.js';

/**
 * @param {object} [opts]
 * @param {string} [opts.staticDir] where `/images/...` resolves from
 */
export function atlasMarkdown({ staticDir = 'static' } = {}) {
  let highlighter = null;

  /**
   * Intrinsic size of a post image, so the compiler can write width/height and
   * the browser reserves the box before the bytes arrive. A missing file is a
   * BUILD failure: a 404 image is invisible in dev and obvious in production,
   * which is the worst possible ordering.
   */
  const measure = (src) => {
    const file = join(staticDir, src.replace(/^\//, ''));
    let buf;
    try {
      buf = readFileSync(file);
    } catch {
      throw new Error(`image not found: ${src} (looked in ${file})`);
    }
    try {
      const { width, height } = imageSize(buf);
      return width && height ? { width, height } : null;
    } catch {
      // SVG without an intrinsic size is legitimate — it scales. Everything
      // else that cannot be measured is a corrupt file, and the build of a
      // corrupt image is not this hook's business to diagnose.
      return null;
    }
  };

  return {
    name: 'atlas-markdown',
    async markup({ content, filename }) {
      if (!filename?.endsWith('.md')) return undefined; // .svelte posts untouched
      if (!highlighter) highlighter = await makeHighlighter();
      const { svelte } = compileMarkdown(content, {
        filename,
        tags,
        authors,
        highlight: highlighter,
        measure
      });
      return { code: svelte };
    }
  };
}
