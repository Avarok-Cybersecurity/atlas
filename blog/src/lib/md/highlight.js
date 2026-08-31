// SPDX-License-Identifier: AGPL-3.0-only

// highlight.js — build-time syntax highlighting and math rendering.
//
// Both tools run here, in the build, and neither reaches the browser: shiki
// emits static spans and KaTeX emits static HTML+MathML. `e2e/check-bundle.mjs`
// asserts that, because "it only runs at build time" is a claim that rots
// quietly the first time someone imports this module from a component.

import katex from 'katex';
import { createHighlighter } from 'shiki';
import { COLOR_REPLACEMENTS, LANGS, atlasTheme } from './atlas-shiki-theme.js';

let cached = null;

/**
 * One shiki instance for the whole build — loading the grammars is the
 * expensive part, and a per-post instance would pay it once per post.
 */
export async function makeHighlighter() {
  if (!cached) {
    cached = createHighlighter({ themes: [atlasTheme], langs: LANGS }).then((shiki) => ({
      /** @returns {string} HTML with `var(--token)` colours */
      code(code, lang) {
        const known = LANGS.includes(lang) ? lang : 'text';
        return shiki.codeToHtml(code, {
          lang: known,
          theme: 'atlas',
          colorReplacements: COLOR_REPLACEMENTS
        });
      }
    }));
  }
  return cached;
}

/**
 * Render TeX to HTML + MathML.
 *
 * `htmlAndMathml` rather than the smaller `html`: the HTML-only output is a
 * pile of positioned spans that a screen reader cannot read, and it costs the
 * accessibility score. `throwOnError` makes a mistyped formula fail the BUILD
 * rather than render a red error box into production.
 *
 * @param {string} tex
 * @param {boolean} displayMode
 */
export const math = (tex, displayMode) =>
  katex.renderToString(tex, { displayMode, output: 'htmlAndMathml', throwOnError: true, strict: 'ignore' });
