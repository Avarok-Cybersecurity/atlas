// SPDX-License-Identifier: AGPL-3.0-only
//
// copy-katex.mjs — put KaTeX's stylesheet and fonts in the build output.
//
// Only posts that actually contain maths link this stylesheet (see the post
// route), so it is copied rather than imported: an import would bundle it into
// the global CSS and charge every page — including every Svelte post, which
// has no maths at all — roughly 23 KB of render-blocking stylesheet for a
// feature it does not use.
//
// Same-origin on purpose: KaTeX's own CDN would be a third-party connection,
// which is a Lighthouse deduction and a hole in any future CSP.

import { cp, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, '..', 'node_modules', 'katex', 'dist');
// Into `static/`, BEFORE the build, not into `build/` after it: the post route
// links this stylesheet, and the blog prerenders with `handleHttpError: 'fail'`
// — so a link whose target does not exist yet fails the build. Copying first
// makes it an ordinary static asset that vite emits and the prerenderer can
// see. It is generated, so `.gitignore` covers it.
const to = join(here, '..', 'static', 'katex');

await mkdir(to, { recursive: true });
await cp(join(from, 'katex.min.css'), join(to, 'katex.min.css'));
await cp(join(from, 'fonts'), join(to, 'fonts'), { recursive: true });
console.log(`katex: stylesheet + fonts -> ${to}`);
