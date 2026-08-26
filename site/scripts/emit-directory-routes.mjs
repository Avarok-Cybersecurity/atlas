// SPDX-License-Identifier: AGPL-3.0-only

// Make every prerendered sub-page reachable at its own clean URL.
//
// adapter-static writes a route as `<name>.html`. Whether that answers a
// request for `/<name>` is entirely the web server's business, and ours does
// not try the `.html` extension — it falls through to the SPA fallback, so
// `/control` served the *homepage*. Same status code, wrong page, and nothing
// in the build could tell.
//
// So each page is also written to `<name>/index.html`. The root already proves
// this server resolves a directory to its index, which is what makes `/control`
// work. The flat `<name>.html` stays where it is: it is what works today, and a
// deploy target nobody here can test against is the wrong place to find out a
// replacement was wrong.
//
// Cheap enough not to matter — these documents inline their stylesheet, so a
// duplicate is tens of kilobytes, and the service worker caches by URL anyway.

import { copyFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const BUILD = 'build';
/** Already the index of its own directory; copying it would be a loop. */
const ROOT = 'index.html';

const pages = readdirSync(BUILD).filter(
  (f) => f.endsWith('.html') && f !== ROOT && statSync(join(BUILD, f)).isFile(),
);

for (const page of pages) {
  const name = page.slice(0, -'.html'.length);
  const dir = join(BUILD, name);
  mkdirSync(dir, { recursive: true });
  copyFileSync(join(BUILD, page), join(dir, ROOT));
  console.log(`[routes] /${name} -> ${name}/index.html`);
}

if (pages.length === 0) console.log('[routes] no sub-pages to mirror');
