// SPDX-License-Identifier: AGPL-3.0-only

// Prerendered in its no-agent state. That is what most visitors get, what
// crawlers get, and what Lighthouse measures — and it means the shipped HTML
// contains no fleet data at all, so the privacy property is structural rather
// than something to maintain.
export const prerender = true;

// Emit `control/index.html` rather than `control.html`, so `/control` resolves
// through the web server's directory index.
//
// This has to be SvelteKit's own mechanism rather than copying the file after
// the build. The page computes its asset base from its own URL
// (`new URL(".", location)`), so a document built for `/control.html` and then
// served at `/control/` resolves every asset against `/control/` and 404s all
// thirty of them. It still *renders* — the HTML is prerendered and the CSS
// inline — so it looks correct and never hydrates, which for a page whose whole
// job is talking to a local agent is worse than serving the wrong page.
export const trailingSlash = 'always';
