// SPDX-License-Identifier: AGPL-3.0-only

// Prerendered like the rest of the site. The deck needs no server and no query
// string — deep-linking to a slide uses the hash, which never reaches the
// prerenderer, so `prerender = true` and `#7` coexist.
export const prerender = true;
