// SPDX-License-Identifier: AGPL-3.0-only

// Fully static output: every route is rendered at build time and served as
// plain files from an nginx docroot. No SSR, no runtime, no origin logic.
export const prerender = true;
export const ssr = true;
export const trailingSlash = 'always';
