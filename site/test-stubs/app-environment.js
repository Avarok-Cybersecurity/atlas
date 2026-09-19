// SPDX-License-Identifier: AGPL-3.0-only

// Stands in for SvelteKit's `$app/environment`, which vite supplies at build
// time and bun does not. `chat/state.svelte.js` imports it, so without this a
// test of that module fails with "Cannot find module" — an error about the
// harness wearing the costume of an error about the code.
//
// `browser` is FALSE on purpose, not for convenience: `bun test` has no DOM, so
// code guarding DOM access with `if (browser)` should skip. Claiming true would
// run those branches against globals that are not there and fail somewhere less
// obvious than here.
export const browser = false;
export const dev = true;
export const building = false;
export const version = 'test';
