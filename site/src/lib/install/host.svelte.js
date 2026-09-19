// SPDX-License-Identifier: AGPL-3.0-only

// The detected host OS, shared by every surface that prints an install line.
//
// One piece of state, set once. The alternative — each component sniffing the
// user-agent itself — is how the hero and the control page end up disagreeing
// about which machine the visitor is on, which is worse than either answer.
//
// It starts at `unknown`, which resolves to the shell command: exactly what
// the prerendered HTML contains, so the page does not visibly rewrite itself
// on hydration for the majority of visitors.

import { detectCurrentOs, installCommandFor } from './platform.js';
import { installerUrl, powershellInstallerUrl } from '$lib/data.js';

const URLS = { shellUrl: installerUrl, powershellUrl: powershellInstallerUrl };

const host = $state({ os: /** @type {import('./platform.js').Os} */ ('unknown') });

/** Detect the host once, from the browser. Safe to call more than once. */
export function detectHost() {
  host.os = detectCurrentOs();
}

/** The install one-liner for this visitor. Reactive. */
export function currentInstall() {
  return installCommandFor(host.os, URLS);
}
