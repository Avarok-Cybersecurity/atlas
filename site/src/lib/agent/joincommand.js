// SPDX-License-Identifier: AGPL-3.0-only

// The one line an operator pastes on the machine they are adding.
//
// Pure and separate because it is a *credential-bearing* string and the rules
// about it are worth stating once: which address it names, what happens when
// there is none, and that it never renders half an invitation. A command that
// is subtly wrong is worse than no command, because it fails on the far
// machine where the operator has the least context.

import { installerUrl } from '../data.js';

/**
 * Build the install-and-join one-liner.
 *
 * Returns an empty string when there is nothing valid to render — no code, or
 * no address the other machine could dial. Half a command would look
 * copy-pasteable and would not be.
 *
 * @param {{code: string, addresses: string[]}|null} join
 * @returns {string}
 */
export function joinCommand(join, grantControl = false) {
  const code = typeof join?.code === 'string' ? join.code.trim() : '';
  if (!code) return '';
  const host = bestAddress(join?.addresses);
  if (!host) return '';
  const base = `curl -fsSL ${installerUrl} | sh -s -- --join ${code}@${host}`;
  // The grant is a VISIBLE flag on the line the operator pastes, never an
  // implication of joining. It is also the only direction that does what
  // someone adding a GPU box actually wants: it is written into THAT machine's
  // pin of this fleet, by the person standing at that keyboard. The reverse —
  // this machine granting the new one control of itself — is a different
  // decision and is not what "add a machine I can run models on" means.
  return grantControl ? `${base} --grant-control` : base;
}

/**
 * The address another machine should dial, or null.
 *
 * The agent already orders these best-link-first and strips loopback, so this
 * takes the first usable one. It re-checks for loopback anyway: a command
 * naming 127.0.0.1 would install cleanly and then fail to pair, which is the
 * most confusing failure available here.
 *
 * @param {string[]|undefined} addresses
 * @returns {string|null}
 */
export function bestAddress(addresses) {
  const list = Array.isArray(addresses) ? addresses : [];
  for (const a of list) {
    const s = String(a ?? '').trim();
    if (!s) continue;
    if (s === '127.0.0.1' || s === '::1' || s.startsWith('127.')) continue;
    return s;
  }
  return null;
}
