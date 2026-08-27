// SPDX-License-Identifier: AGPL-3.0-only

// The rules behind the "Show me how" join guide on /control.
//
// Pure and plain `.js` for the house reason: the test runner cannot import a
// file holding runes, so anything living beside them is untestable by
// construction. The surface (`JoinGuide.svelte`) renders whatever these
// functions decide and decides nothing itself; the shared credential lives in
// `joinstate.svelte.js`.
//
// The countdown here is deadline-based on purpose. A decrementing counter in a
// background tab gets throttled and lies — the operator walks to another
// machine, comes back, and the counter claims minutes that are gone. Computing
// against `mintedAtMs + expiresInS` cannot drift.

import { bestAddress, joinCommand } from './joincommand.js';

/**
 * How long the guide watches for the new machine before escalating: the
 * watching line changes and the troubleshooting causes auto-open. Named and
 * exported so field feedback retunes one constant.
 */
export const STALL_AFTER_MS = 120_000;

/** Countdown remaining below this turns amber and grows the warning copy. */
export const WARN_UNDER_S = 60;

/**
 * The port a joining machine must reach on this one. The agent assumes it when
 * a pairing target names none; the troubleshooting copy names it because a
 * firewall on exactly this port is the most likely real failure.
 */
export const JOIN_PORT = 34334;

/**
 * Map the agent's `mint_join_code` reply to the join shape the UI holds, or
 * null when there is nothing usable in it. One author for the mapping that
 * used to be inlined in `session.svelte.js`.
 *
 * @param {{code?: string, addresses?: string[], expires_in_s?: number}|null|undefined} reply
 * @returns {{code: string, addresses: string[], expiresInS: number}|null}
 */
export function normalizeJoinReply(reply) {
  const code = typeof reply?.code === 'string' ? reply.code.trim() : '';
  if (!code) return null;
  return {
    code,
    addresses: Array.isArray(reply.addresses) ? reply.addresses : [],
    expiresInS: Number.isFinite(reply.expires_in_s) ? reply.expires_in_s : 0
  };
}

/**
 * What the card can offer for a mint outcome.
 *
 * 'command'    — a complete one-liner exists; show it.
 * 'no_address' — a code was minted but no machine could dial this one; the
 *                loopback explanation, then the by-hand path.
 * 'manual'     — no code at all (agent refused, older agent, transport
 *                error); the by-hand path with its own lead.
 *
 * Delegates to `joinCommand`, so the fail-closed empty-string contract — never
 * half a credential — keeps its single author.
 *
 * @param {{code?: string, addresses?: string[]}|null|undefined} join
 * @returns {'command'|'no_address'|'manual'}
 */
export function offerKind(join) {
  const code = typeof join?.code === 'string' ? join.code.trim() : '';
  if (!code) return 'manual';
  return joinCommand(join) ? 'command' : 'no_address';
}

/**
 * When the minted code stops working, in epoch milliseconds.
 *
 * @param {number} mintedAtMs
 * @param {number} expiresInS
 * @returns {number}
 */
export function deadlineMs(mintedAtMs, expiresInS) {
  return mintedAtMs + expiresInS * 1000;
}

/**
 * What the countdown should show right now.
 *
 * Ceiling, not floor: the label reads 0:01 until the deadline has actually
 * passed, and `expired` flips exactly when no time is left. Clamped at zero so
 * a late tick can never render a negative label.
 *
 * @param {number} deadline epoch ms, from `deadlineMs`
 * @param {number} nowMs epoch ms
 * @returns {{seconds: number, label: string, warning: boolean, expired: boolean}}
 */
export function remaining(deadline, nowMs) {
  const leftMs = deadline - nowMs;
  const seconds = Math.max(0, Math.ceil(leftMs / 1000));
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return {
    seconds,
    label: `${m}:${String(s).padStart(2, '0')}`,
    // From the real time left, NOT from `seconds` above. `seconds` is ceiled so
    // the label never reads 0:00 while the code still works, which means at
    // 59.999s left it is 60 — and a threshold built on it would stay silent
    // for a whole second after the window is genuinely inside the last minute.
    warning: leftMs < WARN_UNDER_S * 1000,
    expired: leftMs <= 0
  };
}

/**
 * Whether the watch has gone on long enough to escalate.
 *
 * @param {number} commandShownAtMs when the command first rendered
 * @param {number} nowMs
 * @returns {boolean}
 */
export function stalled(commandShownAtMs, nowMs) {
  return nowMs - commandShownAtMs >= STALL_AFTER_MS;
}

/**
 * The code in read-aloud groups of four — display only; the command itself
 * stays ungrouped, because the grouped form is not what the far machine takes.
 *
 * @param {string} code
 * @returns {string}
 */
export function groupedCode(code) {
  const s = typeof code === 'string' ? code.trim() : '';
  return s.replace(/(.{4})(?=.)/g, '$1 ');
}

/**
 * The carry-note tail — `CODE@HOST` — or ''.
 *
 * Extracted from `joinCommand`'s output rather than rebuilt, so it can never
 * disagree with the line on screen and never renders half a credential: if
 * there is no complete command, there is no tail.
 *
 * @param {{code?: string, addresses?: string[]}|null|undefined} join
 * @returns {string}
 */
export function shortForm(join) {
  const cmd = joinCommand(join);
  if (!cmd) return '';
  return cmd.slice(cmd.lastIndexOf(' ') + 1);
}

/**
 * The address the far machine must reach, for the troubleshooting copy.
 * Same source as the command (`bestAddress`), so the two can never name
 * different hosts. Null when the command state is unreachable anyway.
 *
 * @param {{addresses?: string[]}|null|undefined} join
 * @returns {string|null}
 */
export function dialHost(join) {
  return bestAddress(join?.addresses);
}
