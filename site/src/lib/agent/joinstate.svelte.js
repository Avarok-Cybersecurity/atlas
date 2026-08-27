// SPDX-License-Identifier: AGPL-3.0-only

// The one owner of the join code currently on offer.
//
// A singleton because two surfaces show it — the launch dialog's "add a machine
// that can run models" panel and the control page's join guide — and a code is
// a ONE-USE credential. Two independent stores would each mint their own, so
// the operator could be looking at a command on one screen that was already
// invalidated by the other, with no way to tell which was live.
//
// Only the rune-holding shell lives here. Every rule about the value is in
// `joinwindow.js`, which is plain `.js` and therefore testable — this file
// cannot be imported by `bun test` at all.

import { normalizeJoinReply } from './joinwindow.js';

/** @typedef {{code: string, addresses: string[], expiresInS: number, mintedAtMs: number}} Offer */

class JoinState {
  /** @type {Offer|null} */
  current = $state(null);

  /**
   * Ask the agent to open a join window.
   *
   * Quiet on failure, leaving `current` null: an agent that cannot take on new
   * machines is a normal thing to be, and the surfaces have a designed state
   * for it. Returns whether an offer is now live so a caller can branch.
   */
  async mint(client) {
    const res = await client?.mintJoinCode?.();
    const offer = res?.ok ? normalizeJoinReply(res.reply) : null;
    // `mintedAtMs` is stamped HERE, not taken from the agent: the countdown is
    // rendered against this browser's clock, so it must be measured on it too.
    // Mixing the two makes the timer wrong by the clock skew between machines.
    this.current = offer ? { ...offer, mintedAtMs: Date.now() } : null;
    return this.current !== null;
  }

  /**
   * Withdraw the offer.
   *
   * Cleared locally even if the agent's reply is lost. The code is single-use
   * and short-lived; continuing to show one the operator has explicitly
   * cancelled is worse than briefly disagreeing with the agent about a
   * credential that is about to die anyway.
   */
  async revoke(client) {
    this.current = null;
    await client?.revokeJoinCode?.().catch?.(() => {});
  }
}

export const joinState = new JoinState();
