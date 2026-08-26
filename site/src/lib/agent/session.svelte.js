// SPDX-License-Identifier: AGPL-3.0-only

// One agent connection, and one launch dialog, for the whole page.
//
// Both of those are deliberate. Every recipe card used to construct its own
// AgentClient, which meant thirty clients and thirty potential sockets to the
// same agent. And every card rendered its own dialog *inside the card* — which
// broke it outright, because `.subcard:hover` applies a transform, and a
// transformed ancestor becomes the containing block for `position: fixed`
// descendants. The dialog positioned itself against the card instead of the
// viewport, then jumped when the hover ended: it appeared, flickered, and
// vanished.
//
// Hoisting the dialog to the page root is the fix. Hoisting the client is the
// tidy-up that belongs with it.

import { AgentClient } from './client.svelte.js';
import { looksLikeToken, storeToken } from './protocol.js';

class LaunchSession {
  /** The one client every card shares. */
  agent = new AgentClient();

  /** Recipe whose dialog is open, or null. */
  openRecipe = $state(null);

  /**
   * What the dialog is doing.
   * 'connecting' | 'guide' | 'pairing' | 'settings' | 'running' | 'failed'
   */
  phase = $state('connecting');

  /** Detail for the current phase — an error message, usually. */
  detail = $state('');

  /** Endpoint of a launch that started. */
  endpoint = $state('');

  /** Whether the last connection attempt is still in flight. */
  busy = $state(false);

  /** Open the dialog for a recipe and try to reach the agent. */
  async open(recipeId) {
    this.openRecipe = recipeId;
    this.detail = '';
    this.endpoint = '';
    await this.#connect();
  }

  /** Close the dialog. Never stops a running model — the agent owns that. */
  close() {
    this.openRecipe = null;
    this.detail = '';
  }

  /**
   * Retry because the user asked. User-initiated, so it shows progress.
   */
  async retry() {
    await this.#connect(undefined, { silent: false });
  }

  /**
   * Background poll while the "no agent yet" guide is on screen.
   *
   * Silent on purpose. This used to call retry(), which set phase back to
   * 'connecting' on every tick — so the dialog flipped between "Looking for
   * your agent" and "Run this on your own machine" once a second, swapping its
   * whole body each time. A poll the user did not ask for must not change what
   * they are reading; it may only move the dialog FORWARD, when an agent
   * actually answers.
   */
  async probe() {
    await this.#connect(undefined, { silent: true });
  }

  /** Submit a pairing token the user pasted. */
  async pair(token) {
    const trimmed = token.trim();
    if (!looksLikeToken(trimmed)) {
      this.detail = 'That does not look like a pairing token — it is 64 hexadecimal characters.';
      return false;
    }
    storeToken(trimmed);
    await this.#connect(trimmed);
    return this.phase !== 'pairing';
  }

  /** Note that a launch started, so the dialog can show its endpoint. */
  started(reply) {
    this.endpoint = reply?.endpoint ?? '';
    this.phase = 'running';
  }

  /**
   * @param {string} [token]
   * @param {{ silent?: boolean }} [opts] `silent` suppresses every visible
   *   effect of a FAILED attempt: no spinner, no phase change, no message.
   *   Success and "an agent answered but wants pairing" still advance, because
   *   those are the events the poll exists to catch.
   */
  async #connect(token, opts) {
    const silent = opts?.silent === true;
    if (!silent) {
      this.busy = true;
      this.phase = 'connecting';
    }
    const ok = await (token === undefined ? this.agent.connect() : this.agent.connect(token));
    if (!silent) this.busy = false;

    if (ok) {
      this.phase = 'settings';
      return;
    }
    if (this.agent.phase === 'unpaired') {
      this.phase = 'pairing';
      this.detail = this.agent.message;
      return;
    }
    // Still nothing there. A silent probe leaves the guide exactly as it is;
    // a transient error mid-poll is not worth tearing the dialog down over,
    // and the user can still press Try again to see it.
    if (silent) return;

    if (this.agent.phase === 'error') {
      this.phase = 'failed';
      this.detail = this.agent.message;
    } else {
      this.phase = 'guide';
      this.detail = '';
    }
  }
}

/** The page's launch session. */
export const launch = new LaunchSession();
