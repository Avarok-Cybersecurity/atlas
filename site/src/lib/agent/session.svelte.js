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
import { fleet } from './fleet.svelte.js';
import * as Placement from './placement.js';
import { looksLikeToken, storeToken } from './protocol.js';

class LaunchSession {
  /** The one client every card shares. */
  agent = new AgentClient();

  /** Recipe whose dialog is open, or null. */
  openRecipe = $state(null);

  /**
   * What the dialog is doing.
   * 'connecting' | 'guide' | 'pairing' | 'placement' | 'settings' | 'running'
   * | 'failed'
   */
  phase = $state('connecting');

  /**
   * Where the launch is headed, once chosen.
   *
   * Null means this machine, which is both the default and the only answer a
   * single-candidate fleet has. Set only by an explicit choice, so nothing
   * silently retargets a launch the operator already reviewed.
   */
  target = $state(null);

  /** The placement decision, recomputed as the fleet fills in. */
  placement = $state(null);

  /**
   * An outstanding invitation for another machine, when one has been minted.
   *
   * `{ code, addresses, expiresInS }`, or null when this agent cannot take
   * members — which is a normal thing to be, not an error.
   */
  join = $state(null);

  /** Detail for the current phase — an error message, usually. */
  detail = $state('');

  /** Endpoint of a launch that started. */
  endpoint = $state('');

  /** Whether the last connection attempt is still in flight. */
  busy = $state(false);

  /** Open the dialog for a recipe and try to reach the agent. */
  async open(recipeId) {
    this.openRecipe = recipeId;
    this.target = null;
    this.placement = null;
    this.join = null;
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
      this.#choosePlacement();
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

  /**
   * Decide whether to ask where this should run.
   *
   * Skipped whenever there is nothing to ask — one candidate, or a multi-node
   * recipe, which is a cluster plan rather than a placement question. A
   * chooser with a single option is a nag, and the fleet being unknown (this
   * agent may not carry the fleet verbs at all) is exactly that case.
   */
  #choosePlacement() {
    const recipe = this.agent.recipes.find((r) => r.id === this.openRecipe) ?? null;
    const d = Placement.decide(fleet.nodes, recipe, fleet.localCanLaunch);
    this.placement = d;

    if (d.kind === 'ask' || d.kind === 'none') {
      // 'none' is not a dead end: it is the moment to offer the machine that
      // would fix it. Sending the operator to a settings form for a machine
      // that cannot run the recipe would be the older, worse answer.
      this.phase = 'placement';
      if (d.kind === 'none') this.mintJoin();
      return;
    }
    // 'only' still records the target, so the settings step and the review
    // that follows name the machine rather than assuming this one.
    this.target = d.kind === 'only' && !d.target.isLocal ? d.target : null;
    this.phase = 'settings';
  }

  /**
   * Ask this agent to open a join window, for the onboarding panel.
   *
   * Failure is quiet and leaves `join` null: an agent that cannot take members
   * is a normal thing to be, and the panel falls back to telling the operator
   * how to do it by hand.
   */
  async mintJoin() {
    this.join = null;
    const res = await this.agent.mintJoinCode();
    if (res?.ok && res.reply?.code) {
      this.join = {
        code: res.reply.code,
        addresses: res.reply.addresses ?? [],
        expiresInS: res.reply.expires_in_s ?? 0
      };
    }
  }

  /** Send this launch to a specific machine. */
  chooseTarget(node) {
    this.target = node?.isLocal ? null : node;
    this.phase = 'settings';
  }
}

/** The page's launch session. */
export const launch = new LaunchSession();
