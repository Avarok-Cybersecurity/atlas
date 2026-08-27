// SPDX-License-Identifier: AGPL-3.0-only

// The one stats poller the bridge runs.
//
// Every scheduling rule lives in cadence.js, where it is tested: which nodes
// are polled, at which period, at which stagger offset, and how failures back
// off. Every reading rule lives in iostrip.js: the first-poll rate gate and
// the too-wide-window gate. This file is the thin runes wrapper the runner
// cannot import — it owns the interval, carries replies into `$state`, and
// nothing else.

import { due, failed, plan, polled } from './cadence.js';
import { gateRates } from './iostrip.js';
import * as S from './stats.js';

/**
 * How often the wrapper asks cadence.js what is due. Finer than the fastest
 * cadence so a 1s schedule is honoured within a fraction of its period; the
 * schedule itself never fires faster than `plan` allows.
 */
const TICK_MS = 250;

export class StatsPoller {
  /**
   * Latest per node:
   * `{reading, at, failure, via, decodeHist, promptHist}` keyed by node id.
   * `reading` has been through `gateRates`; `at` is arrival, for staleness.
   */
  byNode = $state({});

  #agent;
  #plan = [];
  /** id → {recipe, on} — what a poll for this node puts on the wire. */
  #meta = new Map();
  #states = {};
  #epoch = 0;
  #cadence = null;
  #seen = new Set();
  #inflight = new Set();
  #timer = null;

  constructor(agent) {
    this.#agent = agent;
  }

  /**
   * (Re)state the schedule.
   *
   * @param {{id: string, selected: boolean, running: boolean, recipe: string|null, on: string|null}[]} nodes
   * @param {number|null} cadenceMs one of `CADENCES`; null pauses everything
   */
  configure(nodes, cadenceMs) {
    this.#meta = new Map(nodes.map((n) => [n.id, { recipe: n.recipe, on: n.on }]));
    // A cadence change re-anchors the epoch so offsets stay meaningful;
    // fleet churn alone does not, or every arriving beacon would rephase
    // every poll.
    if (cadenceMs !== this.#cadence || this.#epoch === 0) this.#epoch = Date.now();
    this.#cadence = cadenceMs;
    this.#plan = plan(
      nodes.map(({ id, selected, running }) => ({ id, selected, running })),
      cadenceMs
    );
  }

  start() {
    if (this.#timer !== null) return;
    this.#timer = setInterval(() => this.#tick(), TICK_MS);
  }

  stop() {
    if (this.#timer !== null) clearInterval(this.#timer);
    this.#timer = null;
  }

  /** The Actions bar's Stats verb: one poll now, schedule untouched. */
  pollNow(id) {
    if (this.#meta.has(id)) this.#poll(id);
  }

  #tick() {
    for (const id of due(this.#plan, this.#states, Date.now(), this.#epoch)) {
      this.#poll(id);
    }
  }

  async #poll(id) {
    if (this.#inflight.has(id)) return;
    const meta = this.#meta.get(id);
    if (!meta?.recipe) return;
    this.#inflight.add(id);
    try {
      const res = await this.#agent.launchStats(meta.recipe, meta.on);
      const now = Date.now();
      const prev = this.byNode[id];
      if (res?.ok && res.reply?.stats) {
        const reading = gateRates(res.reply.stats, { firstPoll: !this.#seen.has(id) });
        this.#seen.add(id);
        this.#states = polled(this.#states, id, now);
        this.byNode = {
          ...this.byNode,
          [id]: {
            reading,
            at: now,
            failure: null,
            via: res.reply.via ?? null,
            decodeHist: S.push(prev?.decodeHist ?? [], reading.decode_tokens_per_s),
            promptHist: S.push(prev?.promptHist ?? [], reading.prompt_tokens_per_s)
          }
        };
      } else {
        this.#states = failed(this.#states, id, now);
        this.byNode = {
          ...this.byNode,
          [id]: {
            // The last reading is kept — the strip decides how to present a
            // launch that has stopped answering; erasing it here would turn
            // "not answering" into "never existed".
            reading: prev?.reading ?? null,
            at: prev?.at ?? null,
            failure: res?.message || 'no reply',
            via: prev?.via ?? null,
            decodeHist: S.push(prev?.decodeHist ?? [], null),
            promptHist: S.push(prev?.promptHist ?? [], null)
          }
        };
      }
    } finally {
      this.#inflight.delete(id);
    }
  }
}
