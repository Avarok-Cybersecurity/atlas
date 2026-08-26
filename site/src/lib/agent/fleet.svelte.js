// SPDX-License-Identifier: AGPL-3.0-only

// One fleet session for the whole app: the /control page and the nav indicator
// share it, so there is exactly one socket to the local agent.
//
// UNTRUSTED INPUT POLICY. Hostnames, agent versions, alert detail and address
// strings all originate from an unauthenticated multicast beacon or from a peer,
// and this is a public website rendering them. Rules, applied once at ingestion
// rather than at every render site:
//
//   1. Svelte text interpolation only. Never {@html}, never into href/src/style.
//   2. Control characters, ANSI escapes and bidi overrides are stripped. The
//      bidi ones matter: without them a hostname can visually reverse the
//      fingerprint printed next to it.
//   3. Everything is length-capped here, not by CSS. An unbounded string is a
//      denial of service against the layout.
//   4. Hostnames are display-only. Every keyed operation — the node map,
//      selection, pairing, launching — uses the fingerprint, because Sparks
//      ship with colliding names like spark-256a.
//
// PRIVACY. Fleet data lives in memory for the life of the tab. Nothing is
// written to storage except the existing browser-pairing token, no fleet value
// ever reaches a URL, and the prerendered page contains no fleet data at all.

import { AgentClient } from './client.svelte.js';

/** Longest display string we will render. */
const NAME_MAX = 63;
/** Longest free-text detail we will render. */
const DETAIL_MAX = 500;

/** Poll cadence while waiting for an agent to appear, and its ceiling. */
const PROBE_START_MS = 1200;
const PROBE_MAX_MS = 8000;
const PROBE_FACTOR = 1.4;

/** A node unheard from for this long is shown as stale rather than removed. */
const STALE_AFTER_MS = 15_000;

/**
 * Strip anything that could rewrite the interface, then cap the length.
 *
 * @param {unknown} raw
 * @param {number} max
 * @returns {string}
 */
export function sanitize(raw, max = NAME_MAX) {
  if (typeof raw !== 'string') return '';
  let out = '';
  for (const ch of raw) {
    const c = ch.codePointAt(0) ?? 0;
    // C0, DEL and C1 controls.
    if (c < 0x20 || (c >= 0x7f && c <= 0x9f)) continue;
    // Bidi overrides and isolates: a name must not be able to reorder the
    // fingerprint rendered beside it.
    if ((c >= 0x202a && c <= 0x202e) || (c >= 0x2066 && c <= 0x2069)) continue;
    out += ch;
    if (out.length >= max) break;
  }
  return out.trim();
}

/**
 * Normalise a node descriptor from the wire into something safe to render.
 *
 * @param {object} raw
 * @returns {object}
 */
function ingestNode(raw) {
  const addresses = Array.isArray(raw?.addresses) ? raw.addresses : [];
  return {
    id: typeof raw?.id === 'string' ? raw.id : '',
    name: sanitize(raw?.name) || 'unnamed',
    isLocal: raw?.is_local === true,
    pairing: raw?.pairing ?? 'discovered',
    addresses: addresses.slice(0, 8).map((a) => ({
      iface: sanitize(a?.iface, 32),
      addr: sanitize(a?.addr, 64),
      class: a?.class ?? 'ethernet',
      speedMbps: Number.isFinite(a?.speed_mbps) ? a.speed_mbps : null,
      rdma: a?.rdma === true
    })),
    canLaunch: raw?.launchability?.can_launch === true,
    cannotLaunchReason: sanitize(raw?.launchability?.reason, DETAIL_MAX),
    agentVersion: sanitize(raw?.agent_version, 32),
    accelerator: sanitize(raw?.accelerator, 32),
    vitals: raw?.vitals ?? null,
    alerts: (Array.isArray(raw?.alerts) ? raw.alerts : []).slice(0, 8).map((a) => ({
      kind: a?.kind ?? 'unknown',
      severity: a?.severity ?? 'warning',
      detail: sanitize(a?.detail, DETAIL_MAX)
    })),
    running: raw?.running ? sanitize(raw.running, 64) : null,
    lastSeen: Date.now()
  };
}

/** The node's best address, which is what a collective would use. */
export function preferredAddress(node) {
  const usable = node.addresses.filter((a) => a.class !== 'virtual' && a.class !== 'loopback');
  if (usable.length === 0) return null;
  // Unverified ranks below everything known: a candidate, never a preference.
  const rank = { infini_band: 5, roce: 4, ethernet: 3, wireless: 2, unverified: 1 };
  return usable.slice().sort((a, b) => {
    const r = (rank[b.class] ?? 0) - (rank[a.class] ?? 0);
    return r !== 0 ? r : (b.speedMbps ?? 0) - (a.speedMbps ?? 0);
  })[0];
}

/** Whether a link class should carry a visible warning. */
export function linkWarns(cls) {
  // Unverified is missing information, not a slow link. Warning about it would
  // be inventing a problem.
  return cls !== 'roce' && cls !== 'infini_band' && cls !== 'unverified';
}

class FleetSession {
  /** The one client the whole app shares. */
  agent = new AgentClient();

  /**
   * What the page is showing.
   * 'idle' | 'probing' | 'no_agent' | 'browser_unpaired' | 'live' | 'reconnecting'
   */
  mode = $state('idle');

  /** Detail for the current mode. */
  detail = $state('');

  /**
   * The fleet, local node first.
   *
   * Plain `$state`, not `$state.raw`: the list is reassigned wholesale on every
   * update, and a raw field did not re-render when it was filled in after the
   * mode had already flipped to 'live' — the page connected and then showed an
   * empty fleet.
   */
  nodes = $state([]);

  /** Whether a watch is open and vitals are flowing. */
  watching = $state(false);

  #stopEvents = null;
  #probeTimer = null;
  #probeDelay = PROBE_START_MS;
  #started = false;
  /**
   * The in-flight start, so concurrent callers await one connect rather than
   * racing two. The nav indicator and the control page share this session, and
   * an effect that re-runs would otherwise tear down a connection that another
   * caller is still opening.
   */
  #starting = null;

  /** This machine, when the agent has told us about it. */
  get local() {
    return this.nodes.find((n) => n.isLocal) ?? null;
  }

  /** Peers, paired first, then discovered, each group by name. */
  get peers() {
    const order = { paired: 0, pairing: 1, unreachable: 2, discovered: 3 };
    return this.nodes
      .filter((n) => !n.isLocal)
      .slice()
      .sort((a, b) => {
        const r = (order[a.pairing] ?? 9) - (order[b.pairing] ?? 9);
        return r !== 0 ? r : a.name.localeCompare(b.name);
      });
  }

  /** Every alert in the fleet, worst first. */
  get alerts() {
    const weight = { critical: 0, warning: 1, info: 2 };
    return this.nodes
      .flatMap((n) => n.alerts.map((a) => ({ ...a, node: n.id, nodeName: n.name })))
      .sort((a, b) => (weight[a.severity] ?? 9) - (weight[b.severity] ?? 9));
  }

  /** Worst severity present, or null. */
  get worstSeverity() {
    return this.alerts[0]?.severity ?? null;
  }

  /** Nodes that could take a rank in a cluster launch. */
  get launchable() {
    return this.nodes.filter((n) => n.canLaunch && (n.isLocal || n.pairing === 'paired'));
  }

  /**
   * Connect, and keep trying while no agent is there.
   *
   * `watch` is false for the nav indicator, which wants one attempt and no
   * retry loop — a marketing page must not poll loopback forever.
   */
  async start({ watch = true } = {}) {
    if (this.#started) return this.#starting ?? Promise.resolve();
    this.#started = true;
    this.#starting = (async () => {
      await this.#connect();
      if (watch && this.mode === 'no_agent') this.#scheduleProbe();
    })();
    try {
      await this.#starting;
    } finally {
      this.#starting = null;
    }
    return undefined;
  }

  /** Stop everything. Used when the control page unmounts. */
  stop() {
    this.#started = false;
    clearTimeout(this.#probeTimer);
    this.#probeTimer = null;
    this.#stopEvents?.();
    this.#stopEvents = null;
    this.watching = false;
  }

  /** Retry because the user asked. Shows progress, unlike the background probe. */
  async retry() {
    clearTimeout(this.#probeTimer);
    this.mode = 'probing';
    await this.#connect();
    if (this.mode === 'no_agent') this.#scheduleProbe();
  }

  /** Pair a discovered peer with a code read off that machine. */
  async pair(nodeId, code) {
    const res = await this.agent.pairPeer(nodeId, code);
    if (!res.ok) return { ok: false, detail: res.message };
    const reply = res.reply;
    return {
      ok: reply.paired === true,
      verification: reply.verification ?? null,
      detail: sanitize(reply.detail, DETAIL_MAX)
    };
  }

  /** Drop trust in a peer. */
  async unpair(nodeId) {
    const res = await this.agent.unpairPeer(nodeId);
    return { ok: res.ok, detail: res.ok ? '' : res.message };
  }

  // ---- internals ---------------------------------------------------------

  async #connect() {
    const ok = await this.agent.connect();
    if (!ok) {
      this.mode = this.agent.phase === 'unpaired' ? 'browser_unpaired' : 'no_agent';
      this.detail = this.agent.message ?? '';
      return;
    }
    // 'live' is announced as soon as the agent answers, not after the first
    // fleet load: leaving it until later showed the "no agent here yet" panel
    // to someone whose agent had just answered. The list filling in a moment
    // later is fine because `nodes` is reactive — which it was not, and that
    // was the actual bug behind an empty fleet on a working connection.
    this.mode = 'live';
    this.detail = '';
    this.#probeDelay = PROBE_START_MS;
    await this.#openWatch();
  }

  async #openWatch() {
    this.#stopEvents?.();
    this.#stopEvents = this.agent.onEvent((msg) => this.#onEvent(msg));

    const res = await this.agent.watchFleet(true);
    if (res.ok && Array.isArray(res.reply?.nodes)) {
      this.nodes = res.reply.nodes.map(ingestNode);
      this.watching = true;
      return;
    }
    // An agent too old to know the fleet verbs is not an error: it is a
    // single-node agent, and the page should show this machine and say so.
    const list = await this.agent.listNodes();
    this.nodes = Array.isArray(list.reply?.nodes) ? list.reply.nodes.map(ingestNode) : [];
    this.watching = false;
  }

  #onEvent(msg) {
    if (msg?.type !== 'fleet_event') return;
    const ev = msg.event;
    const next = this.nodes.slice();
    const at = (id) => next.findIndex((n) => n.id === id);

    switch (ev?.change) {
      case 'node_changed': {
        const node = ingestNode(ev.node);
        const i = at(node.id);
        if (i >= 0) next[i] = node;
        else next.push(node);
        break;
      }
      case 'node_gone': {
        const i = at(ev.node);
        if (i >= 0) next.splice(i, 1);
        break;
      }
      case 'vitals': {
        const i = at(ev.node);
        if (i >= 0) next[i] = { ...next[i], vitals: ev.vitals, lastSeen: Date.now() };
        break;
      }
      case 'alert_raised': {
        const i = at(ev.node);
        if (i >= 0) {
          const alert = {
            kind: ev.alert?.kind ?? 'unknown',
            severity: ev.alert?.severity ?? 'warning',
            detail: sanitize(ev.alert?.detail, DETAIL_MAX)
          };
          const kept = next[i].alerts.filter((a) => a.kind !== alert.kind);
          next[i] = { ...next[i], alerts: [...kept, alert] };
        }
        break;
      }
      case 'alert_cleared': {
        const i = at(ev.node);
        if (i >= 0) {
          next[i] = { ...next[i], alerts: next[i].alerts.filter((a) => a.kind !== ev.kind) };
        }
        break;
      }
      default:
        // An agent newer than this page may send a change we do not know.
        // Ignoring it is correct; crashing on it is not.
        return;
    }
    this.nodes = next;
  }

  #scheduleProbe() {
    clearTimeout(this.#probeTimer);
    this.#probeTimer = setTimeout(async () => {
      if (!this.#started) return;
      // Silent: a poll the user did not ask for must not repaint what they are
      // reading. It may only move the page forward, when an agent answers.
      const ok = await this.agent.connect();
      if (ok) {
        this.mode = 'live';
        this.#probeDelay = PROBE_START_MS;
        await this.#openWatch();
        return;
      }
      this.#probeDelay = Math.min(this.#probeDelay * PROBE_FACTOR, PROBE_MAX_MS);
      this.#scheduleProbe();
    }, this.#probeDelay);
  }
}

/** Whether a node's last sample is old enough to show as stale. */
export function isStale(node, now = Date.now()) {
  return now - node.lastSeen > STALE_AFTER_MS;
}

/** The app's fleet session. */
export const fleet = new FleetSession();
