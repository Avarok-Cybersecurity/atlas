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
// PRIVACY. Fleet data lives in memory for the life of the tab, no fleet value
// ever reaches a URL, and the prerendered page contains no fleet data at all.
// Two things are written to storage, both by other modules and neither by this
// one: the browser-pairing token (`protocol.js`) and the operator's own
// preferences (`profile.js`). The latter includes the fingerprints of machines
// the operator selected themselves — see that file for why that is the one
// fleet value worth persisting. Names, addresses, vitals and alerts are not.

import { AgentClient } from './client.svelte.js';

/** Longest display string we will render. */
import { DETAIL_MAX, MAX_NODES, alert, ingestNode, sanitize, vitals } from './ingest.js';
import { readDecision, readExchange, readExchangeAt } from './pairing.js';

/** Poll cadence while waiting for an agent to appear, and its ceiling. */
const PROBE_START_MS = 1200;
const PROBE_MAX_MS = 8000;
const PROBE_FACTOR = 1.4;

/** A node unheard from for this long is shown as stale rather than removed. */
const STALE_AFTER_MS = 15_000;

export { sanitize } from './ingest.js';

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
   * Whether the machine running this browser can host a rank itself.
   *
   * Null until the agent has said, which is not the same as false: a page that
   * assumed false while connecting would flash a control-only banner at every
   * operator on a Spark. Callers must treat null as "not yet known".
   */
  get localCanLaunch() {
    return this.local ? this.local.canLaunch : null;
  }

  /**
   * True only once the agent has confirmed this machine cannot run a model.
   *
   * This is an ordinary, supported way to run — a laptop driving headless
   * boxes — so it drives explanation, never an error.
   */
  get controlOnly() {
    return this.localCanLaunch === false;
  }

  /** Why this machine cannot run a model, as the agent explained it. */
  get controlOnlyReason() {
    return this.local?.cannotLaunchReason ?? '';
  }

  /**
   * Machines other than this one that could hold a rank.
   *
   * What a control-only browser is actually offering to drive, and the number
   * that decides whether its empty state should teach pairing or launching.
   */
  get remoteLaunchable() {
    return this.launchable.filter((n) => !n.isLocal);
  }

  /**
   * Connect, and keep trying while no agent is there.
   *
   * `watch` is false for the nav indicator, which wants one attempt and no
   * retry loop — a marketing page must not poll loopback forever.
   */
  async start({ watch = true } = {}) {
    if (this.#started) {
      // A later caller asking to WATCH must get watching, even though the
      // session is already up. FleetPill is mounted by <Nav/> INSIDE the
      // control page, and child effects run first, so on a machine with a
      // stored token the pill's `start({watch: false})` always won the race.
      // The page's own `start({watch: true})` then short-circuited here, and a
      // paired operator whose agent was down read "Watching for it — this page
      // will continue on its own" while nothing was watching at all.
      if (watch) {
        await (this.#starting ?? Promise.resolve());
        // Idempotent: #scheduleProbe clears any existing timer first, so the
        // second caller cannot stack a second backoff loop on the first.
        if (this.mode === 'no_agent') this.#scheduleProbe();
      }
      return this.#starting ?? Promise.resolve();
    }
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
  /**
   * Run the pairing exchange. This establishes NO trust.
   *
   * `ok` means the two machines derived the same key and there are words to
   * compare — not that anything is pinned. `confirm` is what trusts.
   */
  async pair(nodeId, code) {
    const res = await this.agent.pairPeer(nodeId, code);
    if (!res.ok) return { ok: false, detail: res.message };
    return readExchange(res.reply);
  }

  /**
   * Run the ceremony against an address the operator typed.
   *
   * Returns the identity that answered as well as the words, because nothing
   * was discovered: this reply is the first statement of who is at that
   * address, and the operator needs it before they decide.
   */
  async pairAt(target, code) {
    const res = await this.agent.pairPeerAt(target, code);
    if (!res.ok) return { ok: false, node: null, name: '', address: '', detail: res.message };
    return readExchangeAt(res.reply);
  }

  /**
   * Trust a peer after a human compared the words.
   *
   * `allowControl` is the second, separate decision the ceremony asks:
   * whether the newly trusted machine may drive THIS one (launch and stop
   * models here). Trust without it is one-way — this machine can still see
   * and drive the peer wherever the peer has granted control.
   */
  async confirm(nodeId, allowControl = false) {
    const res = await this.agent.confirmPairing(nodeId, allowControl);
    if (!res.ok) return { ok: false, detail: res.message };
    return readDecision(res.reply, true);
  }

  /**
   * Refuse a completed exchange.
   *
   * Distinct from `unpair`: nothing was written, so this discards rather than
   * removes. The difference matters when it fails — a failed reject leaves no
   * trust behind, whereas a failed unpair leaves a machine trusted.
   */
  async reject(nodeId) {
    const res = await this.agent.rejectPairing(nodeId);
    if (!res.ok) return { ok: false, detail: res.message };
    // Not an unconditional `ok: true`. This asked the agent to leave the peer
    // untrusted, so the answer worth reporting is whether it IS untrusted — an
    // affirmative here would tell the operator their refusal took effect
    // without ever having read what the agent said about it.
    return readDecision(res.reply, false);
  }

  async unpair(nodeId) {
    const res = await this.agent.unpairPeer(nodeId);
    if (!res.ok) return { ok: false, detail: res.message };
    // `unpair_peer` answers a decision now, not a pairing result.
    return readDecision(res.reply, false);
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
      // Same filter and cap as the fallback path below. This is the branch a
      // modern agent actually takes, so it was the one that mattered: an
      // invalid id put a null in `nodes`, and the `local` getter reads
      // `n.isLocal` on every entry.
      this.nodes = res.reply.nodes.map(ingestNode).filter(Boolean).slice(0, MAX_NODES);
      this.watching = true;
      return;
    }
    // An agent too old to know the fleet verbs is not an error: it is a
    // single-node agent, and the page should show this machine and say so.
    const list = await this.agent.listNodes();
    this.nodes = Array.isArray(list.reply?.nodes)
      ? list.reply.nodes.map(ingestNode).filter(Boolean).slice(0, MAX_NODES)
      : [];
    this.watching = false;
  }

  #onEvent(msg) {
    // The socket died. Until this existed nothing observed it, so a page that
    // had reached 'live' stayed 'live' through an agent restart — showing a
    // fleet that could no longer change, with no probe scheduled, because
    // probes only ever started from 'no_agent'.
    if (msg?.type === 'agent_closed') {
      if (this.mode === 'live') {
        this.mode = 'reconnecting';
        this.watching = false;
        this.#probeDelay = PROBE_START_MS;
        this.#scheduleProbe();
      }
      return;
    }
    if (msg?.type !== 'fleet_event') return;
    const ev = msg.event;
    const next = this.nodes.slice();
    const at = (id) => next.findIndex((n) => n.id === id);

    switch (ev?.change) {
      case 'node_changed': {
        const node = ingestNode(ev.node);
        // A descriptor this page cannot make sense of is dropped rather than
        // rendered as a blank card.
        if (!node) break;
        const i = at(node.id);
        if (i >= 0) next[i] = node;
        // This is the flood path: an update for a node already known is
        // always accepted, but a NEW id past the cap is refused. Beacons are
        // unauthenticated, so without this a stream of fresh ids grows the
        // list until the page stops responding.
        else if (next.length < MAX_NODES) next.push(node);
        break;
      }
      case 'node_gone': {
        const i = at(ev.node);
        if (i >= 0) next.splice(i, 1);
        break;
      }
      case 'vitals': {
        const i = at(ev.node);
        // Through the same validator as the snapshot. This path assigned the
        // wire value raw, so a hostile `{state:'reading',value:'x'}` reached
        // `format(value)` in VitalTile and threw — the exact crash class the
        // ingestion whitelist exists to prevent, on the one path that fires
        // while somebody is watching.
        if (i >= 0) next[i] = { ...next[i], vitals: vitals(ev.vitals), lastSeen: Date.now() };
        break;
      }
      case 'alert_raised': {
        const i = at(ev.node);
        if (i >= 0) {
          // Was built inline with `?? 'unknown'` / `?? 'warning'`, so neither
          // field went through the whitelist the snapshot path uses: a
          // non-string `kind` reached `.replaceAll` and a chosen `severity`
          // reached `class="al-{...}"`.
          const raised = alert(ev.alert);
          const kept = next[i].alerts.filter((a) => a.kind !== raised.kind);
          next[i] = { ...next[i], alerts: [...kept, raised].slice(-8) };
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
      // Reached from 'no_agent' and from 'reconnecting' alike: both mean "no
      // usable socket", and the only difference is whether the page ever had
      // one.
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
