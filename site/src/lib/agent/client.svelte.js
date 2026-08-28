// SPDX-License-Identifier: AGPL-3.0-only

// A connection to the local agent.
//
// Every method resolves rather than throwing: a page that cannot reach an agent
// is an ordinary state, not an exception. The Run button treats "no agent" as
// something to explain, not something to fail on.

import {
  AGENT_URL,
  PROTOCOL_VERSION,
  describeError,
  storedToken,
  versionAdvice
} from './protocol.js';
import * as msgs from './control-msgs.js';

/** How long to wait for an agent before deciding there is not one. */
const CONNECT_TIMEOUT_MS = 1500;

/** How long to wait for a reply once connected. */
const REPLY_TIMEOUT_MS = 20000;

export class AgentClient {
  /** 'idle' | 'connecting' | 'ready' | 'unavailable' | 'unpaired' | 'error' */
  phase = $state('idle');
  /** Recipes the agent can see, keyed by id. */
  recipes = $state([]);
  /** The settings schema the agent will validate against. */
  schema = $state([]);
  /** Whether the agent's machine can launch at all. */
  canLaunch = $state(false);
  /** Why it cannot, when it cannot. */
  canLaunchReason = $state(null);
  /** Human-readable detail for the current phase. */
  message = $state('');

  #socket = null;
  #nextId = 1;
  #pending = new Map();
  /** The in-flight connect, so a second caller joins it rather than racing. */
  #connecting = null;
  /**
   * Listeners for frames nobody asked for.
   *
   * The agent pushes `fleet_event` continuously once a watch is open, and
   * those frames match no pending request — `#onMessage` used to drop anything
   * uncorrelated on the floor, so a stream would have silently gone nowhere.
   */
  #listeners = new Set();

  /**
   * Listen for unsolicited frames. Returns a function that stops listening.
   *
   * @param {(msg: object) => void} fn
   * @returns {() => void}
   */
  onEvent(fn) {
    this.#listeners.add(fn);
    return () => this.#listeners.delete(fn);
  }

  /** Whether a recipe id is one this agent can actually run. */
  runnable(recipeId) {
    const r = this.recipes.find((x) => x.id === recipeId);
    return Boolean(r?.runnable) && this.canLaunch;
  }

  /**
   * Connect and complete the handshake.
   *
   * Resolves to true when the agent is ready. A missing agent leaves the phase
   * at 'unavailable'; a bad token leaves it at 'unpaired'. Both are expected.
   */
  async connect(token = storedToken()) {
    if (this.phase === 'ready') return true;
    // One connect at a time. Two are reachable in practice — the launch
    // dialog's "Try again" while a silent fleet probe is mid-flight — and they
    // do not merely race: both build a socket, the second overwrites `#socket`
    // while the orphan's message listener keeps feeding `#onMessage`, and both
    // wait on the string key 'welcome', so the second clobbers the first's
    // entry in `#pending` and one of them hangs until its timeout.
    if (this.#connecting) return this.#connecting;
    this.#connecting = this.#connect(token).finally(() => {
      this.#connecting = null;
    });
    return this.#connecting;
  }

  async #connect(token) {
    this.phase = 'connecting';
    this.message = '';

    let socket;
    try {
      socket = new WebSocket(AGENT_URL);
    } catch {
      // Safari refuses ws:// from an https page outright.
      this.phase = 'unavailable';
      return false;
    }

    const opened = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), CONNECT_TIMEOUT_MS);
      socket.addEventListener('open', () => {
        clearTimeout(timer);
        resolve(true);
      });
      socket.addEventListener('error', () => {
        clearTimeout(timer);
        resolve(false);
      });
    });

    if (!opened) {
      try {
        socket.close();
      } catch {
        /* already gone */
      }
      this.phase = 'unavailable';
      return false;
    }

    this.#socket = socket;
    socket.addEventListener('message', (ev) => this.#onMessage(ev));
    socket.addEventListener('close', () => {
      // Only if this is still the live socket. An abandoned one — a wedged
      // agent that accepted a connection and said nothing — can fire `close`
      // long after a later connect succeeded, and without this guard it would
      // null the NEW socket, flip a ready phase to 'unavailable', fail that
      // connection's pending waits and emit a spurious `agent_closed`.
      if (this.#socket !== socket) return;
      this.#socket = null;
      const wasReady = this.phase === 'ready';
      if (wasReady) this.phase = 'unavailable';

      // Fail everything in flight NOW rather than letting each request sit out
      // its full REPLY_TIMEOUT_MS. The socket is gone; no reply is coming, and
      // twenty seconds of a spinner is twenty seconds the operator spends
      // wondering whether the agent is slow or dead.
      const waiting = [...this.#pending.values()];
      this.#pending.clear();
      for (const resolve of waiting) resolve(null);

      // And say so. Nothing observed this before, so a page that had reached
      // 'live' stayed 'live' after the agent restarted — watching a fleet that
      // could no longer change, with no probe scheduled because probes only
      // start from 'no_agent'.
      if (wasReady) {
        for (const fn of this.#listeners) {
          try {
            fn({ type: 'agent_closed' });
          } catch {
            /* a listener that throws must not stop the others hearing it */
          }
        }
      }
    });

    // The agent speaks first, so wait for its welcome before greeting back.
    // Every failure below closes the socket. They used to return with it
    // open and its listeners attached, so a wedged agent — one that accepts a
    // connection and then says nothing — collected a fresh socket from the
    // probe loop every 1.2 to 8 seconds, none of which were ever released.
    const welcome = await this.#await('welcome');
    if (!welcome) {
      return this.#abandon(socket, 'unavailable');
    }
    if (PROTOCOL_VERSION < welcome.protocol_min || PROTOCOL_VERSION > welcome.protocol_max) {
      this.message = versionAdvice(
        PROTOCOL_VERSION,
        welcome.protocol_min,
        welcome.protocol_max
      ).message;
      return this.#abandon(socket, 'error');
    }

    this.#send({ type: 'hello', protocol_version: PROTOCOL_VERSION, token });
    const ready = await this.#await('ready', 'error');
    if (!ready || ready.type === 'error') {
      this.message = ready ? describeError(ready.error) : 'The agent closed the connection.';
      return this.#abandon(socket, ready?.error?.code === 'not_paired' ? 'unpaired' : 'error');
    }

    this.recipes = ready.recipes ?? [];
    this.schema = ready.schema ?? [];
    this.canLaunch = Boolean(ready.can_launch);
    this.canLaunchReason = ready.can_launch_reason ?? null;
    this.phase = 'ready';
    return true;
  }

  // The control verbs delegate every frame they put on the wire to
  // `control-msgs.js`, where the shapes are built pure and tested byte-exact
  // against protocol 4. This file is transport only.
  //
  // `on` defaults to null — this machine — and `allow_control` to false at
  // THIS layer only: existing call sites are all local and consentless, and
  // the builders still demand the value written out, so the default is said
  // exactly once, here, rather than implied at every call site.

  /** Render the command a launch would run, without running it. */
  async preview(recipe, settings = {}, on = null) {
    return this.#dispatch((id) => msgs.preview(id, recipe, settings, on));
  }

  /** Start a recipe. */
  async launch(recipe, settings = {}, on = null) {
    return this.#dispatch((id) => msgs.launch(id, recipe, settings, on));
  }

  /** Stop a recipe. */
  async stop(recipe, on = null) {
    return this.#dispatch((id) => msgs.stop(id, recipe, on));
  }

  /** What is running on a node right now. */
  async status(on = null) {
    return this.#dispatch((id) => msgs.status(id, on));
  }

  /** The recipe inventory of any node, not just this one. */
  async listRecipes(on = null) {
    return this.#dispatch((id) => msgs.listRecipes(id, on));
  }

  /** The fleet as the agent currently sees it. */
  listNodes() {
    return this.#request({ type: 'list_nodes' });
  }

  /**
   * Subscribe to fleet changes.
   *
   * `vitals` is dropped for a background tab: losing 1 Hz telemetry costs
   * nothing, while structural changes and alerts must keep arriving so the nav
   * indicator stays truthful.
   */
  watchFleet(vitals = true) {
    return this.#request({ type: 'watch_fleet', vitals });
  }

  /** Pair a discovered peer with a code read off that machine. */
  pairPeer(node, code) {
    return this.#request({ type: 'pair_peer', node, code });
  }
  /**
   * Pair with a machine at an address the operator typed.
   *
   * `target` is `host`, `host:port` or `[v6]:port`; the agent assumes the peer
   * port when none is given. Discovery is mDNS and therefore link-local, so
   * this is the only way to reach a machine across a router.
   */
  pairPeerAt(target, code) {
    return this.#request({ type: 'pair_peer_at', target, code });
  }

  /** Trust a peer whose exchange completed and whose words the operator accepted. */
  confirmPairing(node, allowControl = false) {
    return this.#dispatch((id) => msgs.confirmPairing(id, node, allowControl));
  }

  /** Discard a completed exchange. Nothing was written, so nothing is undone. */
  rejectPairing(node) {
    return this.#request({ type: 'reject_pairing', node });
  }

  /**
   * Open a window in which one new machine may join this fleet.
   *
   * Answers with the digits and this node's dialable addresses, which is
   * everything needed to build the command the operator pastes elsewhere.
   */
  mintJoinCode(allowControl = false) {
    return this.#dispatch((id) => msgs.mintJoinCode(id, allowControl));
  }

  /** Close an outstanding join window. */
  revokeJoinCode() {
    return this.#request({ type: 'revoke_join_code' });
  }

  /** Drop trust in a peer. */
  unpairPeer(node) {
    return this.#request({ type: 'unpair_peer', node });
  }

  /** Render each rank's command without running anything. */
  previewCluster(recipe, nodes, head, settings) {
    return this.#request({ type: 'preview_cluster', recipe, nodes, head, settings });
  }

  /** Ask every selected node to validate and reserve. Nothing starts. */
  prepareCluster(recipe, nodes, head, settings) {
    return this.#request({ type: 'prepare_cluster', recipe, nodes, head, settings });
  }

  /** Start every rank of a prepared cluster. */
  commitCluster(epoch) {
    return this.#request({ type: 'commit_cluster', epoch });
  }

  /** Abandon a prepare, releasing every reservation. */
  /** How a running launch is doing. */
  launchStats(recipe, on = null) {
    return this.#dispatch((id) => msgs.launchStats(id, recipe, on));
  }

  /** The tail of a launch's log. */
  launchLogs(recipe, lines = 200, on = null) {
    return this.#dispatch((id) => msgs.launchLogs(id, recipe, lines, on));
  }

  /** Stop every rank of the cluster this agent started. */
  stopCluster() {
    return this.#request({ type: 'stop_cluster' });
  }

  abortCluster(epoch) {
    return this.#request({ type: 'abort_cluster', epoch });
  }

  /** Close the connection. */
  dispose() {
    try {
      this.#socket?.close();
    } catch {
      /* already gone */
    }
    this.#socket = null;
    if (this.phase === 'ready') this.phase = 'idle';
  }

  #send(msg) {
    this.#socket?.send(JSON.stringify(msg));
  }

  async #request(msg) {
    if (this.phase !== 'ready') return { ok: false, message: 'No agent is connected.' };
    const id = this.#nextId++;
    this.#send({ ...msg, id });
    return this.#settle(id);
  }

  /**
   * Send a frame built by `control-msgs.js`, which receives the correlation
   * id because the builders emit it in wire order rather than appended.
   *
   * A builder throw is a caller bug — a malformed recipe id, a forgotten
   * target — but this class's contract is that every method resolves, so it
   * is reported the same way every other local failure is.
   */
  async #dispatch(build) {
    if (this.phase !== 'ready') return { ok: false, message: 'No agent is connected.' };
    const id = this.#nextId++;
    let frame;
    try {
      frame = build(id);
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : String(err) };
    }
    this.#send(frame);
    return this.#settle(id);
  }

  /** Wait for the reply to `id` and fold it into the resolve-only contract. */
  async #settle(id) {
    const reply = await this.#awaitId(id);
    if (!reply) return { ok: false, message: 'The agent did not reply.' };
    if (reply.type === 'error') return { ok: false, message: describeError(reply.error), error: reply.error };
    return { ok: true, reply };
  }

  #onMessage(ev) {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    for (const [key, resolve] of this.#pending) {
      const matchesId = typeof key === 'number' && msg.id === key;
      const matchesType = typeof key === 'string' && key.split('|').includes(msg.type);
      if (matchesId || matchesType) {
        this.#pending.delete(key);
        resolve(msg);
        return;
      }
    }
    // Nothing was waiting for it, so it is a pushed frame. A listener that
    // throws must not take down the socket handler with it.
    for (const fn of this.#listeners) {
      try {
        fn(msg);
      } catch {
        /* a broken listener is not the socket's problem */
      }
    }
  }

  #await(...types) {
    return this.#waitFor(types.join('|'));
  }

  /**
   * Give up on a half-built connection, closing it.
   *
   * Always returns false, so a caller can `return this.#abandon(...)` and not
   * be able to forget the close — which is how three of these paths leaked.
   *
   * @param {WebSocket} socket
   * @param {string} phase
   * @returns {false}
   */
  #abandon(socket, phase) {
    this.phase = phase;
    if (this.#socket === socket) this.#socket = null;
    try {
      socket.close();
    } catch {
      /* already gone */
    }
    return false;
  }

  #awaitId(id) {
    return this.#waitFor(id);
  }

  #waitFor(key) {
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.#pending.delete(key);
        resolve(null);
      }, REPLY_TIMEOUT_MS);
      this.#pending.set(key, (msg) => {
        clearTimeout(timer);
        resolve(msg);
      });
    });
  }
}
