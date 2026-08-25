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
  storedToken
} from './protocol.js';

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
      this.#socket = null;
      if (this.phase === 'ready') this.phase = 'unavailable';
    });

    // The agent speaks first, so wait for its welcome before greeting back.
    const welcome = await this.#await('welcome');
    if (!welcome) {
      this.phase = 'unavailable';
      return false;
    }
    if (PROTOCOL_VERSION < welcome.protocol_min || PROTOCOL_VERSION > welcome.protocol_max) {
      this.phase = 'error';
      this.message = `This page speaks protocol ${PROTOCOL_VERSION}; your agent speaks ${welcome.protocol_min}–${welcome.protocol_max}. Update whichever is older.`;
      return false;
    }

    this.#send({ type: 'hello', protocol_version: PROTOCOL_VERSION, token });
    const ready = await this.#await('ready', 'error');
    if (!ready || ready.type === 'error') {
      this.phase = ready?.error?.code === 'not_paired' ? 'unpaired' : 'error';
      this.message = ready ? describeError(ready.error) : 'The agent closed the connection.';
      return false;
    }

    this.recipes = ready.recipes ?? [];
    this.schema = ready.schema ?? [];
    this.canLaunch = Boolean(ready.can_launch);
    this.canLaunchReason = ready.can_launch_reason ?? null;
    this.phase = 'ready';
    return true;
  }

  /** Render the command a launch would run, without running it. */
  async preview(recipe, settings = {}) {
    return this.#request({ type: 'preview', recipe, settings });
  }

  /** Start a recipe. */
  async launch(recipe, settings = {}) {
    return this.#request({ type: 'launch', recipe, settings });
  }

  /** Stop a recipe. */
  async stop(recipe) {
    return this.#request({ type: 'stop', recipe });
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
  }

  #await(...types) {
    return this.#waitFor(types.join('|'));
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
