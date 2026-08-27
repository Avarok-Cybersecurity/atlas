// SPDX-License-Identifier: AGPL-3.0-only

// Turning wire data into something safe to render.
//
// Pure and separate from `fleet.svelte.js` for the reason the rest of this
// directory is: the session file holds runes and cannot be imported by a test
// runner, so anything living in it is untestable by construction. Every field
// below arrives over an UNAUTHENTICATED beacon path — anyone on the LAN can
// send it — and several of them reach a `class` attribute or a string method,
// so "untestable" was not an acceptable place for them to be.

const NAME_MAX = 63;
/** Longest free-text detail we will render. */
export const DETAIL_MAX = 500;

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
 * The enumerations the agent can legitimately send.
 *
 * Whitelists rather than sanitisation, because every one of these reaches a
 * `class` attribute or a string method. `sanitize()` would make them harmless
 * to render and still let an attacker pick the class name, and it would still
 * hand `.replaceAll` a number. A value off the list is not a display problem,
 * it is a message this agent did not send.
 *
 * Mirrors the serde `snake_case` spelling of PairingState, Severity and
 * LinkClass in atlasctl-protocol.
 */
/**
 * How many nodes this page will hold.
 *
 * Beacons are unauthenticated, so anyone on the LAN can announce new ids as
 * fast as they like. Without a cap that is an unbounded array of NodeCards and
 * a layout that stops responding — a denial of service that needs no exploit,
 * only patience. Far above any real fleet.
 */
export const MAX_NODES = 64;

const PAIRING = ['discovered', 'pairing', 'paired', 'unreachable'];
const SEVERITY = ['info', 'warning', 'critical'];
const LINK_CLASS = [
  'roce',
  'infini_band',
  'ethernet',
  'wireless',
  'virtual',
  'loopback',
  'unverified'
];

/**
 * One of `allowed`, or `fallback`.
 *
 * The fallback is explicit rather than "the first entry" because the safe
 * default differs per field: an unrecognised pairing state is `discovered`
 * (assume nothing), but an unrecognised severity is `warning` — quietly
 * downgrading an alert this page cannot classify to `info` is the wrong way to
 * be wrong.
 */
function oneOf(raw, allowed, fallback) {
  return typeof raw === 'string' && allowed.includes(raw) ? raw : fallback;
}

/**
 * A node id is 64 lowercase hex characters — `NodeId`'s own contract, which
 * the Rust side enforces on parse and this side did not.
 *
 * It matters more than the other fields: it is the fingerprint the pairing
 * ceremony asks a human to compare, so an id carrying bidi marks could reorder
 * the very string being checked. Anything else is not a node.
 */
function nodeId(raw) {
  return typeof raw === 'string' && /^[0-9a-f]{64}$/.test(raw) ? raw : null;
}

/** A vital is a finite number or it is absent. Never a string to `.toFixed`. */
function vitals(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    if (typeof k !== 'string' || k.length > 64) continue;
    out[sanitize(k, 64)] = Number.isFinite(v) ? v : null;
  }
  return out;
}

/**
 * Normalise a node descriptor from the wire into something safe to render.
 *
 * @param {object} raw
 * @returns {object}
 */
export function ingestNode(raw) {
  const addresses = Array.isArray(raw?.addresses) ? raw.addresses : [];
  const id = nodeId(raw?.id);
  // A descriptor with no usable id is not a node. Keeping it gave every
  // id-less beacon the same `{#each}` key, so two of them collided and Svelte
  // patched the wrong card.
  if (id === null) return null;
  return {
    id,
    name: sanitize(raw?.name) || 'unnamed',
    isLocal: raw?.is_local === true,
    pairing: oneOf(raw?.pairing, PAIRING, 'discovered'),
    addresses: addresses.slice(0, 8).map((a) => ({
      iface: sanitize(a?.iface, 32),
      addr: sanitize(a?.addr, 64),
      class: oneOf(a?.class, LINK_CLASS, 'unverified'),
      speedMbps: Number.isFinite(a?.speed_mbps) ? a.speed_mbps : null,
      rdma: a?.rdma === true
    })),
    canLaunch: raw?.launchability?.can_launch === true,
    cannotLaunchReason: sanitize(raw?.launchability?.reason, DETAIL_MAX),
    agentVersion: sanitize(raw?.agent_version, 32),
    accelerator: sanitize(raw?.accelerator, 32),
    // Reported only over the authenticated channel — a beacon carries none —
    // so a machine we have merely seen shows a blank rather than a guess.
    // Sanitised regardless, because everything on this path is untrusted input.
    os: sanitize(raw?.os, 32),
    vitals: vitals(raw?.vitals),
    alerts: (Array.isArray(raw?.alerts) ? raw.alerts : []).slice(0, 8).map((a) => ({
      // `kind` is rendered through .replaceAll('_', ' '); a non-string there
      // throws and blanks the whole control page.
      kind: sanitize(a?.kind, 64) || 'unknown',
      severity: oneOf(a?.severity, SEVERITY, 'warning'),
      detail: sanitize(a?.detail, DETAIL_MAX)
    })),
    running: raw?.running ? sanitize(raw.running, 64) : null,
    lastSeen: Date.now()
  };
}

/** The node's best address, which is what a collective would use. */
