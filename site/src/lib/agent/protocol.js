// SPDX-License-Identifier: AGPL-3.0-only

// The wire contract with the local agent.
//
// Mirrors crates/atlasctl-protocol in atlas-recipes. Kept deliberately small:
// the whole surface a page can reach is a handful of message types, and that is
// the point. There is no raw-command verb, no nested-message verb and no relay
// of opaque bytes, and the enum is closed — an unknown `type` fails to
// deserialize rather than reaching a handler.
//
// One scoped exception, stated here so this file cannot outgrow the doctrine it
// mirrors: the seven single-node control verbs carry an optional `on` target,
// which the agent honours by re-issuing the request AS ITSELF over its
// authenticated peer channel — one hop, only toward a machine it has itself
// pinned AND whose pin of the requester carries an explicit `controller` grant.
// Forwarding is an ANNOTATION on closed verbs, never a wrapper around arbitrary
// messages: the forwardable vocabulary cannot express pairing, joining, cluster
// reservation, or a further hop. That is what still keeps the agent from being
// an open proxy for whatever page is talking to it.

// 2: pairing became two-phase. `pair_peer` runs the exchange and writes no
// pin; `confirm_pairing` establishes trust and `reject_pairing` discards it.
// `pair_result.paired` became `.exchanged` because it no longer means trusted.
// The agent enforces an exact match, so a page still on 1 is refused at the
// handshake rather than reading `exchanged` as "trusted" and showing a machine
// as paired that the agent has not accepted.
//
// 4: control verbs gained an optional `on` target and the replies gained
// `on`/`via`, so a page can drive a machine reached through a peer. See the
// doctrine note above for what that deliberately cannot do.
//
// 3: `pair_peer_at` added, so a machine can be added by typing its address.
// mDNS is link-local — it does not cross a router and is off on plenty of
// managed networks — so without it the page could only reach machines on one
// broadcast domain. Additive, but the handshake is exact-match by design.
export const PROTOCOL_VERSION = 4;

// The agent binds loopback only. Connecting to anything else would defeat the
// entire security model, so the address is a literal here too.
export const AGENT_PORT = 34333;
export const AGENT_URL = `ws://127.0.0.1:${AGENT_PORT}/ws`;

/** Where the pairing token is remembered between visits. */
export const TOKEN_KEY = 'atlas.agent.token';

/** Read the stored pairing token, if the user has pasted one. */
export function storedToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    // Private browsing, or storage disabled. Not fatal: the user can paste the
    // token again for this session.
    return '';
  }
}

/** Remember a pairing token. */
export function storeToken(token) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* nothing we can do, and nothing that should break the page */
  }
}

/** Forget the pairing token. */
export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* see above */
  }
}

/** A token is 32 bytes of hex. Checked here only to catch a bad paste early. */
export function looksLikeToken(value) {
  // Every other export here tolerates junk; this one used to throw on a
  // non-string, which is the wrong answer to "does this look like a token".
  if (typeof value !== 'string') return false;
  return /^[0-9a-f]{64}$/.test(value.trim());
}

// Said when the agent reports something this page cannot name. Kept in one
// place so every unnameable path says the same thing rather than going blank.
const UNKNOWN_PROBLEM = 'The agent reported an unknown problem.';

/** Human text for an agent error code. */
export function describeError(error) {
  if (!error || typeof error !== 'object') return UNKNOWN_PROBLEM;
  switch (error.code) {
    case 'not_paired':
      return 'The agent did not accept that pairing token. Run `atlasctl agent token` and paste the value it prints.';
    case 'unsupported_protocol':
      return `This page speaks protocol ${PROTOCOL_VERSION}; your agent speaks ${error.min}–${error.max}. Update whichever is older.`;
    case 'unknown_recipe':
      return `Your agent does not have a recipe called “${error.recipe}”. Update atlasctl to get the latest recipe set.`;
    case 'not_launchable':
      return `That recipe cannot run here: ${error.reason}`;
    case 'bad_settings': {
      // This arrives over the socket, so `errors` is whatever the agent sent.
      // A non-array used to throw here — turning the one function whose job is
      // to explain a failure into a second failure.
      const listed = Array.isArray(error.errors)
        ? error.errors.map((e) => (typeof e?.key === 'string' ? e.key : 'setting'))
        : [];
      // An empty list rendered as an empty string: something was rejected and
      // the screen said nothing at all.
      return listed.length > 0
        ? `The agent rejected these settings: ${listed.join(', ')}`
        : 'The agent rejected the settings but did not say which.';
    }
    case 'already_running':
      return 'That recipe is already running.';
    case 'docker_unavailable':
      return `Docker is not available on that machine: ${error.detail}`;
    case 'launch_failed':
      return `The launch failed: ${error.detail}`;
    default:
      // A non-string code would otherwise be returned as-is and reach the UI
      // as "[object Object]".
      return typeof error.code === 'string' && error.code !== '' ? error.code : UNKNOWN_PROBLEM;
  }
}

/**
 * What to tell an operator whose agent and page disagree about the protocol.
 *
 * "Update whichever is older" was true and unhelpful: it named no command, and
 * it made the operator work out which side was behind from two version numbers
 * they have no reason to care about. Protocol 4 shipped to a fleet of agents
 * that all speak something older, so this is now the first thing many people
 * will see — it has to end with something they can run.
 *
 * The two directions have genuinely different remedies, which is why this does
 * not just print both:
 *
 * - the AGENT is behind — reinstall it, which is one line;
 * - the PAGE is behind — the browser is holding a cached bundle, and no amount
 *   of updating the agent will fix it. That one is a hard reload.
 *
 * @param {number} page this bundle's `PROTOCOL_VERSION`
 * @param {number} min the agent's lowest supported version
 * @param {number} max the agent's highest
 * @returns {{ok: true} | {ok: false, side: 'agent'|'page', message: string}}
 */
export function versionAdvice(page, min, max) {
  if (!Number.isInteger(page) || !Number.isInteger(min) || !Number.isInteger(max)) {
    // A malformed welcome is not a version mismatch, and guessing which side is
    // behind from a non-number would name a remedy at random.
    return {
      ok: false,
      side: 'agent',
      message: 'The agent did not say which protocol it speaks, so this page cannot tell whether it is compatible. Reinstall the agent: curl -fsSL https://atlasinference.io/install.sh | sh'
    };
  }
  if (page >= min && page <= max) return { ok: true };

  if (page > max) {
    return {
      ok: false,
      side: 'agent',
      message: `Your agent is out of date — it speaks protocol ${min === max ? min : `${min}–${max}`}, this page speaks ${page}. Update it on that machine: curl -fsSL https://atlasinference.io/install.sh | sh`
    };
  }
  return {
    ok: false,
    side: 'page',
    message: `This page is out of date — it speaks protocol ${page}, your agent speaks ${min === max ? min : `${min}–${max}`}. Your browser is holding an old copy: reload with Ctrl-Shift-R (⌘-Shift-R on a Mac).`
  };
}
