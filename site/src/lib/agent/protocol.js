// SPDX-License-Identifier: AGPL-3.0-only

// The wire contract with the local agent.
//
// Mirrors crates/atlasctl-protocol in atlas-recipes. Kept deliberately small:
// the whole surface a page can reach is a handful of message types, and that is
// the point — the agent has no relay or raw-command verb to reach for.

// 2: pairing became two-phase. `pair_peer` runs the exchange and writes no
// pin; `confirm_pairing` establishes trust and `reject_pairing` discards it.
// `pair_result.paired` became `.exchanged` because it no longer means trusted.
// The agent enforces an exact match, so a page still on 1 is refused at the
// handshake rather than reading `exchanged` as "trusted" and showing a machine
// as paired that the agent has not accepted.
//
// 3: `pair_peer_at` added, so a machine can be added by typing its address.
// mDNS is link-local — it does not cross a router and is off on plenty of
// managed networks — so without it the page could only reach machines on one
// broadcast domain. Additive, but the handshake is exact-match by design.
export const PROTOCOL_VERSION = 3;

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
  return /^[0-9a-f]{64}$/.test(value.trim());
}

/** Human text for an agent error code. */
export function describeError(error) {
  if (!error || typeof error !== 'object') return 'The agent reported an unknown problem.';
  switch (error.code) {
    case 'not_paired':
      return 'The agent did not accept that pairing token. Run `atlasctl agent token` and paste the value it prints.';
    case 'unsupported_protocol':
      return `This page speaks protocol ${PROTOCOL_VERSION}; your agent speaks ${error.min}–${error.max}. Update whichever is older.`;
    case 'unknown_recipe':
      return `Your agent does not have a recipe called “${error.recipe}”. Update atlasctl to get the latest recipe set.`;
    case 'not_launchable':
      return `That recipe cannot run here: ${error.reason}`;
    case 'bad_settings':
      return (error.errors ?? []).map((e) => e.key ?? 'setting').join(', ');
    case 'already_running':
      return 'That recipe is already running.';
    case 'docker_unavailable':
      return `Docker is not available on that machine: ${error.detail}`;
    case 'launch_failed':
      return `The launch failed: ${error.detail}`;
    default:
      return error.code ?? 'The agent reported an unknown problem.';
  }
}
