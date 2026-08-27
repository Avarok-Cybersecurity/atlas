// SPDX-License-Identifier: AGPL-3.0-only

// Turning an address and a prefix into the name of a network.
//
// Pure and separate from any component for the reason the rest of this
// directory is split that way: a file holding runes cannot be imported by the
// test runner. Subnet arithmetic that is wrong by one bit is invisible on
// screen and obvious in a test, so it belongs here.
//
// IPv4 only, deliberately. Masking an IPv6 address correctly means 128-bit
// arithmetic over eight groups with `::` expansion, and a half-right answer
// here would label two machines as sharing a network when they do not — which
// is exactly the claim this exists to make. For IPv6 the honest answer is
// `null`: the UI then names the interface and link class, which it knows, and
// says nothing about the subnet, which it does not.

/** Human names for the link classes the agent reports. */
const CLASS_LABEL = {
  infini_band: 'InfiniBand',
  infiniband: 'InfiniBand',
  roce: 'RoCE',
  ethernet: 'Ethernet',
  wireless: 'Wi-Fi',
  virtual: 'virtual',
  loopback: 'loopback',
  unverified: 'unverified'
};

/**
 * The network an address sits on, as `10.10.10.0/24`.
 *
 * @param {string} addr dotted-quad IPv4
 * @param {number} prefixLen 1-32; 0 means the agent did not report one
 * @returns {string|null} null when it cannot be computed honestly
 */
export function subnetOf(addr, prefixLen) {
  if (!Number.isInteger(prefixLen) || prefixLen < 1 || prefixLen > 32) return null;
  const parts = String(addr ?? '').split('.');
  if (parts.length !== 4) return null;
  const octets = parts.map((p) => Number(p));
  if (octets.some((o) => !Number.isInteger(o) || o < 0 || o > 255)) return null;
  if (parts.some((p) => p === '' || !/^\d+$/.test(p))) return null;

  // `>>> 0` on both: a 32-bit mask built with `<<` is SIGNED in JS, so a /1
  // yields a negative number and every octet derived from it is wrong.
  //
  // No `prefixLen === 0` branch. It cannot reach here — the guard above returns
  // null below /1 — and writing one implied a case that does not exist. It was
  // there to dodge `0xffffffff << 32`, which JS evaluates as a shift by 32 % 32
  // = 0 and so leaves the mask all-ones rather than all-zeros. That trap is
  // real; it is simply not reachable, and a guard against an impossible input
  // reads as if the input were possible.
  const value = ((octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]) >>> 0;
  const mask = (0xffffffff << (32 - prefixLen)) >>> 0;
  const net = (value & mask) >>> 0;
  return `${(net >>> 24) & 255}.${(net >>> 16) & 255}.${(net >>> 8) & 255}.${net & 255}/${prefixLen}`;
}

/** A readable name for a link class. Unknown classes keep their own name. */
export function classLabel(cls) {
  return CLASS_LABEL[String(cls ?? '')] ?? String(cls ?? '');
}

/**
 * One address, described for a human.
 *
 * @param {{addr: string, iface: string, class: string, prefixLen: number,
 *   speedMbps: number|null, rdma: boolean}} a
 */
export function describeAddress(a) {
  const subnet = subnetOf(a?.addr, a?.prefixLen);
  const bits = [classLabel(a?.class)];
  if (a?.rdma) bits.push('RDMA');
  if (Number.isFinite(a?.speedMbps) && a.speedMbps > 0) {
    bits.push(a.speedMbps >= 1000 ? `${Math.round(a.speedMbps / 1000)} Gb` : `${a.speedMbps} Mb`);
  }
  return {
    addr: a?.addr ?? '',
    iface: a?.iface ?? '',
    subnet,
    detail: bits.filter(Boolean).join(' · ')
  };
}

/**
 * The distinct networks a node is reachable on, best link first.
 *
 * Addresses arrive ranked, so the order is preserved rather than re-sorted:
 * the agent already knows which of its links is the fabric, and re-deriving
 * that from a class name here would be a second, disagreeing source.
 *
 * @param {{addresses: Array<any>}} node
 */
export function networksOf(node) {
  const out = [];
  const seen = new Set();
  for (const a of node?.addresses ?? []) {
    const d = describeAddress(a);
    // Key on the subnet when there is one, so two addresses on the same LAN
    // read as one network — and on the address when there is not, so two IPv6
    // addresses do not collapse into a single unnamed row.
    const key = d.subnet ?? `addr:${d.addr}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(d);
  }
  return out;
}

/**
 * Whether a string is plausibly `host`, `host:port` or `[v6]:port`.
 *
 * Deliberately permissive about the host — a hostname, an IPv4 literal and a
 * bracketed IPv6 literal are all legitimate, and the agent resolves it anyway.
 * What this catches is the mistakes a person actually makes at this field:
 * pasting a URL, pasting the whole install one-liner, leaving it blank, or
 * typing a port that is not a port. The agent would reject those too, but a
 * round trip to be told "that is not an address" is a worse way to learn it.
 *
 * @param {string} s
 * @returns {{ok: true} | {ok: false, why: string}}
 */
export function checkTarget(s) {
  const t = String(s ?? '').trim();
  if (!t) return { ok: false, why: 'Enter the address of the machine to add.' };
  if (t.length > 253) return { ok: false, why: 'That is too long to be an address.' };
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(t)) {
    return { ok: false, why: 'Just the address — no http:// or other scheme.' };
  }
  if (/\s/.test(t)) return { ok: false, why: 'An address has no spaces in it.' };

  // Split off a port, being careful that a bare IPv6 literal is all colons.
  let host = t;
  let port = null;
  const bracketed = /^\[([^\]]+)\](?::(\d+))?$/.exec(t);
  if (bracketed) {
    host = bracketed[1];
    port = bracketed[2] ?? null;
  } else if (t.includes(':') && t.indexOf(':') === t.lastIndexOf(':')) {
    [host, port] = t.split(':');
  }

  if (!host) return { ok: false, why: 'That address has no host in it.' };
  if (port !== null) {
    const n = Number(port);
    if (!/^\d+$/.test(port) || n < 1 || n > 65535) {
      return { ok: false, why: 'The port after the colon must be between 1 and 65535.' };
    }
  }
  return { ok: true };
}
