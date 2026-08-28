// SPDX-License-Identifier: AGPL-3.0-only

// The one line an operator pastes on the machine they are adding.
//
// Pure and separate because it is a *credential-bearing* string and the rules
// about it are worth stating once: which address it names, what happens when
// there is none, and that it never renders half an invitation. A command that
// is subtly wrong is worse than no command, because it fails on the far
// machine where the operator has the least context.

import { installerUrl, powershellInstallerUrl } from '../data.js';

/**
 * Build the install-and-join one-liner.
 *
 * Returns an empty string when there is nothing valid to render — no code, or
 * no address the other machine could dial. Half a command would look
 * copy-pasteable and would not be.
 *
 * ALL the addresses go in, comma-separated, because this machine cannot know
 * which of its networks the new one shares. A DGX offers its RoCE fabric first
 * — the right answer for another DGX, and unreachable from a laptop on the
 * ordinary LAN. Naming only the first worked for whichever machine we guessed
 * and failed on the other, remotely, after a clean install. The joiner walks
 * the list; `atlasctl` stops the walk as soon as a machine actually answers,
 * so the alternatives never cost the code its limited attempts.
 *
 * @param {{code: string, addresses: string[]}|null} join
 * @returns {string}
 */
/**
 * The `code@host,host` operand itself, unquoted and unwrapped.
 *
 * Extracted because THREE places need it and two of them used to get it by
 * string surgery on the finished command: `shortForm` took everything after the
 * last space, which quietly became `'code@host'` -- quotes and all, on screen --
 * the moment the sh line started quoting its operand. A shared builder cannot
 * drift like that.
 *
 * @param {{code: string, addresses: string[]}|null} join
 * @returns {string} '' when there is no usable code or no dialable address
 */
export function joinOperand(join) {
  const code = typeof join?.code === 'string' ? join.code.trim() : '';
  // Same reasoning as the host allowlist: this ends up in a shell line.
  if (!code || !CODE_OK.test(code)) return '';
  const hosts = dialableAddresses(join?.addresses);
  if (hosts.length === 0) return '';
  return `${code}@${hosts.join(',')}`;
}

export function joinCommand(join, grantControl = false) {
  const operand = joinOperand(join);
  if (!operand) return '';
  // SINGLE-QUOTED, for the same reason the PowerShell line below is. HOST_OK
  // admits `[` and `]` so a bracketed IPv6 address can reach this string, and
  // bare brackets in an operand are a GLOB: zsh — the default shell on macOS —
  // refuses the whole line with `no matches found` rather than running it. No
  // address we mint today is bracketed, so this is latent, not broken; it stops
  // being latent the first time an address carries a port or a v6 literal.
  // Nothing needs escaping inside the quotes: CODE_OK and HOST_OK both exclude `'`.
  const base = `curl -fsSL ${installerUrl} | sh -s -- --join '${operand}'`;
  // The grant is a VISIBLE flag on the line the operator pastes, never an
  // implication of joining. It is also the only direction that does what
  // someone adding a GPU box actually wants: it is written into THAT machine's
  // pin of this fleet, by the person standing at that keyboard. The reverse —
  // this machine granting the new one control of itself — is a different
  // decision and is not what "add a machine I can run models on" means.
  return grantControl ? `${base} --grant-control` : base;
}

/**
 * The same invitation, for a machine running Windows.
 *
 * A separate function and not a flag, because BOTH lines are shown: the
 * operator is standing at the machine being added and this one cannot see it,
 * so guessing its platform would be guessing about a computer that is not
 * here. Showing the wrong single line is a paste that fails on the far machine,
 * where they have the least context — the exact failure `joinCommand`'s header
 * already warns about.
 *
 * `irm | iex` cannot pass arguments, so this uses `[scriptblock]::Create`,
 * which is the idiom every Windows installer that takes options uses.
 *
 * @param {{code: string, addresses: string[]}|null} join
 * @returns {string}
 */
export function joinCommandPowerShell(join, grantControl = false) {
  const operand = joinOperand(join);
  if (!operand) return '';
  // The operand is SINGLE-QUOTED, and that is load-bearing. Unquoted, a comma
  // in PowerShell's argument mode builds an ARRAY — `-Join a@h1,h2` binds
  // `@('a@h1','h2')` and stringifies it with a SPACE, so the far machine
  // receives `a@h1 h2` and the multi-homed case this function exists for fails
  // remotely after a clean install. Verified against pwsh 7.4.6.
  //
  // No escaping is needed inside the quotes: CODE_OK and HOST_OK both exclude
  // `'`, so nothing that reaches here can close the string. Quoting also
  // neutralises a code beginning with `-`, which would otherwise parse as an
  // unknown parameter and silently install without joining.
  const base =
    `& ([scriptblock]::Create((irm ${powershellInstallerUrl}))) ` +
    `-Join '${operand}'`;
  return grantControl ? `${base} -GrantControl` : base;
}

/**
 * Every address another machine could actually dial, best link first.
 *
 * The agent already orders these and strips loopback. This re-checks anyway:
 * a command naming 127.0.0.1 would install cleanly and then fail to pair,
 * which is the most confusing failure available here.
 *
 * One filter, so the command and the troubleshooting copy beside it can never
 * disagree about which machines are in play.
 *
 * @param {string[]|undefined} addresses
 * @returns {string[]}
 */
/**
 * What a host may contain. An allowlist, not an escape: this value is
 * interpolated into a line built to be pasted into a shell on a machine
 * someone just walked to, and the control page is served over plain http on a
 * LAN address, so the socket feeding it is not authenticated.
 *
 * Covers IPv4, a bracketed or bare IPv6 literal, a hostname, and `host:port`.
 * Excludes whitespace (which word-splits into extra flags), the comma (which
 * is the separator between hosts, so one inside a host forges a second), and
 * every shell metacharacter.
 */
const HOST_OK = /^[A-Za-z0-9._~:[\]-]+$/;

/** Digits and dashes only — the code is minted, never typed into this. */
const CODE_OK = /^[A-Za-z0-9_-]+$/;

/**
 * The host part of an address, with any port and brackets removed.
 *
 * A bare IPv6 literal is all colons, so "strip the port" cannot mean "cut at
 * the colon": only a bracketed form, or a single colon followed by digits, is
 * a port. Same rule `network.js` applies to typed input.
 *
 * @param {string} s
 * @returns {string}
 */
function bareHost(s) {
  const bracketed = /^\[([^\]]+)\](?::\d+)?$/.exec(s);
  if (bracketed) return bracketed[1].toLowerCase();
  const i = s.indexOf(':');
  if (i !== -1 && i === s.lastIndexOf(':') && /^\d+$/.test(s.slice(i + 1))) {
    return s.slice(0, i).toLowerCase();
  }
  return s.toLowerCase();
}

/**
 * Whether this address points back at the machine showing the command.
 *
 * The literal `127.0.0.1` was the only form checked, and every other spelling
 * of loopback walked straight through: `localhost`, `[::1]:8443`, the
 * IPv4-mapped `::ffff:127.0.0.1`, and the expanded `0:0:0:0:0:0:0:1`. Each
 * produced a complete, pasteable command that installs cleanly and then fails
 * to pair — which this file's own header calls the most confusing failure
 * available here.
 *
 * @param {string} host already reduced by `bareHost`
 * @returns {boolean}
 */
function isLoopback(host) {
  if (host === 'localhost' || host.endsWith('.localhost')) return true;
  if (host.startsWith('127.')) return true;
  if (host.startsWith('::ffff:127.')) return true;
  const groups = host.split(':');
  if (groups.length > 1 && groups.every((g) => g === '' || /^0+$/.test(g))) {
    // `::`, `::1` and `0:0:0:0:0:0:0:1` all reduce to all-zero groups; `::1`
    // ends in a 1, so check it separately.
    return true;
  }
  return host === '::1' || /^(0+:){7}0*1$/.test(host);
}

export function dialableAddresses(addresses) {
  const list = Array.isArray(addresses) ? addresses : [];
  const out = [];
  for (const a of list) {
    // A non-string is dropped, not coerced: `String({addr})` yields
    // "[object Object]", which is non-empty and therefore renders a command
    // that looks pasteable and cannot work. Node addresses ARE objects
    // elsewhere in this protocol, so one shape drift would ship exactly that.
    if (typeof a !== 'string') continue;
    const s = a.trim();
    if (!s || !HOST_OK.test(s)) continue;
    if (isLoopback(bareHost(s))) continue;
    if (!out.includes(s)) out.push(s);
  }
  return out;
}

/**
 * The first address another machine should dial, or null.
 *
 * Still single, because the troubleshooting copy asks about one port on one
 * machine and a list there would read as three separate problems.
 *
 * @param {string[]|undefined} addresses
 * @returns {string|null}
 */
export function bestAddress(addresses) {
  return dialableAddresses(addresses)[0] ?? null;
}
