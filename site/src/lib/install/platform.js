// SPDX-License-Identifier: AGPL-3.0-only

// Which install command this visitor can actually run.
//
// Every visitor was shown `curl -fsSL … | sh`, including the ones on Windows,
// where it cannot work: PowerShell has no `sh`, so it fails on a parse error
// that names nothing. Git Bash reaches install.sh and is refused by it. The
// only paste that ever worked was inside WSL, which nothing told anyone about.
//
// Detection is a hint, never a gate. Both commands stay reachable — an OS guess
// from a user-agent string is wrong often enough that hiding the other one
// would strand the people it guessed wrong about, and they are the ones with
// the least idea why.

/** @typedef {'windows' | 'macos' | 'linux' | 'unknown'} Os */

/**
 * Classify a host from what the browser will tell us.
 *
 * `uaDataPlatform` is `navigator.userAgentData.platform`, which is the
 * accurate answer where it exists; the user-agent string is the fallback for
 * everything else. Order matters: Chromium's UA string still carries
 * "Windows NT" inside the frozen `Windows NT 10.0` token regardless of the
 * real version, so the structured value is preferred when present.
 *
 * @param {string} [userAgent]
 * @param {string} [uaDataPlatform]
 * @returns {Os}
 */
export function detectOs(userAgent, uaDataPlatform) {
  const structured = String(uaDataPlatform ?? '').toLowerCase();
  if (structured) {
    if (structured.includes('win')) return 'windows';
    if (structured.includes('mac')) return 'macos';
    if (structured.includes('linux') || structured.includes('android')) return 'linux';
  }
  const ua = String(userAgent ?? '');
  // Checked before Windows: "Windows Phone" is gone, but a UA claiming both is
  // still a UA, and a wrong guess here hands someone a command for the wrong
  // shell.
  if (/Android/i.test(ua)) return 'linux';
  if (/Windows NT|Win64|WOW64/i.test(ua)) return 'windows';
  if (/Macintosh|Mac OS X|iPhone|iPad/i.test(ua)) return 'macos';
  if (/Linux|X11|CrOS/i.test(ua)) return 'linux';
  return 'unknown';
}

/**
 * The install one-liner for a host, and what to call the shell it goes in.
 *
 * `unknown` gets the unix command rather than a chooser: it is what the
 * overwhelming majority of unclassified visitors can run, and it is also what
 * the prerendered HTML contains, so an unknown host sees no flicker.
 *
 * @param {Os} os
 * @param {{ shellUrl: string, powershellUrl: string }} urls
 */
export function installCommandFor(os, urls) {
  if (os === 'windows') {
    return {
      os,
      // `irm | iex` is the idiom every Windows tool ships, and the one a
      // Windows user recognises as "the install line".
      command: `irm ${urls.powershellUrl} | iex`,
      shell: 'PowerShell',
      // Where install.ps1 actually puts it. Printed beside the command, so a
      // caller cannot narrate a path the installer never writes to.
      installDir: '%LOCALAPPDATA%\\Programs\\atlasctl',
      note: 'Needs Docker Desktop to run a model; the control page works without it.',
    };
  }
  return {
    os: os === 'unknown' ? 'linux' : os,
    command: `curl -fsSL ${urls.shellUrl} | sh`,
    shell: os === 'macos' ? 'Terminal' : 'bash',
    installDir: '~/.local/bin',
    note: '',
  };
}

/**
 * Read the running browser, when there is one.
 *
 * Returns `'unknown'` during prerendering rather than guessing, so the HTML
 * that ships and the HTML that hydrates agree.
 *
 * @returns {Os}
 */
export function detectCurrentOs() {
  if (typeof navigator === 'undefined') return 'unknown';
  return detectOs(navigator.userAgent, navigator.userAgentData?.platform);
}
