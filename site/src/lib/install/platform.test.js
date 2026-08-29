// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, it } from 'bun:test';
import { detectOs, installCommandFor } from './platform.js';

const URLS = {
  shellUrl: 'https://atlasinference.io/install.sh',
  powershellUrl: 'https://atlasinference.io/install.ps1',
};

describe('detectOs', () => {
  // Real strings. A hand-written approximation of a user-agent proves the
  // regex matches the approximation.
  const REAL = {
    windows:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    macos:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
    linux:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  };

  it('classifies each real user-agent', () => {
    expect(detectOs(REAL.windows)).toBe('windows');
    expect(detectOs(REAL.macos)).toBe('macos');
    expect(detectOs(REAL.linux)).toBe('linux');
  });

  // Android's UA contains "Linux", and a phone must not be told to curl an
  // installer — but of the two answers, "linux" is the one whose command at
  // least exists. What must never happen is "windows".
  it('does not read Android as Windows', () => {
    const android =
      'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36';
    expect(detectOs(android)).toBe('linux');
  });

  // The structured value is the accurate one where it exists, and must win:
  // Chromium freezes the UA string's platform token, so trusting the string
  // over the API is trusting the value that was deliberately made unreliable.
  it('prefers the structured platform over the user-agent string', () => {
    expect(detectOs(REAL.linux, 'Windows')).toBe('windows');
    expect(detectOs(REAL.windows, 'macOS')).toBe('macos');
  });

  // 'Darwin'.includes('win') is true, so a substring test in the wrong order
  // hands a Mac the PowerShell line — from the module that exists to stop it.
  it('does not read a Darwin platform as Windows', () => {
    expect(detectOs(REAL.macos, 'Darwin')).toBe('macos');
    expect(detectOs('', 'darwin')).toBe('macos');
  });

  it('answers unknown rather than guessing', () => {
    expect(detectOs('', undefined)).toBe('unknown');
    expect(detectOs(undefined, undefined)).toBe('unknown');
    expect(detectOs('Mozilla/5.0 (Nintendo Switch)')).toBe('unknown');
  });
});

describe('installCommandFor', () => {
  // The bug this whole module exists for: a Windows visitor handed `| sh`.
  it('never hands a Windows visitor a shell pipeline', () => {
    const { command } = installCommandFor('windows', URLS);
    expect(command).not.toContain('| sh');
    expect(command).not.toContain('curl');
    expect(command).toContain('install.ps1');
  });

  it('hands everyone else the shell installer', () => {
    for (const os of ['macos', 'linux', 'unknown']) {
      const { command } = installCommandFor(os, URLS);
      expect(command).toBe('curl -fsSL https://atlasinference.io/install.sh | sh');
    }
  });

  // An unclassified visitor must see exactly what the prerendered HTML holds,
  // or the page visibly rewrites itself on hydration for no reason.
  it('gives unknown the same command as linux', () => {
    expect(installCommandFor('unknown', URLS).command).toBe(
      installCommandFor('linux', URLS).command
    );
  });

  // The narration beside the command must describe the command that is there:
  // a Windows visitor told the binary lands in ~/.local/bin has been told
  // something no installer on their machine will do.
  it('draws the prompt the named shell actually draws', () => {
    // A `$` beside a window labelled PowerShell is the same kind of small lie
    // as `bash` beside an `irm` line.
    expect(installCommandFor('windows', URLS).prompt).toBe('PS>');
    expect(installCommandFor('linux', URLS).prompt).toBe('$');
    expect(installCommandFor('macos', URLS).prompt).toBe('$');
  });

  it('describes the install directory its own installer writes to', () => {
    expect(installCommandFor('windows', URLS).installDir).toContain('LOCALAPPDATA');
    expect(installCommandFor('linux', URLS).installDir).toBe('~/.local/bin');
    expect(installCommandFor('windows', URLS).shell).toBe('PowerShell');
  });

  // Both installers are built from the same two URLs, so a moved endpoint
  // cannot update one and leave the other pointing at a 404.
  it('builds both commands from the urls it is given', () => {
    const urls = { shellUrl: 'https://example.test/a.sh', powershellUrl: 'https://example.test/b.ps1' };
    expect(installCommandFor('linux', urls).command).toContain('https://example.test/a.sh');
    expect(installCommandFor('windows', urls).command).toContain('https://example.test/b.ps1');
  });
});
