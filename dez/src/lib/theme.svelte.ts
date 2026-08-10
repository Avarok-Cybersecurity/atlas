// SPDX-License-Identifier: AGPL-3.0-only

import { browser } from '$app/environment';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'dez-theme';

function isTheme(value: unknown): value is Theme {
  return value === 'light' || value === 'dark';
}

/** What the OS is asking for, right now. */
function systemTheme(): Theme {
  if (!browser) return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/** The user's stored override, or `null` when they have never chosen. */
function storedTheme(): Theme | null {
  if (!browser) return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return isTheme(raw) ? raw : null;
  } catch {
    // Storage can be disabled (private mode, hardened profiles). Not fatal:
    // the toggle still works for the session, it just will not persist.
    return null;
  }
}

class ThemeController {
  /** `null` means "follow the OS" — no explicit choice has been made. */
  #override = $state<Theme | null>(null);
  #system = $state<Theme>('dark');

  /** The theme actually being rendered. */
  get current(): Theme {
    return this.#override ?? this.#system;
  }

  get isExplicit(): boolean {
    return this.#override !== null;
  }

  /**
   * Reads the stored preference and starts tracking the OS setting.
   * Returns a teardown function for the caller's effect.
   */
  init(): () => void {
    if (!browser) return () => {};

    this.#override = storedTheme();
    this.#system = systemTheme();

    const query = window.matchMedia('(prefers-color-scheme: light)');
    const onChange = (event: MediaQueryListEvent) => {
      this.#system = event.matches ? 'light' : 'dark';
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }

  toggle(): void {
    this.set(this.current === 'dark' ? 'light' : 'dark');
  }

  set(theme: Theme): void {
    this.#override = theme;
    if (!browser) return;
    document.documentElement.setAttribute('data-theme', theme);
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // See storedTheme(): persistence is best-effort by design.
    }
  }
}

export const theme = new ThemeController();
