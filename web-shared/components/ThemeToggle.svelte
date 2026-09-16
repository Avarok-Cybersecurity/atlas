<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import { toggleTheme } from '../theme.js';
</script>

<button
  type="button"
  class="theme-toggle"
  aria-label="Switch between light and dark theme"
  onclick={() => toggleTheme()}
>
  <svg class="theme-toggle-moon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 14.3A8.4 8.4 0 1 1 9.7 3 7 7 0 0 0 21 14.3z" />
  </svg>
  <svg class="theme-toggle-sun" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
</button>

<style>
  .theme-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    flex-shrink: 0;
    cursor: pointer;
    background: var(--card, transparent);
    color: var(--t2, currentColor);
    border: 1px solid var(--border-strong, currentColor);
    border-radius: 9px;
  }
  .theme-toggle:hover {
    color: var(--accent, currentColor);
    border-color: var(--accent, currentColor);
  }
  .theme-toggle-sun {
    display: none;
  }
  :global(html[data-theme='dark']) .theme-toggle-sun {
    display: block;
  }
  :global(html[data-theme='dark']) .theme-toggle-moon {
    display: none;
  }
</style>
