// SPDX-License-Identifier: AGPL-3.0-only

// The bridge's keyboard map — one table, two consumers.
//
// Pure and plain `.js` for the house reason: which key does what, and when a
// key must do nothing, are testable rules, and a file holding runes cannot be
// imported by the test runner.
//
// The same table drives the dispatcher AND renders the "?" shortcut sheet, so
// the sheet can never document a key the page does not honour, and the page
// can never grow a key the sheet does not document.
//
// Three suppression rules, each a bug when violated:
//
// **Typing wins.** A key pressed inside an input, textarea, select or
// editable region is text, not a command — an operator typing "spark-1" into
// the add-by-address field must not have the page teleport its selection.
//
// **Overlays own their keys.** While any overlay or dialog is open, the page
// dispatches nothing: the overlay is focus-trapped, and a hotkey that changed
// the stage behind it would act on something the operator cannot see. Esc is
// the overlay's own affair.
//
// **Modifiers mean the browser.** Ctrl/Cmd/Alt chords belong to the browser
// and the OS; stealing Cmd+1 from tab switching is hostile.
//
// The arrow keys are deliberately DOCUMENTED here but not DISPATCHED: they
// rove the roster only while focus is inside it (Roster.svelte's own
// handler), because a global arrow key would steal keyboard scrolling from
// every scroll region the spec requires to stay keyboard-reachable.

/**
 * The map. `test` matches a key; `act` builds the action, or null for rows
 * that exist only for the sheet (arrows, Esc — handled elsewhere, by design).
 */
export const SHORTCUTS = [
  {
    keys: '1–8',
    label: 'Select that roster row',
    test: (k) => /^[1-8]$/.test(k),
    act: (k) => ({ kind: 'select', key: k })
  },
  {
    keys: '↑ ↓',
    label: 'Rove the roster (while it has focus)',
    test: () => false,
    act: null
  },
  {
    keys: 'l',
    label: 'Logs tab',
    test: (k) => k === 'l',
    act: () => ({ kind: 'tab', tab: 'logs' })
  },
  {
    keys: 'n',
    label: 'Launch tab',
    test: (k) => k === 'n',
    act: () => ({ kind: 'tab', tab: 'launch' })
  },
  {
    keys: 's',
    label: 'Stop the selected node — two presses, arm then confirm',
    test: (k) => k === 's',
    act: () => ({ kind: 'stop' })
  },
  {
    keys: 'a',
    label: 'Jump to the alert lane',
    test: (k) => k === 'a',
    act: () => ({ kind: 'alerts' })
  },
  {
    keys: 'c',
    label: 'Cluster launch overlay',
    test: (k) => k === 'c',
    act: () => ({ kind: 'cluster' })
  },
  {
    keys: 'p',
    label: 'Pause / resume polling',
    test: (k) => k === 'p',
    act: () => ({ kind: 'pause' })
  },
  {
    keys: '?',
    label: 'This sheet',
    test: (k) => k === '?',
    act: () => ({ kind: 'sheet' })
  },
  {
    keys: 'Esc',
    label: 'Close the open overlay or popover',
    test: () => false,
    act: null
  }
];

/**
 * What a key press should do, or null for "nothing".
 *
 * All three context flags are required. A caller that forgot to say whether
 * an overlay is open would dispatch hotkeys underneath one, and that is a
 * bug to surface at the call site, not a default to guess.
 *
 * @param {string} key `KeyboardEvent.key`
 * @param {{typing: boolean, overlayOpen: boolean, modified: boolean}} ctx
 * @returns {{kind: string}|null}
 */
export function shortcut(key, ctx) {
  for (const flag of ['typing', 'overlayOpen', 'modified']) {
    if (ctx?.[flag] !== true && ctx?.[flag] !== false) {
      throw new TypeError(`shortcut() must be told ${flag}`);
    }
  }
  if (ctx.typing || ctx.overlayOpen || ctx.modified) return null;
  if (typeof key !== 'string') return null;
  // Caps lock must not disable the console: letters match case-insensitively.
  // '?' is already shifted; further normalisation would break nothing else.
  const k = key.length === 1 ? key.toLowerCase() : key;
  for (const s of SHORTCUTS) {
    if (s.act && s.test(k)) return s.act(k);
  }
  return null;
}
