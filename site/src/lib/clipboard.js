// SPDX-License-Identifier: AGPL-3.0-only

// Copying text, and saying so honestly when it did not work.
//
// Eight components had their own copy of this, and all eight failed the same
// way: `catch { /* the text is on screen either way */ }`. That comment is true
// and the behaviour it excuses is not — the button had already flashed nothing,
// so the operator believes they copied a command they did not copy, and pastes
// whatever was on the clipboard before. On the install path that means running
// the wrong line on a machine they walked to.
//
// The clipboard genuinely refuses: it needs a secure context (plain http on a
// LAN address is not one — exactly where this control page runs), and some
// browsers require a user gesture they do not think they have. So a refusal is
// an ordinary outcome to be reported, not an exception to be swallowed.

/**
 * Put `text` on the clipboard.
 *
 * Never throws: callers are click handlers, and an exception there is an
 * unhandled rejection rather than anything the operator can act on.
 *
 * @param {string} text
 * @param {{clipboard?: {writeText: (t: string) => Promise<void>}}} [nav]
 *   injected for tests, which have no real clipboard
 * @returns {Promise<'copied'|'denied'>} `denied` means the caller must show the
 *   text for manual copying — it must NOT report success.
 */
export async function copyText(text, nav = globalThis.navigator) {
  const write = nav?.clipboard?.writeText;
  if (typeof write !== 'function') return 'denied';
  try {
    await write.call(nav.clipboard, String(text ?? ''));
    return 'copied';
  } catch {
    return 'denied';
  }
}

/**
 * Select an element's text, so the next keystroke copies it.
 *
 * The fallback when `copyText` returns `denied`. Doing nothing there leaves the
 * operator with a button that visibly did nothing and no idea why.
 *
 * @param {Element|null|undefined} el
 * @returns {boolean} whether a selection was made
 */
export function selectText(el) {
  if (!el || typeof document === 'undefined') return false;
  try {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    if (!sel) return false;
    sel.removeAllRanges();
    sel.addRange(range);
    return true;
  } catch {
    return false;
  }
}

/**
 * Copy, or fall back to selecting the text so the keyboard can.
 *
 * `copyText` fixed half of the original defect — the swallowed exception. This
 * is the other half, and three components still had the first version of it:
 * `if (await copyText(t) !== 'copied') return;`, which renders NOTHING on a
 * refusal. The button does not change, so the operator reads the absence as
 * either "it worked" or "I mis-clicked", walks to the other machine and pastes
 * whatever was on the clipboard before.
 *
 * @param {string} text
 * @param {Element|null|undefined} el the element holding the visible text
 * @returns {Promise<'copied'|'manual'|'blocked'>} `manual` — it is selected,
 *   press ⌘/Ctrl+C. `blocked` — not even that worked; the text is on screen.
 */
export async function copyOrSelect(text, el) {
  if ((await copyText(text)) === 'copied') return 'copied';
  return selectText(el) ? 'manual' : 'blocked';
}

/**
 * The button's label for a copy state.
 *
 * Pure, and shared, so a refusal cannot be worded as a success in one place and
 * a failure in another. `idle` differs per call site ("Copy", "Copy command").
 *
 * @param {'idle'|'copied'|'manual'|'blocked'} state
 * @param {string} [idle]
 * @returns {string}
 */
export function copyLabel(state, idle = 'Copy') {
  switch (state) {
    case 'copied':
      return 'Copied';
    case 'manual':
      return 'Press ⌘/Ctrl+C';
    case 'blocked':
      // Not "above": in every one of these layouts the text sits BESIDE the
      // button, and a direction that is wrong is worse than none.
      return 'Select it manually';
    default:
      return idle;
  }
}
