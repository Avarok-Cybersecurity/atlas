// SPDX-License-Identifier: AGPL-3.0-only

// The one focus-trap for every dialog on the site.
//
// It sat in `components/control/` while only the control surface used it, and
// three dialogs outside that folder claimed `aria-modal="true"` without any
// of the duties below — LaunchModal, the chat skeleton in Nav, and
// GatePointCard. `aria-modal` tells assistive tech the rest of the page is
// inert; with no trap, Tab walks straight out into content the screen reader
// has been told to ignore, which is worse than never claiming it.
//
// DOM plumbing, not business logic. Kept thin for the same reason
// client.svelte.js is: the test runner has no DOM, so anything here is
// untestable by construction and must stay too small to hide a rule.
//
// Three duties, all of them WCAG's, none of them optional once a surface
// claims `aria-modal="true"`:
//
//   1. focus moves INTO the dialog when it opens,
//   2. Tab cycles inside it — the page behind a modal must be unreachable,
//   3. focus RETURNS to the opener when it closes, so a keyboard operator is
//      put back where they were instead of dropped at the top of the page.
//
// Escape stays each dialog's own affair: closing is a plain dismissal in
// most, but in the pairing ceremony it is an explicit rejection, and a
// generic handler would flatten that difference.

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
  'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Svelte action: `use:modal` on the element carrying `role="dialog"`.
 *
 * @param {HTMLElement} node
 */
export function modal(node) {
  const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  node.focus();

  function onKeydown(ev) {
    if (ev.key !== 'Tab') return;
    // Window-level, because a step change can unmount the focused control
    // and drop focus to <body> — a dialog-scoped listener goes deaf exactly
    // then, and the next Tab walks the page behind the modal.
    const focusables = [...node.querySelectorAll(FOCUSABLE)].filter(
      (el) => el.offsetParent !== null || el === document.activeElement
    );
    if (focusables.length === 0) {
      // Nothing to land on: the dialog itself keeps focus.
      ev.preventDefault();
      node.focus();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (ev.shiftKey && (active === first || active === node || !node.contains(active))) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && (active === last || !node.contains(active))) {
      ev.preventDefault();
      first.focus();
    }
  }

  window.addEventListener('keydown', onKeydown);
  return {
    destroy() {
      window.removeEventListener('keydown', onKeydown);
      // The opener can be gone — a row that vanished, a button the close
      // re-rendered away. Focusing a detached element is a silent no-op, so
      // the check is only about not throwing on a null.
      opener?.focus?.();
    }
  };
}
