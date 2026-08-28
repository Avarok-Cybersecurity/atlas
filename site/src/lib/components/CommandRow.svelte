<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // One copyable command.
  //
  // Its own component because the same six lines — the row, the button, the
  // "Copied" flash and its timeout — were repeated in three places, and a copy
  // button that silently fails is the kind of defect nobody reports: the
  // operator assumes they mis-clicked.
  //
  // The clipboard can genuinely refuse: it needs a secure context and, in some
  // browsers, a user gesture it does not think it has. Rather than pretend it
  // worked, a refusal selects the text so the operator can copy it with the
  // keyboard, and says so.

  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';

  let { command, label = 'Copy', extra = '' } = $props();

  let state = $state('idle'); // idle | copied | manual | blocked
  let codeEl = $state(null);
  let timer;

  // A dialog can close while the "Copied" flash is still pending. Without this
  // the timeout fires against a component that no longer exists.
  $effect(() => () => clearTimeout(timer));

  async function copy() {
    clearTimeout(timer);
    // Select-on-refusal lives in `clipboard.js` now: three other components
    // had the version that renders nothing instead.
    state = await copyOrSelect(command, codeEl);
    timer = setTimeout(() => (state = 'idle'), 2400);
  }
</script>

<div class="ld-cmd {extra}">
  <!-- `tabindex` because this scrolls horizontally: a join command naming
       every address a machine offers is reliably wider than the box, and a
       scrollable region a keyboard cannot reach is content a keyboard cannot
       read. Not a button — it is text, and `role`/`aria-label` name it so the
       stop is explicable rather than a mystery focus. -->
  <code
    class="mono"
    bind:this={codeEl}
    tabindex="0"
    role="group"
    aria-label="Command, scrollable">{command}</code>
  <button type="button" class="cmd-copy" onclick={copy}>{copyLabel(state, label)}</button>
</div>
