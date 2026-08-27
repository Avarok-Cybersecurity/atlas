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

  import { copyText, selectText } from '$lib/clipboard.js';

  let { command, label = 'Copy' } = $props();

  let state = $state('idle'); // idle | copied | manual | blocked
  let codeEl = $state(null);
  let timer;

  // A dialog can close while the "Copied" flash is still pending. Without this
  // the timeout fires against a component that no longer exists.
  $effect(() => () => clearTimeout(timer));

  async function copy() {
    clearTimeout(timer);
    if ((await copyText(command)) === 'copied') {
      state = 'copied';
    } else {
      // Select it instead, so the next keystroke can copy it. Flashing nothing
      // would leave the operator believing they had it.
      state = selectText(codeEl) ? 'manual' : 'blocked';
    }
    timer = setTimeout(() => (state = 'idle'), 2400);
  }
</script>

<div class="ld-cmd">
  <code class="mono" bind:this={codeEl}>{command}</code>
  <button type="button" class="cmd-copy" onclick={copy}>
    {state === 'copied'
      ? 'Copied'
      : state === 'manual'
        ? 'Press ⌘/Ctrl+C'
        : state === 'blocked'
          ? 'Select it above'
          : label}
  </button>
</div>
