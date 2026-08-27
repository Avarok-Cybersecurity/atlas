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

  let { command, label = 'Copy' } = $props();

  let state = $state('idle'); // idle | copied | manual
  let codeEl = $state(null);
  let timer;

  async function copy() {
    clearTimeout(timer);
    try {
      await navigator.clipboard.writeText(command);
      state = 'copied';
    } catch {
      // Select it instead, so the next keystroke can copy it.
      if (codeEl) {
        const range = document.createRange();
        range.selectNodeContents(codeEl);
        const sel = window.getSelection();
        sel?.removeAllRanges();
        sel?.addRange(range);
      }
      state = 'manual';
    }
    timer = setTimeout(() => (state = 'idle'), 2400);
  }
</script>

<div class="ld-cmd">
  <code class="mono" bind:this={codeEl}>{command}</code>
  <button type="button" class="cmd-copy" onclick={copy}>
    {state === 'copied' ? 'Copied' : state === 'manual' ? 'Press ⌘/Ctrl+C' : label}
  </button>
</div>
