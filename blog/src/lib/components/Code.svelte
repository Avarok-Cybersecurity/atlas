<!-- A code block with a recessed filename tab and a copy button.

     The source is passed as a plain string rather than as slot markup: slot
     content is HTML, so `<T>` inside a snippet of Rust would be parsed as a
     tag and silently disappear. A string cannot be misread that way, and it is
     also what the clipboard needs — copying the DOM's textContent picks up the
     line numbers and the gutter along with the code. -->
<script>
  let { name = '', lang = '', code } = $props();
  let copied = $state(false);
  let timer = 0;

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      copied = true;
      clearTimeout(timer);
      timer = setTimeout(() => (copied = false), 1600);
    } catch {
      copied = false; // no clipboard permission: the code is still selectable
    }
  }
</script>

<div class="codeblock">
  <div class="cb-bar">
    {#if name}<span class="cb-name">{name}</span>{/if}
    {#if lang}<span class="cb-lang">{lang}</span>{/if}
    <button class="cb-copy" type="button" onclick={copy} aria-label="Copy code">{copied ? 'Copied' : 'Copy'}</button>
  </div>
  <pre><code>{code}</code></pre>
</div>

<style>
  /* When there is no filename the language chip is the only thing in the bar,
     and `margin-left:auto` on it would shove the copy button off the left. */
  .cb-bar :global(.cb-lang:first-child) { margin-left: 0; margin-right: auto; }
</style>
