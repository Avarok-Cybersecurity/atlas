<script>
  import { copyText } from '$lib/clipboard.js';
  // A command an analyst is expected to paste. Copy is the point of the
  // component: a step someone has to retype by eye is a step they will get
  // wrong and then report as a failure to reproduce.
  let { label = '', lines = [], note = '' } = $props();

  let copied = $state(false);
  const text = $derived(lines.join('\n'));

  async function copy() {
    if ((await copyText(text)) !== 'copied') return;
    copied = true;
    setTimeout(() => (copied = false), 1600);
  }
</script>

<figure class="cmd">
  <figcaption>
    {#if label}<span class="cmd-label mono">{label}</span>{/if}
    <button type="button" class="cmd-copy mono" onclick={copy}>{copied ? 'copied' : 'copy'}</button>
  </figcaption>
  <pre class="mono">{#each lines as line}<span class="cmd-line">{line}</span>{/each}</pre>
  {#if note}<p class="cmd-note">{note}</p>{/if}
</figure>

<style>
  .cmd {
    border: 1px solid var(--border);
    border-left: 2px solid var(--sx);
    border-radius: 6px;
    background: var(--bg2);
    overflow: hidden;
  }
  figcaption {
    display: flex;
    align-items: center;
    gap: 1em;
    padding: 0.45em 0.8em;
    border-bottom: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.18);
  }
  .cmd-label {
    flex: 1;
    font-size: 0.72em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--t3);
  }
  .cmd-copy {
    background: none;
    border: 1px solid var(--border-strong);
    color: var(--t2);
    border-radius: 4px;
    font-size: 0.68em;
    padding: 0.1em 0.5em;
    cursor: pointer;
  }
  .cmd-copy:hover {
    border-color: var(--sx);
    color: var(--sx);
  }
  pre {
    padding: 0.75em 0.9em;
    font-size: 0.8em;
    line-height: 1.75;
    color: var(--t1);
    overflow-x: auto;
    white-space: pre;
  }
  .cmd-line {
    display: block;
  }
  /* A blank line in a command block is a paragraph break between stages, so it
     has to occupy a line rather than collapsing to nothing. */
  .cmd-line:empty::before {
    content: '\00a0';
  }
  .cmd-note {
    padding: 0 0.9em 0.7em;
    font-size: 0.78em;
    color: var(--t3);
  }
</style>
