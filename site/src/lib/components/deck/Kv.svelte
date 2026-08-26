<script>
  // Fingerprint rows. `rows` is [label, value, note?] — the note carries the
  // reason a value is what it is, which is the part an analyst is actually
  // auditing.
  //
  // `mark` renders a parity column: true means the axis is pinned identically
  // on both engines, false means it is deliberately not, and the value column
  // has to say why.
  let { rows = [], mark = false, cols = 1 } = $props();
</script>

<dl class="kv" class:kv-two={cols === 2}>
  {#each rows as [label, value, note], i}
    <div class="kv-row at" style="--n: {i + 1}">
      {#if mark}<span class="kv-mark" aria-hidden="true">{note === false ? '·' : '='}</span>{/if}
      <dt class="mono">{label}</dt>
      <dd>
        <span class="kv-val mono">{value}</span>
        {#if note && note !== true}<span class="kv-note">{note}</span>{/if}
      </dd>
    </div>
  {/each}
</dl>

<style>
  .kv {
    display: grid;
    gap: 0.15em;
    align-content: start;
  }
  .kv-two {
    grid-template-columns: 1fr 1fr;
    column-gap: 2.2em;
  }
  .kv-row {
    display: grid;
    grid-template-columns: 13ch 1fr;
    gap: 0.9em;
    padding: 0.42em 0;
    border-bottom: 1px solid var(--border);
    align-items: baseline;
  }
  .kv-row:has(.kv-mark) {
    grid-template-columns: 1.2em 13ch 1fr;
  }
  .kv-mark {
    color: var(--sx);
    font-weight: 700;
  }
  dt {
    font-size: 0.76em;
    letter-spacing: 0.04em;
    color: var(--t3);
    text-transform: uppercase;
  }
  dd {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2em 0.7em;
    align-items: baseline;
  }
  .kv-val {
    color: var(--t1);
    font-size: 0.92em;
  }
  .kv-note {
    color: var(--t3);
    font-size: 0.8em;
  }
</style>
