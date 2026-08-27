<script>
  // A self-audit table: each row is a known way to fake a benchmark, and what
  // this campaign did about it. `state` is 'clear' (addressed), 'open' (a real
  // limitation we are not claiming past) or 'note'.
  //
  // The point of rendering the open rows in the same table as the clear ones is
  // that a checklist with no open rows is not a checklist, it is marketing.
  let { rows = [] } = $props();

  const glyph = { clear: '✓', open: '!', note: '·' };
</script>

<ul class="au">
  {#each rows as row, i}
    <li class="au-row at" data-state={row.state} style="--n: {Math.floor(i / 4) + 1}">
      <span class="au-g" aria-hidden="true">{glyph[row.state]}</span>
      <span class="au-risk">{row.risk}</span>
      <span class="au-ans">{row.answer}</span>
    </li>
  {/each}
</ul>

<style>
  .au {
    list-style: none;
    display: grid;
    gap: 0;
  }
  .au-row {
    display: grid;
    grid-template-columns: 1.6em 22ch 1fr;
    gap: 0.9em;
    align-items: baseline;
    padding: 0.3em 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.78em;
  }
  .au-g {
    font-weight: 700;
    text-align: center;
  }
  .au-row[data-state='clear'] .au-g {
    color: var(--green);
  }
  .au-row[data-state='open'] .au-g {
    color: var(--amber);
  }
  .au-row[data-state='note'] .au-g {
    color: var(--t3);
  }
  .au-risk {
    color: var(--t2);
  }
  .au-ans {
    color: var(--t1);
  }
</style>
