<script>
  // The "reproduction steps" panel of a point card: the pipeline that produced
  // one record, in execution order, each step naming its recorded evidence.
  //
  // Every string here comes from `reproSteps()` — this component builds no
  // prose and no command of its own, so what a diligence reader copies is
  // exactly what the pure module (and its tests) say the record contains.
  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';
  import { gateData } from '$lib/gates.js';
  import { reproSteps } from '$lib/repro-steps.js';

  let { record } = $props();

  // The generated data's own provenance and the descriptor/limits it scanned.
  // An absent descriptor or limit is passed as null so the plan NAMES it
  // rather than quoting a number the generator never found.
  const plan = $derived(
    reproSteps(record, {
      generated: { sha: gateData.generated_sha, date: gateData.generated_date },
      meta: gateData.registered_meta?.[record.benchmark_id] ?? null,
      serve_allowance_s: gateData.limits?.serve_allowance_s ?? null
    })
  );

  // Independent toggles rather than an exclusive `name=` accordion: "expand
  // all" is a real need for a reader printing the page, and the two cannot
  // coexist.
  let open = $state({});
  const setAll = (v) => {
    for (const s of plan.steps) open[s.id] = v;
  };

  // One copy state per block, keyed by step and command index. `idle` is the
  // state before any attempt; a refusal is reported, never rendered as
  // success (see clipboard.js).
  let copyState = $state({});
  let pres = $state({});
  const timers = {};
  $effect(() => () => Object.values(timers).forEach(clearTimeout));
  async function copy(key, text, el) {
    clearTimeout(timers[key]);
    copyState[key] = await copyOrSelect(text, el);
    timers[key] = setTimeout(() => (copyState[key] = 'idle'), 2400);
  }
  const isLink = (v) => /^https?:\/\//.test(v);
</script>

<section class="gpc-repro" id="gpc-repro" aria-label="Reproduction steps">
  <h3 class="gpc-repro-h" tabindex="-1">Reproduction steps</h3>
  <p class="gpc-repro-headline">{plan.headline}</p>

  <div class="gpc-repro-actions">
    <button type="button" class="gpc-copy" onclick={() => setAll(true)}>expand all</button>
    <button type="button" class="gpc-copy" onclick={() => setAll(false)}>collapse all</button>
    <button type="button" class="gpc-copy" aria-live="polite" onclick={() => copy('all', plan.script, null)}>
      {copyLabel(copyState.all ?? 'idle', 'copy all steps').toLowerCase()}
    </button>
  </div>

  <dl class="gpc-rows gpc-preface">
    <dt>required</dt>
    <dd>{plan.preface.required}</dd>
    <dt>costs</dt>
    <dd>{plan.preface.costs}</dd>
    <dt>expect</dt>
    <dd>{#each plan.preface.expect as line}<span class="gpc-expect-line">{line}</span>{/each}</dd>
  </dl>

  {#if plan.missing.length}
    <ul class="gpc-missing" aria-label="Not recorded">
      {#each plan.missing as m}
        <li><b>not recorded: {m.field}</b> — {m.need}</li>
      {/each}
    </ul>
  {/if}
  {#if plan.caveats.length}
    <ul class="gpc-caveats" aria-label="Caveats">
      {#each plan.caveats as c}<li>{c}</li>{/each}
    </ul>
  {/if}

  <ol class="gpc-steps">
    {#each plan.steps as s, i (s.id)}
      <li>
        <details class="gpc-step" bind:open={open[s.id]}>
          <summary>
            <span class="gpc-step-n">{i + 1}</span>
            <span class="gpc-step-title">{s.title}</span>
            <span class="gpc-step-sum">{s.summary}</span>
          </summary>
          {#if s.facts.length}
            <dl class="gpc-rows gpc-step-facts">
              {#each s.facts as [k, v]}
                <dt>{k}</dt>
                <dd>
                  {#if isLink(v)}
                    <a href={v} target="_blank" rel="noopener">{v.replace(/^https:\/\/github\.com\//, '')}</a>
                  {:else}{v}{/if}
                </dd>
              {/each}
            </dl>
          {/if}
          {#each s.commands as c, j}
            {@const key = `${s.id}-${j}`}
            <figure class="gpc-cmd">
              <figcaption>
                <span class="gpc-cmd-label">{c.label}</span>
                <button type="button" class="gpc-copy" aria-live="polite" onclick={() => copy(key, c.lines.join('\n'), pres[key])}>
                  {copyLabel(copyState[key] ?? 'idle', 'copy').toLowerCase()}
                </button>
              </figcaption>
              <!-- `tabindex` because the argv scrolls horizontally and a scrollable
                   region a keyboard cannot reach is content a keyboard cannot read;
                   `role`/`aria-label` name the stop (same reason as CommandRow). -->
              <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
              <pre tabindex="0" role="group" aria-label="Command, scrollable" bind:this={pres[key]}>{#each c.lines as line, k}<span
                    class="gpc-cmd-line"
                    class:gpc-derived={c.derivedLines.includes(k)}
                    title={c.derivedLines.includes(k) ? 'added by this page, not in the record' : undefined}
                  >{line}</span>{/each}</pre>
            </figure>
          {/each}
          {#each s.notes as n}<p class="gpc-note">{n}</p>{/each}
        </details>
      </li>
    {/each}
  </ol>
</section>
