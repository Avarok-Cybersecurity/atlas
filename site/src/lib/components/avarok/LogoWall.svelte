<!--
  "Built by people who have stood up operations for". Company marks come from
  static/logos (public domain vector masters from Wikimedia Commons, see
  static/logos/README.md). Government entities render as set text because
  their seals are restricted insignia. The note under the wall says exactly
  what the wall is.
-->
<script>
  import { logoWall } from '$lib/content/home.js';
  let { compact = false } = $props();
</script>

<section class="av-section av-section-tight av-wall" class:is-compact={compact}>
  <div class="av-container">
    <p class="av-wall-label av-reveal">{logoWall.label}</p>
    <ul class="av-logo-wall av-reveal">
      {#each logoWall.items as it}
        <li class="av-logo" title={it.name}>
          {#if it.kind === 'text'}
            <span class="av-logo-text">{it.name}</span>
          {:else}
            <img src={`/logos/${it.file}.svg`} alt={it.name} class="is-dark-invert" height="30" width="120" loading="lazy" />
          {/if}
        </li>
      {/each}
    </ul>
    <p class="av-wall-note">{logoWall.note}</p>
    {#if !compact}
      <p class="av-wall-label av-reveal" style="margin-top:2.6rem">{logoWall.programsLabel}</p>
      <ul class="av-programs av-reveal">
        {#each logoWall.programs as p}
          <li>
            <svelte:element this={p.href ? 'a' : 'div'} href={p.href} target={p.href ? '_blank' : undefined} rel={p.href ? 'noopener' : undefined} class="av-program">
              {#if p.src || p.file}
                <img src={p.src ?? `/logos/${p.file}.svg`} alt={p.name} height="34" width="150" loading="lazy" class="is-dark-invert" />
              {:else}
                <span class="av-logo-text">{p.name}</span>
              {/if}
              <span class="av-program-b">{p.blurb}</span>
            </svelte:element>
          </li>
        {/each}
      </ul>
    {/if}
  </div>
</section>

<style>
  .av-wall { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
  .av-programs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; }
  .av-program { display: grid; gap: 0.45rem; padding: 1rem 1.1rem; border: 1px solid var(--border); border-radius: var(--av-radius-sm); background: var(--card); text-decoration: none; color: inherit; transition: border-color 0.15s; }
  a.av-program:hover { border-color: var(--accent); }
  .av-program img { height: 26px; width: auto; }
  .av-program-b { font-size: 0.8rem; color: var(--t3); line-height: 1.45; }
  @media (max-width: 900px) { .av-programs { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 560px) { .av-programs { grid-template-columns: minmax(0, 1fr); } }
</style>
