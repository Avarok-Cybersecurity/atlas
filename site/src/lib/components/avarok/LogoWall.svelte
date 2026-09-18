<!--
  "Built by people who have stood up operations for", and under it the programs
  and partners. Every file is self hosted under static/logos, and
  static/logos/README.md records where each one came from and on what terms.

  Three kinds of item, all data in `logoWall` (src/lib/content/home.js):
    file     a company wordmark. Greyed, and inverted on the dark theme, so a
             wall of different brands reads as one row.
    emblem   an official round emblem, in its own colours, with the
             organisation's name set beside it on two lines.
    text     the name alone, set in type. Setting `logoWall.emblems` to false
             turns every emblem back into this, in one edit.

  Programs show a partner's own logo in its own colours, through BrandMark.svelte,
  which the recognition cards on the front page use as well.
-->
<script>
  import { logoWall } from '$lib/content/home.js';
  import BrandMark from './BrandMark.svelte';
  let { compact = false } = $props();
</script>

<section class="av-section av-section-tight av-wall" class:is-compact={compact}>
  <div class="av-container">
    <p class="av-wall-label av-reveal">{logoWall.label}</p>
    <ul class="av-logo-wall av-reveal" class:has-emblems={logoWall.emblems}>
      {#each logoWall.items as it}
        {#if it.emblem && logoWall.emblems}
          <li class="av-logo has-emblem" title={it.name}>
            <img src={`/logos/${it.emblem}.webp`} alt="" height="44" width="44" loading="lazy" />
            <span class="av-logo-text av-logo-lines">{#each it.lines ?? [it.name] as line}<span>{line}</span>{/each}</span>
          </li>
        {:else if it.file}
          <li class="av-logo" title={it.name}>
            <img src={`/logos/${it.file}.svg`} alt={it.name} class="is-dark-invert" height="30" width="120" loading="lazy" />
          </li>
        {:else}
          <li class="av-logo" title={it.name}><span class="av-logo-text">{it.short ?? it.name}</span></li>
        {/if}
      {/each}
    </ul>
    <p class="av-wall-note">{logoWall.note}</p>
    {#if !compact}
      <p class="av-wall-label av-reveal" style="margin-top:2.6rem">{logoWall.programsLabel}</p>
      <ul class="av-programs av-reveal">
        {#each logoWall.programs as p}
          <li>
            <svelte:element this={p.href ? 'a' : 'div'} href={p.href} target={p.href ? '_blank' : undefined} rel={p.href ? 'noopener' : undefined} class="av-program">
              <BrandMark mark={p} />
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
  /* With the emblems the row is too wide for one line. Narrowing it makes the
     wrap fall four and three, which reads as meant, not as one mark left over. */
  .av-logo-wall.has-emblems { max-width: 900px; margin-inline: auto; row-gap: 1.7rem; }
  .av-wall :global(.av-wall-note) { margin-top: 1.9rem; max-width: 86ch; margin-inline: auto; }
  /* An emblem keeps its colours: a greyed seal is a smudge. It is a little taller
     than a wordmark because its detail needs the pixels. */
  .av-logo.has-emblem { height: 44px; gap: 0.7rem; filter: none; opacity: 0.92; }
  .av-logo.has-emblem img { height: 44px; width: 44px; max-width: none; }
  .av-logo-lines { display: grid; gap: 0.15rem; font-size: 0.7rem; line-height: 1.15; letter-spacing: 0.1em; }
  .av-programs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; }
  .av-program { display: grid; gap: 0.6rem; align-content: start; height: 100%; padding: 1.05rem 1.15rem; border: 1px solid var(--border); border-radius: var(--av-radius-sm); background: var(--card); text-decoration: none; color: inherit; transition: border-color 0.15s, transform 0.18s var(--av-ease), box-shadow 0.18s; }
  a.av-program:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: var(--av-shadow); }
  .av-program-b { font-size: 0.8rem; color: var(--t3); line-height: 1.45; }
  @media (max-width: 900px) { .av-programs { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 560px) { .av-programs { grid-template-columns: minmax(0, 1fr); } .av-logo.has-emblem { height: 40px; } .av-logo.has-emblem img { height: 40px; width: 40px; } }
</style>
