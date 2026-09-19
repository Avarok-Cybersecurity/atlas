<script>
  // The b-roll render page. With ?scene=<name> it runs that loop full screen
  // for the recorder; without one it lists the loops so a person can preview
  // them. Prerendered in the list state, noindex, not in the sitemap.
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import Broll from '$lib/broll/Broll.svelte';
  import { LOOP, scenes } from '$lib/broll/scenes.js';
  import { pages, routes } from '$lib/content/index.js';

  const meta = pages.find((p) => p.path === routes.broll);
  let scene = $state(null);
  onMount(() => {
    const s = page.url.searchParams.get('scene');
    scene = s && scenes[s] ? s : null;
  });
</script>

<svelte:head>
  <title>{meta.title}</title>
  <meta name="description" content={meta.description} />
  <meta name="robots" content="noindex" />
</svelte:head>

{#if scene}
  <Broll {scene} />
{:else}
  <main class="av broll-index">
    <h1>B-roll loops</h1>
    <p>Procedural ambient clips, {LOOP} seconds each, drawn from the brand palette. The media pipeline records these pages; nothing links here from the site.</p>
    <ul>
      {#each Object.entries(scenes) as [name, def]}
        <li><a href={`${routes.broll}?scene=${name}`}>{name}</a> <span>{def.alt}</span></li>
      {/each}
    </ul>
  </main>
{/if}

<style>
  .broll-index { max-width: 640px; margin: 0 auto; padding: 4rem 1.5rem; font-family: var(--font-sans); color: var(--t2); }
  h1 { color: var(--t1); font-size: 1.6rem; margin: 0 0 0.6rem; }
  ul { list-style: none; padding: 0; margin: 1.5rem 0 0; display: grid; gap: 0.8rem; }
  li { display: grid; gap: 0.15rem; }
  a { color: var(--accent); font-weight: 600; text-decoration: none; }
  span { color: var(--t3); font-size: 0.92rem; }
</style>
