<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import LogoWall from '$lib/components/avarok/LogoWall.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { story, mission, showTeam, team, companyCta } from '$lib/content/company.js';
  import { routes, links } from '$lib/content/brand.js';
  import { live } from '$lib/content/live.js';
</script>

<PageShell path={routes.company}>
  <PageHero eyebrow={story.eyebrow} title={story.title} lede={story.lede} primary={{ text: 'Read the thread', href: links.llamaCppPr, external: true }} secondary={{ text: 'Open roles', href: routes.careers }} />

  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">How it started</h2>
      <div class="av-split av-split-wide" style="align-items:start">
        <div class="av-reveal av-yourpoint">
          <p class="av-eyebrow">The exchange</p>
          <div class="av-card av-thread">
            <p class="av-thread-line"><span class="av-mono">XMR13</span> why are you so aggressive? Your PR looks completely 100% AI generated</p>
            <p class="av-thread-line is-reply"><span class="av-mono">tbraun96</span> Your point?</p>
          </div>
          <p class="av-small" style="margin-top:0.8rem">llama.cpp pull request 18680, January 7 and 8, 2026. <a class="av-link" href={links.llamaCppPr} target="_blank" rel="noopener">The whole thread ↗</a></p>
          <p class="av-body" style="margin-top:1.5rem">{live.stars} stars later, the repository those two words started runs on hardware from NVIDIA and AMD, merged a kernel into Hugging Face Transformers, and out serves the incumbent on the published ladder. The answer to the question is this website.</p>
        </div>
        <div class="av-timeline av-reveal">
          {#each story.beats as b}
            <div class="av-beat">
              <p class="av-beat-when">{b.when}</p>
              <h3>{b.title}</h3>
              <p>{b.body}</p>
              {#if b.href}<a class="av-link" style="margin-top:0.5rem" href={b.href} target="_blank" rel="noopener">{b.cta} ↗</a>{/if}
            </div>
          {/each}
        </div>
      </div>
    </div>
  </section>

  <section class="av-section">
    <div class="av-container">
      <div class="av-head av-center av-reveal">
        <p class="av-eyebrow" style="justify-content:center">Mission</p>
        <h2 class="av-h2">{mission.title}</h2>
        <p class="av-lede">{mission.body}</p>
      </div>
      <div class="av-grid av-grid-4 av-reveal">
        {#each mission.principles as p, i}
          <div class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i]}"><h3>{p.title}</h3><p>{p.body}</p></div>
        {/each}
      </div>
    </div>
  </section>

  {#if showTeam}
    <section class="av-section av-section-alt" id="team">
      <div class="av-container">
        <div class="av-head av-reveal">
          <p class="av-eyebrow">{team.eyebrow}</p>
          <h2 class="av-h2">{team.title}</h2>
        </div>
        <div class="av-people av-reveal">
          {#each team.people as p}
            <div class="av-person">
              <div class="av-avatar" aria-hidden="true">{p.name.split(' ').map((n) => n[0]).join('')}</div>
              <h3 class="av-h4">{p.name}</h3>
              <p class="av-role">{p.role} · {p.focus}</p>
              <p class="av-body" style="font-size:0.92rem">{p.bio}</p>
            </div>
          {/each}
        </div>
      </div>
    </section>
  {/if}

  <LogoWall />
  <CtaBand eyebrow={companyCta.eyebrow} title={companyCta.title} body={companyCta.body} primary={companyCta.primary} secondary={companyCta.secondary} />
</PageShell>

<style>
  .av-thread { display: grid; gap: 0.8rem; }
  .av-thread-line { padding: 0.9rem 1rem; border-radius: var(--av-radius-sm); background: var(--bg2); border: 1px solid var(--border); font-size: 0.98rem; color: var(--t1); }
  .av-thread-line .av-mono { display: block; font-size: 0.72rem; color: var(--t3); margin-bottom: 0.3rem; }
  .av-thread-line.is-reply { border-color: var(--accent); font-size: 1.35rem; font-weight: 600; letter-spacing: -0.02em; }
  .av-yourpoint { position: sticky; top: 100px; }
  @media (max-width: 900px) { .av-yourpoint { position: static; margin-bottom: 2rem; } }
</style>
