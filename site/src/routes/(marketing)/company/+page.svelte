<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import LogoWall from '$lib/components/avarok/LogoWall.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { story, mission, showTeam, team, exchange, companyCta } from '$lib/content/company.js';
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
          <p class="av-eyebrow">{exchange.eyebrow}</p>
          <figure class="av-shot">
            <a href={exchange.link.href} target="_blank" rel="noopener" aria-label="Open the pull request thread on GitHub">
              <img class="only-light" src={exchange.image.light} alt={exchange.alt} width={exchange.image.width} height={exchange.image.height} loading="lazy" decoding="async" />
              <img class="only-dark" src={exchange.image.dark} alt={exchange.alt} width={exchange.image.width} height={exchange.image.height} loading="lazy" decoding="async" />
            </a>
            <details>
              <summary>{exchange.before.summary}</summary>
              <img class="only-light" src={exchange.before.image.light} alt={exchange.before.alt} width={exchange.before.image.width} height={exchange.before.image.height} loading="lazy" decoding="async" />
              <img class="only-dark" src={exchange.before.image.dark} alt={exchange.before.alt} width={exchange.before.image.width} height={exchange.before.image.height} loading="lazy" decoding="async" />
            </details>
            <figcaption class="av-small">{exchange.caption} <a class="av-link" href={exchange.link.href} target="_blank" rel="noopener">{exchange.link.text} ↗</a></figcaption>
          </figure>
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
        <div class="av-team av-reveal">
          {#each team.people as p}
            <article class="av-member av-sx-{p.hue}">
              <img class="av-member-photo" src={`/team/${p.photo}.webp`} alt="" width="320" height="320" loading="lazy" decoding="async" />
              <div>
                <h3 class="av-h4">{p.name}</h3>
                <p class="av-member-role">{p.role} <span aria-hidden="true">·</span> {p.focus}</p>
              </div>
              <p class="av-member-bio">{p.bio}</p>
              <a class="av-member-in" href={p.linkedin} target="_blank" rel="noopener" aria-label={`${p.name} on LinkedIn`}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0Z"/></svg>
                LinkedIn <span aria-hidden="true">↗</span>
              </a>
            </article>
          {/each}
        </div>
        <p class="av-deck av-reveal">
          <span>{team.deck.body}</span>
          {#if team.deck.file}
            <a class="av-btn av-btn-secondary av-btn-sm" href={team.deck.file} download>{team.deck.download} <span class="av-arrow">↓</span></a>
          {:else}
            <a class="av-btn av-btn-secondary av-btn-sm" href={team.deck.request.href}>{team.deck.request.text} <span class="av-arrow">→</span></a>
          {/if}
        </p>
      </div>
    </section>
  {/if}

  <LogoWall />
  <CtaBand eyebrow={companyCta.eyebrow} title={companyCta.title} body={companyCta.body} primary={companyCta.primary} secondary={companyCta.secondary} />
</PageShell>

<style>
  /* The real thread, as GitHub drew it. One picture per theme, and the hidden one
     is display:none with loading=lazy, so only the one in use is fetched. */
  .av-shot { margin: 0; display: grid; gap: 0.7rem; }
  .av-shot > a, .av-shot details img { display: block; border: 1px solid var(--border-strong); border-radius: var(--av-radius-sm); overflow: hidden; box-shadow: var(--av-shadow); background: var(--card); }
  .av-shot img { width: 100%; height: auto; }
  .av-shot > a { transition: transform 0.2s var(--av-ease), border-color 0.2s; }
  .av-shot > a:hover { transform: translateY(-2px); border-color: var(--accent); }
  .av-shot summary { cursor: pointer; font-size: 0.86rem; font-weight: 600; color: var(--accent); padding: 0.2rem 0; }
  .av-shot details[open] summary { margin-bottom: 0.6rem; }
  .only-dark { display: none; }
  :global(html:not([data-theme='light'])) .only-dark { display: block; }
  :global(html:not([data-theme='light'])) .only-light { display: none; }

  .av-team { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.1rem; }
  .av-member {
    position: relative; isolation: isolate; overflow: hidden; display: grid; gap: 0.8rem; grid-template-rows: auto auto 1fr auto;
    padding: 1.4rem 1.3rem 1.2rem; background: var(--card); border: 1px solid var(--border); border-radius: var(--av-radius);
    transition: transform 0.22s var(--av-ease), border-color 0.22s, box-shadow 0.22s;
  }
  .av-member::before { content: ''; position: absolute; z-index: -1; right: -35%; top: -45%; width: 90%; aspect-ratio: 1; background: radial-gradient(closest-side, color-mix(in srgb, var(--sx) 30%, transparent), transparent); opacity: 0.5; transition: opacity 0.3s; }
  .av-member:hover { transform: translateY(-3px); border-color: color-mix(in srgb, var(--sx) 65%, var(--border)); box-shadow: var(--av-shadow); }
  .av-member:hover::before { opacity: 1; }
  .av-member-photo { width: 76px; height: 76px; border-radius: 50%; object-fit: cover; box-shadow: 0 0 0 2px var(--card), 0 0 0 4px var(--sx); }
  .av-member-role { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--sx-text); margin-top: 0.3rem; line-height: 1.5; }
  .av-member-bio { font-size: 0.9rem; line-height: 1.55; color: var(--t2); }
  .av-member-in { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.84rem; font-weight: 600; color: var(--t2); text-decoration: none; justify-self: start; padding: 0.15rem 0; }
  .av-member-in:hover { color: var(--accent); }
  .av-deck { display: flex; align-items: center; justify-content: space-between; gap: 1rem 1.5rem; flex-wrap: wrap; margin-top: 1.6rem; padding: 1rem 1.2rem; border: 1px dashed var(--border-strong); border-radius: var(--av-radius-sm); color: var(--t2); font-size: 0.92rem; }
  .av-yourpoint { position: sticky; top: 100px; }
  @media (max-width: 900px) { .av-yourpoint { position: static; margin-bottom: 2rem; } }
</style>
