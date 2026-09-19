<!--
  /company/careers. How the work is done, the fastest way in, the planned first
  hires, and a way to raise a hand. Copy is `careers` in
  src/lib/content/company.js. The form is the site's one form component.
-->
<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import DemoForm from '$lib/components/avarok/DemoForm.svelte';
  import { careers as c } from '$lib/content/company.js';
  import { routes, links, contacts } from '$lib/content/brand.js';

  const hues = ['violet', 'cyan', 'green', 'gold'];
  const mailFor = (role) => `mailto:${contacts.technical}?subject=${encodeURIComponent(role)}`;
</script>

<PageShell path={routes.careers}>
  <PageHero eyebrow={c.eyebrow} title={c.title} lede={c.lede} primary={c.apply} secondary={{ text: 'The repository', href: links.github, external: true }} color="gold" />

  <section class="av-section av-section-alt">
    <div class="av-container">
      <div class="av-split av-split-wide av-reveal" style="align-items:start">
        <div>
          <p class="av-eyebrow av-sx-gold">{c.culture.eyebrow}</p>
          <h2 class="av-creed">{#each c.culture.lines as line}<span>{line}</span>{/each}</h2>
          <p class="av-body" style="margin-top:1.4rem;max-width:52ch">{c.culture.body}</p>
        </div>
        <div class="av-grid av-grid-2">
          {#each c.benefits as b, i}
            <div class="av-card av-lively av-sx-{hues[i]}">
              <span class="av-lively-n av-mono" aria-hidden="true">0{i + 1}</span>
              <h3>{b.title}</h3>
              <p>{b.body}</p>
            </div>
          {/each}
        </div>
      </div>
    </div>
  </section>

  <section class="av-section av-section-tight">
    <div class="av-container">
      <div class="av-band av-reveal">
        <div class="av-band-grid">
          <div>
            <p class="av-eyebrow">{c.fastTrack.eyebrow}</p>
            <h2 class="av-h2">{c.fastTrack.title}</h2>
            <p class="av-lede">{c.fastTrack.body}</p>
          </div>
          <div class="av-row" style="justify-content:flex-end">
            <a class="av-btn av-btn-primary av-btn-lg" href={c.fastTrack.primary.href} target="_blank" rel="noopener">{c.fastTrack.primary.text} <span class="av-arrow">↗</span></a>
            <a class="av-btn av-btn-ghost av-btn-lg" href={c.fastTrack.secondary.href} target="_blank" rel="noopener">{c.fastTrack.secondary.text}</a>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="av-section" id="roles">
    <div class="av-container">
      <div class="av-head-split av-reveal">
        <div><p class="av-eyebrow">{c.rolesTitle}</p><h2 class="av-h2">{c.rolesHeading}</h2></div>
        <p class="av-lede" style="max-width:40ch">{c.how}</p>
      </div>
      <div class="av-accordion av-reveal">
        {#each c.roles as r}
          <details>
            <summary><span>{r.title} <span class="av-small" style="font-weight:500;margin-left:0.6rem">{r.location} · {r.team}</span></span><span class="av-plus" aria-hidden="true">+</span></summary>
            <div class="av-answer">
              <p>{r.body}</p>
              <div class="av-role-more">
                <div>
                  <p class="av-card-tag">The work</p>
                  <ul class="av-list-check av-sx-green">{#each r.does as d}<li>{d}</li>{/each}</ul>
                </div>
                <div>
                  <p class="av-card-tag">What gets our attention</p>
                  <p>{r.signal}</p>
                </div>
              </div>
              <div class="av-row" style="margin-top:1.1rem">
                <a class="av-btn av-btn-secondary av-btn-sm" href={mailFor(r.title)}>Apply by email</a>
                <a class="av-btn av-btn-ghost av-btn-sm" href={c.apply.href}>Or use the form</a>
              </div>
            </div>
          </details>
        {/each}
      </div>
    </div>
  </section>

  <section class="av-section av-section-alt" id="apply" style="scroll-margin-top:80px">
    <div class="av-container av-narrow av-reveal">
      <DemoForm form={c.form} source="careers" to={contacts.technical} subject={(v) => `Avarok careers, ${v.role || ''}, ${v.name || ''}`} />
    </div>
  </section>

  <CtaBand title="Do not see your role?" body="Send the note anyway. A pull request against the engine is the best introduction there is." primary={c.apply} secondary={{ text: 'Good first issues', href: `${links.github}/labels/good%20first%20issue`, external: true }} />
</PageShell>

<style>
  /* The creed: four short lines, set large, each a step quieter than the last. */
  .av-creed { display: grid; gap: 0.35rem; font-size: clamp(1.5rem, 2.6vw, 2.15rem); line-height: 1.16; letter-spacing: -0.03em; }
  .av-creed span:nth-child(2) { color: var(--t2); }
  .av-creed span:nth-child(3) { color: var(--t2); opacity: 0.85; }
  .av-creed span:nth-child(4) { color: var(--ch-gold-text); }
  .av-role-more { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 1.5rem; margin-top: 1.1rem; padding-top: 1.1rem; border-top: 1px dashed var(--border-strong); }
  @media (max-width: 720px) { .av-role-more { grid-template-columns: minmax(0, 1fr); } }
</style>
