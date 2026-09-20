<!--
  /waitlist. The Community Edition is announced, not released, so every button
  that used to say "install" comes here. Same shape as /demo on purpose: what
  you get on the left, the form on the right, and the closing band loops back
  to the demo. Copy is `waitlistPage` in src/lib/content/company.js.
-->
<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import DemoForm from '$lib/components/avarok/DemoForm.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { waitlistPage as w } from '$lib/content/company.js';
  import { routes, links, contacts } from '$lib/content/brand.js';
</script>

<PageShell path={routes.waitlist}>
  <PageHero eyebrow={w.eyebrow} title={w.title} lede={w.lede} />
  <section class="av-section av-section-tight">
    <div class="av-container">
      <h2 class="av-sr">What joining gets you</h2>
      <div class="av-split av-split-wide" style="align-items:start">
        <div class="av-stack av-reveal" style="gap:1.5rem">
          <ul class="av-list-check av-sx-green">{#each w.bullets as b}<li>{b}</li>{/each}</ul>
          <div class="av-card">
            <h3>{w.today.title}</h3>
            <p>{w.today.body}</p>
            <p style="margin-top:0.6rem"><a class="av-link" href={w.today.cta.href}>{w.today.cta.text} <span class="av-arrow">→</span></a></p>
            <p style="margin-top:0.6rem">{w.today.discord} <a class="av-link" href={links.discord} target="_blank" rel="noopener">Join ↗</a></p>
          </div>
        </div>
        <div class="av-reveal" id="join" style="scroll-margin-top:96px">
          <DemoForm form={w.form} source="waitlist" to={contacts.sales} subject={(v) => `Community Edition waitlist, ${v.name || v.email || ''}`} />
        </div>
      </div>
    </div>
  </section>
  <CtaBand title={w.cta.title} body={w.cta.body} primary={w.cta.primary} secondary={w.cta.secondary} />
</PageShell>
