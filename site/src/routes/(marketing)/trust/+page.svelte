<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import FaqList from '$lib/components/avarok/FaqList.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { trust as t } from '$lib/content/company.js';
  import { faqFor } from '$lib/content/faq.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.trust}>
  <PageHero eyebrow={t.eyebrow} title={t.title} lede={t.lede} primary={{ text: t.disclosure.cta, href: t.disclosure.href, external: true }} secondary={{ text: 'Security overview', href: routes.security }} color="cyan" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <div class="av-grid av-grid-2 av-reveal">
        {#each t.sections as s, i}
          <div class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i]}">
            <h3>{s.title}</h3>
            <ul class="av-list-check" style="margin-top:0.8rem">{#each s.items as it}<li>{it}</li>{/each}</ul>
          </div>
        {/each}
      </div>
      <div class="av-bridge av-reveal">
        <h3>{t.disclosure.title}</h3>
        <p>{t.disclosure.body}</p>
      </div>
    </div>
  </section>
  <FaqList items={faqFor('security')} />
  <CtaBand title="Send the security questionnaire." body="We answer it the way this page is written, control by control, with what we run and what we do not." primary={{ text: 'Book a security review', href: routes.demo }} secondary={{ text: 'Contact', href: routes.contact }} />
</PageShell>
