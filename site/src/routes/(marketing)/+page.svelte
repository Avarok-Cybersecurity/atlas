<script>
  // The front page. Sections in the order a buyer reads them; every section
  // reads its copy from src/lib/content/home.js and ends where the next one
  // starts. Old bookmarks into the developer sections (#faq, #hardware) still
  // resolve: the fragment is forwarded to /engine, the same way the previous
  // homepage did it.
  import { afterNavigate } from '$app/navigation';
  import { legacyEngineDestination } from '$lib/marketing.js';
  import { routes } from '$lib/content/brand.js';
  import { faqFor } from '$lib/content/faq.js';
  import { cta } from '$lib/content/home.js';
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import Announcement from '$lib/components/avarok/home/Announcement.svelte';
  import Hero from '$lib/components/avarok/home/Hero.svelte';
  import LogoWall from '$lib/components/avarok/LogoWall.svelte';
  import Problem from '$lib/components/avarok/home/Problem.svelte';
  import Solution from '$lib/components/avarok/home/Solution.svelte';
  import ValueBand from '$lib/components/avarok/home/ValueBand.svelte';
  import Proof from '$lib/components/avarok/home/Proof.svelte';
  import Tour from '$lib/components/avarok/home/Tour.svelte';
  import Recognition from '$lib/components/avarok/home/Recognition.svelte';
  import Differences from '$lib/components/avarok/home/Differences.svelte';
  import Chain from '$lib/components/avarok/home/Chain.svelte';
  import Deliveries from '$lib/components/avarok/home/Deliveries.svelte';
  import Voices from '$lib/components/avarok/home/Voices.svelte';
  import FaqList from '$lib/components/avarok/FaqList.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';

  function forward({ hash, search }) {
    const destination = legacyEngineDestination(hash, search);
    if (destination) window.location.replace(destination);
  }
  afterNavigate(({ to }) => { if (to) forward(to.url); });
</script>

<svelte:window onhashchange={() => forward(window.location)} />

<PageShell path={routes.home}>
  <Announcement />
  <Hero />
  <LogoWall />
  <Problem />
  <Solution />
  <ValueBand />
  <Proof />
  <Tour />
  <Recognition />
  <Differences />
  <Chain />
  <Deliveries />
  <Voices />
  <FaqList items={faqFor('home')} />
  <CtaBand eyebrow={cta.eyebrow} title={cta.title} body={cta.body} primary={cta.primary} secondary={cta.secondary} />
</PageShell>
