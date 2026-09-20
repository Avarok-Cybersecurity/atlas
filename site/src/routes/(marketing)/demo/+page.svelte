<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import DemoForm from '$lib/components/avarok/DemoForm.svelte';
  import VideoClip from '$lib/components/avarok/VideoClip.svelte';
  import { demoPage as d } from '$lib/content/company.js';
  import { media } from '$lib/content/media.js';
  import { routes, links, contacts } from '$lib/content/brand.js';
  import { onMount } from 'svelte';

  // Calls to action across the site land on /demo#book, the form itself. The
  // browser scrolls there. This puts the caret in the first field as well, so
  // "book a demo" ends with the visitor typing, not looking for where to start.
  onMount(() => {
    const focusForm = () => {
      if (location.hash !== '#book') return;
      document.querySelector('#book input')?.focus({ preventScroll: true });
    };
    focusForm();
    addEventListener('hashchange', focusForm);
    return () => removeEventListener('hashchange', focusForm);
  });
</script>

<PageShell path={routes.demo}>
  <PageHero eyebrow={d.eyebrow} title={d.title} lede={d.lede} />
  <section class="av-section av-section-tight">
    <div class="av-container">
      <h2 class="av-sr">What the session covers</h2>
      <div class="av-split av-split-wide" style="align-items:start">
        <div class="av-stack av-reveal" style="gap:1.5rem">
          <ul class="av-list-check av-sx-green">{#each d.bullets as b}<li>{b}</li>{/each}</ul>
          <!-- The film, once, with controls. It fetches nothing until it is in view. -->
          <div id="film" class="av-frame" style="scroll-margin-top:96px"><div><VideoClip clip={media.reel} controls /></div></div>
          <p class="av-video-caption" style="margin-top:-0.6rem">One minute. Product footage is recorded from the product mockup on demo data.</p>
          <div class="av-card">
            <h3>{d.aside.title}</h3>
            <p>{d.aside.body}</p>
            <p style="margin-top:0.6rem"><a class="av-link" href={`mailto:${contacts.sales}`}>{contacts.sales}</a></p>
            <p style="margin-top:0.6rem">{d.aside.discord} <a class="av-link" href={links.discord} target="_blank" rel="noopener">Join ↗</a></p>
          </div>
        </div>
        <div class="av-reveal" id="book" style="scroll-margin-top:96px"><DemoForm /></div>
      </div>
    </div>
  </section>
</PageShell>
