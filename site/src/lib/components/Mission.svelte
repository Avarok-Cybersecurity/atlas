<script>
  // Not a ledger entry: a quote band. The mission was a 7rem section carrying
  // three paragraphs that made one claim; it now makes that claim once, at
  // size, over the mark. Everything numbered on this page is evidence — this
  // is the one place the page is allowed to simply state something.
  import { mission } from '$lib/data.js';
</script>

<section id="mission" class="quote-band sx-violet">
  <svg class="quote-mark" viewBox="0 0 956 636" width="956" height="636" aria-hidden="true">
    <g fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round"
       stroke="currentColor">
      <path d="M38 38L358 318L38 598" />
      <path d="M318 38L638 318L318 598" />
      <path d="M598 38L918 318L598 598" />
    </g>
  </svg>
  <div class="container">
    <h2 class="quote-title">{mission.title}</h2>
    <p class="quote-statement">{mission.statement}</p>
    <p class="quote-foot">{mission.footnote}</p>
  </div>
</section>
