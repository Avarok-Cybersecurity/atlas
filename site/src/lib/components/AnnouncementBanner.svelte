<script>
  import { announcement } from '$lib/data.js';
</script>

<div class="ann">
  <!-- Each row is conditional on its own content, so a piece of news can be
       taken down by emptying it in data.js rather than by editing this markup.
       The partnership row is currently empty and therefore not rendered. -->
  {#if announcement.line}
    <div class="ann-inner">
      <svg class="ann-mark" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 2 4 5v6.1c0 4.6 3.1 8.9 8 10.9 4.9-2 8-6.3 8-10.9V5l-8-3Zm0 2.2 6 2.2v5.7c0 3.6-2.4 7-6 8.7-3.6-1.7-6-5.1-6-8.7V6.4l6-2.2Zm-1 4.3v3h-3v2h3v3h2v-3h3v-2h-3v-3h-2Z"/>
      </svg>
      <span class="ann-line">{announcement.line}</span>
      <span class="ann-sub">{announcement.sub}</span>
      {#if announcement.ctaUrl}
        <a class="ann-cta" href={announcement.ctaUrl} target="_blank" rel="noopener">{announcement.ctaText} ↗</a>
      {/if}
    </div>
  {/if}
  {#if announcement.note}
    <!-- A row per piece of news rather than more words in one: the launcher
         change and the browser client are unrelated, and running them together
         reads as one long sentence about neither. -->
    <div class="ann-inner ann-note">
      <span class="ann-tag">New</span>
      <span class="ann-sub">{announcement.note}</span>
      <a class="ann-cta" href="/control.html">Control plane →</a>
    </div>
  {/if}
  {#if announcement.pwa}
    <div class="ann-inner ann-note">
      <span class="ann-tag">Preview</span>
      <span class="ann-sub">{announcement.pwa}</span>
      <a class="ann-cta" href={announcement.pwaCtaUrl} target="_blank" rel="noopener">
        {announcement.pwaCtaText} ↗
      </a>
    </div>
  {/if}
</div>
