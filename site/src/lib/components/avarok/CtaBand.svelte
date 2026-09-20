<!-- The closing band every page ends on. Everything loops back to the demo. -->
<script>
  let { eyebrow = 'Next step', title, body = '', primary, secondary = null, id = 'next' } = $props();
</script>

<section class="av-section av-section-tight" {id}>
  <div class="av-container">
    <div class="av-band av-reveal">
      <div class="av-band-grid">
        <div>
          <p class="av-eyebrow">{eyebrow}</p>
          <h2 class="av-h2">{title}</h2>
          {#if body}<p class="av-lede">{body}</p>{/if}
        </div>
        <div class="av-row av-band-actions">
          <a class="av-btn av-btn-primary av-btn-lg" href={primary.href}>{primary.text} <span class="av-arrow">→</span></a>
          {#if secondary}
            <a class="av-btn av-btn-ghost av-btn-lg" href={secondary.href} target={secondary.external ? '_blank' : undefined} rel={secondary.external ? 'noopener' : undefined}>{secondary.text}</a>
          {/if}
        </div>
      </div>
    </div>
  </div>
</section>

<style>
  .av-band-actions { justify-content: flex-end; }
  @media (max-width: 900px) { .av-band-actions { justify-content: flex-start; } }
</style>
