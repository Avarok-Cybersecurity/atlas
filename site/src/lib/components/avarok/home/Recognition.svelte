<!--
  "Receipts, not adjectives." Four cards, each one a receipt: who issued it (their
  own logo), when, what it says, a tear line, and the source it links to with
  the host printed the way a receipt prints the till. Copy is `recognition` in
  src/lib/content/home.js. `hue` picks one of the four brand colours for the
  glow and the hover edge. `mark` is a BrandMark (see BrandMark.svelte).
-->
<script>
  import { recognition } from '$lib/content/home.js';
  import BrandMark from '../BrandMark.svelte';

  const host = (href) => new URL(href).host.replace(/^www\./, '');
</script>

<section class="av-section" id="recognition">
  <div class="av-container">
    <div class="av-head av-center av-reveal">
      <p class="av-eyebrow av-sx-gold" style="justify-content:center">{recognition.eyebrow}</p>
      <h2 class="av-h2">{recognition.title}</h2>
      <p class="av-lede">{recognition.lede}</p>
    </div>
    <div class="av-grid av-grid-4 av-reveal">
      {#each recognition.cards as c}
        <a class="av-receipt av-sx-{c.hue}" href={c.href} target="_blank" rel="noopener">
          <span class="av-receipt-head">
            <BrandMark mark={c.mark} />
            <span class="av-receipt-date">{c.date}</span>
          </span>
          <h3>{c.title}</h3>
          <span class="av-receipt-foot">
            <span class="av-receipt-cta">{c.cta} <span class="av-arrow">↗</span></span>
            <span class="av-receipt-host">{host(c.href)}</span>
          </span>
        </a>
      {/each}
    </div>
  </div>
</section>

<style>
  .av-receipt {
    position: relative; isolation: isolate; overflow: hidden;
    display: grid; grid-template-rows: auto 1fr auto; gap: 1rem;
    padding: 1.35rem 1.35rem 1.15rem;
    background: var(--card); border: 1px solid var(--border); border-radius: var(--av-radius);
    color: inherit; text-decoration: none;
    transition: transform 0.22s var(--av-ease), border-color 0.22s, box-shadow 0.22s;
  }
  /* The card's colour, as light from its top right corner. It brightens on hover. */
  .av-receipt::before {
    content: ''; position: absolute; z-index: -1; right: -30%; top: -55%; width: 90%; aspect-ratio: 1;
    background: radial-gradient(closest-side, color-mix(in srgb, var(--sx) 34%, transparent), transparent);
    opacity: 0.55; transition: opacity 0.3s, transform 0.5s var(--av-ease);
  }
  .av-receipt:hover { transform: translateY(-4px); border-color: color-mix(in srgb, var(--sx) 70%, var(--border)); box-shadow: 0 24px 48px -28px color-mix(in srgb, var(--sx) 60%, transparent), var(--av-shadow); }
  .av-receipt:hover::before { opacity: 1; transform: scale(1.15); }
  .av-receipt-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; min-height: 36px; }
  .av-receipt-date { font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--sx-text); white-space: nowrap; padding: 0.22rem 0.5rem; border: 1px solid color-mix(in srgb, var(--sx) 40%, transparent); border-radius: 999px; background: color-mix(in srgb, var(--sx) 10%, transparent); }
  .av-receipt h3 { font-size: 1rem; line-height: 1.38; letter-spacing: -0.01em; font-weight: 600; }
  /* The tear line of a receipt, and the till it was printed at. */
  /* Two rows on every card, never one on some: the tear lines have to line up. */
  .av-receipt-foot { display: grid; gap: 0.35rem; padding-top: 0.9rem; border-top: 1px dashed var(--border-strong); }
  .av-receipt-cta { font-weight: 600; font-size: 0.9rem; color: var(--accent); display: inline-flex; gap: 0.3rem; }
  .av-receipt-host { font-family: var(--font-mono); font-size: 0.68rem; color: var(--t3); letter-spacing: 0.02em; }
  .av-receipt .av-arrow { transition: transform 0.22s var(--av-ease); }
  .av-receipt:hover .av-arrow { transform: translate(2px, -2px); }
</style>
