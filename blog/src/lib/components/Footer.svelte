<script>
  import { footerCols, blog, MAIN_SITE } from '$lib/content.js';
  const year = 2026; // prerendered: a live clock would bake the build date in
</script>

<footer class="ftr">
  <div class="shell ftr-grid">
    <div class="ftr-col ftr-brand">
      <a class="ftr-logo" href={MAIN_SITE}>
        <img src="/favicon.svg" alt="" width="28" height="28" /> Atlas
      </a>
      <p>Zero-trust inference on hardware you own. Pure Rust and CUDA, built in North Carolina.</p>
    </div>
    {#each footerCols as col}
      <div class="ftr-col">
        <h4>{col.heading}</h4>
        <ul>
          {#each col.links as l}
            <li>
              <a href={l.href} target={l.href.startsWith('http') ? '_blank' : undefined} rel={l.href.startsWith('http') ? 'noopener' : undefined}>{l.text}</a>
            </li>
          {/each}
        </ul>
      </div>
    {/each}
  </div>
  <div class="ftr-legal">
    <div>© {year} Atlas Inference · Community Edition AGPLv3</div>
    <div>{blog.kicker}</div>
  </div>
</footer>
