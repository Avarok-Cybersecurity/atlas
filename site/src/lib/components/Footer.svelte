<script>
  import { footer, contactEmails, discordUrl, mlperfTrademark } from '$lib/data.js';
</script>

<footer>
  <div class="footer-inner">
    <div class="footer-brand">
      <div class="flogo"><img src="/favicon.svg" alt="" width="28" height="28" /> Atlas</div>
      <p>{footer.tagline}</p>
      <p class="lic">{footer.license}</p>
    </div>

    {#each footer.cols as col}
      <div class="fcol">
        <h4>{col.heading}</h4>
        {#each col.links as l}
          <a href={l.url} target="_blank" rel="noopener">{l.text}</a>
        {/each}
      </div>
    {/each}

    <div class="fcol">
      <h4>Contact</h4>
      <a href={discordUrl} target="_blank" rel="noopener">Discord</a>
      {#each contactEmails as e}
        <a href={`mailto:${e}`}>{e}</a>
      {/each}
    </div>
  </div>

  <div class="footer-legal">
    <div>© Atlas Inference · Pure Rust + CUDA · Community Edition AGPLv3.</div>
    <div class="tm">{mlperfTrademark}</div>
  </div>
</footer>
