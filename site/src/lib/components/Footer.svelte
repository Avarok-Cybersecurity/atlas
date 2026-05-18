<script>
  import { discordUrl, redditUrl } from '$lib/data.js';
</script>

<footer>
  <div class="footer-inner">
    <div class="footer-brand">
      <div class="flogo">Atlas</div>
      <p>LLM inference from scratch in Rust and CUDA. Zero python dependencies, zero complex recipes. Maximum throughput.</p>
    </div>
    <div class="fcol">
      <h4>Product</h4>
      <a href="#speed">Why Atlas</a>
      <a href="#models">Models</a>
      <a href="#roadmap">Roadmap</a>
    </div>
    <div class="fcol">
      <h4>Run Atlas</h4>
      <a href="https://sparkrun.dev/runtimes/atlas/" target="_blank" rel="noopener">sparkrun</a>
      <a href="https://github.com/Avarok-Cybersecurity/atlas-recipes" target="_blank" rel="noopener">Recipes (SSOT)</a>
    </div>
    <div class="fcol">
      <h4>Community</h4>
      <a href={discordUrl}>Discord</a>
      <a href={redditUrl}>Reddit</a>
    </div>
    <div class="fcol">
      <h4>Contact</h4>
      <a href={discordUrl}>Discord</a>
      <a href="mailto:debaterishaqui@gmail.com">debaterishaqui@gmail.com</a>
      <a href="mailto:thomas@avarok.net">thomas@avarok.net</a>
    </div>
  </div>
  <div class="footer-bottom">Atlas Inference Engine. Built with Rust and CUDA.</div>
</footer>
