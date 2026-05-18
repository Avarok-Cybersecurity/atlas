<script>
  import { dockerCommand } from '$lib/data.js';

  let copied = $state(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(dockerCommand);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {}
  }
</script>

<section id="try" class="try-section">
  <div class="container">
    <div class="slabel">Try It Yourself</div>
    <h2 class="stitle">Up and running in two commands</h2>
    <p class="ssub">
      Don't take our word for it. Install
      <a href="https://sparkrun.dev/runtimes/atlas/" class="ssub-link">sparkrun</a>,
      run an Atlas recipe on your DGX Spark, and see the difference. sparkrun pulls
      &amp; runs the Atlas image for you — it uses your existing Docker/Podman +
      NVIDIA container runtime.
    </p>

    <div class="term-card">
      <div class="term-head">
        <div class="term-dots" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        <div class="term-title">Run with sparkrun — Qwen3.6-35B-A3B FP8 + MTP on a single Spark</div>
        <button type="button" class="hero-install-copy" onclick={copy} aria-label="Copy sparkrun commands">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre class="term-body">{dockerCommand}</pre>
    </div>

    <p class="try-foot">
      OpenAI compatible at <span class="mono" style="color: var(--cyan);">http://localhost:8888/v1</span>.
      Works with Claude Code, Cline, OpenCode, Open WebUI, and any OpenAI-compatible client.
    </p>
  </div>
</section>
