<script>
  import { dockerCommand, dockerCommand122B } from '$lib/data.js';

  let copied35 = $state(false);
  let copied122 = $state(false);
  async function copy(text, which) {
    try {
      await navigator.clipboard.writeText(text);
      if (which === '35') {
        copied35 = true;
        setTimeout(() => (copied35 = false), 1600);
      } else {
        copied122 = true;
        setTimeout(() => (copied122 = false), 1600);
      }
    } catch {}
  }
</script>

<section id="try" class="try-section">
  <div class="container">
    <div class="slabel">Try It Yourself</div>
    <h2 class="stitle">Up and running in two commands</h2>
    <p class="ssub">
      Don't take our word for it. Pull the image. Run it on your DGX Spark. See the difference.
    </p>

    <div class="term-card">
      <div class="term-head">
        <div class="term-dots" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        <div class="term-title">Qwen3.5-35B &mdash; 130 tok/s on a single Spark</div>
        <button type="button" class="hero-install-copy" onclick={() => copy(dockerCommand, '35')} aria-label="Copy 35B command">
          {copied35 ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre class="term-body">{dockerCommand}</pre>
    </div>

    <div class="term-card" style="margin-top: 1.25rem;">
      <div class="term-head">
        <div class="term-dots" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        <div class="term-title">Qwen3.5-122B &mdash; ~33 tok/s on a single Spark</div>
        <button type="button" class="hero-install-copy" onclick={() => copy(dockerCommand122B, '122')} aria-label="Copy 122B command">
          {copied122 ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre class="term-body">{dockerCommand122B}</pre>
    </div>

    <p class="try-foot">
      OpenAI compatible at <span class="mono" style="color: var(--cyan);">http://localhost:8888/v1</span>.
      Works with Claude Code, Cline, OpenCode, Open WebUI, and any OpenAI-compatible client.
    </p>
  </div>
</section>
