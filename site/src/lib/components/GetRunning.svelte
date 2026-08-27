<script>
  import { getRunning, runCommand, quickInstall, runCommandRaw, guideUrl, githubUrl } from '$lib/data.js';
  import { copyText } from '$lib/clipboard.js';

  let copied = $state('');
  async function copy(cmd) {
    if ((await copyText(cmd)) !== 'copied') return;
    copied = cmd;
    setTimeout(() => { if (copied === cmd) copied = ''; }, 1600);
  }
  import SectionHead from './SectionHead.svelte';
</script>

<section id="run" class="section-alt sx-cyan">
  <div class="container">
    <SectionHead
      label={getRunning.label}
      title={getRunning.title}
      sub={getRunning.sub}
    />

    <div class="run-grid">
      <div>
        <div class="term">
          <div class="term-head">
            <div class="term-dots"><span></span><span></span><span></span></div>
            <span class="term-title">bash</span>
          </div>
          <pre class="term-body"><span class="p">$</span> <span class="c">{runCommand}</span>
<span class="d"># downloads atlasctl, verifies its checksum, installs to ~/.local/bin</span></pre>
        </div>
        <div class="run-copy">
          <button type="button" class="btn btn-secondary" onclick={() => copy(runCommand)}>
            {copied === runCommand ? 'Copied' : 'Copy command'}
          </button>
        </div>
        <p class="run-note">{getRunning.quickstartHint}</p>
      </div>

      <div class="run-side">
        <h3>Prefer to inspect first?</h3>
        <p class="run-note">{getRunning.inspectNote}</p>
        <div class="hero-cmd" style="margin-top:0.6rem">
          <span class="prompt">$</span>
          <code>{quickInstall}</code>
          <button type="button" class="copy-btn" onclick={() => copy(quickInstall)}>{copied === quickInstall ? 'Copied' : 'Copy'}</button>
        </div>
        <div class="hero-cmd" style="margin-top:0.5rem">
          <span class="prompt">$</span>
          <code>{runCommandRaw}</code>
          <button type="button" class="copy-btn" onclick={() => copy(runCommandRaw)}>{copied === runCommandRaw ? 'Copied' : 'Copy'}</button>
        </div>
        <p class="run-note">
          The first 60 seconds live here. Everything after, per model recipes, EP=2, tuning,
          lives in the docs. <a class="link" href={guideUrl} target="_blank" rel="noopener">{getRunning.docsCta}</a>
          · <a class="link" href={githubUrl} target="_blank" rel="noopener">README</a>
        </p>
      </div>
    </div>
  </div>
</section>
