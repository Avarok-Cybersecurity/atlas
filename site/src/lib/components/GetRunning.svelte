<script>
  import { getRunning, quickInstall, runCommandRaw, guideUrl, githubUrl } from '$lib/data.js';
  import { currentInstall } from '$lib/install/host.svelte.js';

  // The command, the shell it goes in, and where it lands all move together:
  // a Windows visitor shown `bash` and `~/.local/bin` has been told three
  // things, none of which are true on their machine.
  const install = $derived(currentInstall());
  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';

  let copied = $state('');
  let copyState = $state('idle'); // idle | copied | manual | blocked
  let copyTimer;
  // The flash outlives the component on navigation without this.
  $effect(() => () => clearTimeout(copyTimer));

  async function copy(cmd, el) {
    clearTimeout(copyTimer);
    copied = cmd;
    // Was a silent `return` on refusal, leaving the button unchanged — which
    // the person clicking it reads as success.
    copyState = await copyOrSelect(cmd, el);
    copyTimer = setTimeout(() => {
      if (copied === cmd) {
        copied = '';
        copyState = 'idle';
      }
    }, 2400);
  }
  let termEl = $state(null);
  let quickEl = $state(null);
  let rawEl = $state(null);
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
            <span class="term-title">{install.shell}</span>
          </div>
          <pre class="term-body"><span class="p">{install.prompt}</span> <span class="c" bind:this={termEl}>{install.command}</span>
<span class="d"># downloads atlasctl, verifies its checksum, installs to {install.installDir}</span></pre>
        </div>
        <div class="run-copy">
          <button type="button" class="btn btn-secondary" onclick={() => copy(install.command, termEl)}>
            {copied === install.command ? copyLabel(copyState, 'Copy command') : 'Copy command'}
          </button>
        </div>
        <p class="run-note">{getRunning.quickstartHint}</p>
      </div>

      <div class="run-side">
        <h3>Prefer to inspect first?</h3>
        <p class="run-note">{getRunning.inspectNote}</p>
        <div class="hero-cmd" style="margin-top:0.6rem">
          <span class="prompt">$</span>
          <code bind:this={quickEl}>{quickInstall}</code>
          <button type="button" class="copy-btn" onclick={() => copy(quickInstall, quickEl)}>{copied === quickInstall ? copyLabel(copyState) : 'Copy'}</button>
        </div>
        <div class="hero-cmd" style="margin-top:0.5rem">
          <span class="prompt">$</span>
          <code bind:this={rawEl}>{runCommandRaw}</code>
          <button type="button" class="copy-btn" onclick={() => copy(runCommandRaw, rawEl)}>{copied === runCommandRaw ? copyLabel(copyState) : 'Copy'}</button>
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
