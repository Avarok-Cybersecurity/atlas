<script>
  import { hero, runCommand, githubUrl, discordUrl } from '$lib/data.js';
  import Receipt from './Receipt.svelte';
  import DiscordIcon from './DiscordIcon.svelte';

  let copied = $state(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(runCommand);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {}
  }
</script>

<section class="hero">
  <div class="hero-inner">
    <div class="hero-copy">
      <span class="hero-badge"><span class="dot"></span> {hero.badge}</span>
      <h1>{hero.headline[0]} <span class="lede2">{hero.headline[1]}</span></h1>
      <p class="hero-sub">{hero.sub}</p>

      <div class="hero-cmd" role="group" aria-label="Run Atlas">
        <span class="prompt">$</span>
        <code>{runCommand}</code>
        <button type="button" class="copy-btn" onclick={copy} aria-label="Copy run command">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      <div class="hero-buttons">
        <a class="btn btn-primary" href={githubUrl} target="_blank" rel="noopener">{hero.primaryCta}</a>
        <a class="btn btn-discord" href={discordUrl} target="_blank" rel="noopener"><DiscordIcon size={17} /> {hero.discordCta}</a>
        <a class="btn btn-ghost" href="#run">{hero.secondaryCta}</a>
      </div>

      <p class="hero-challenge">
        {hero.challenge.lead} <strong>{hero.challenge.claim}</strong>
        <span class="fine">{hero.challenge.fine}</span>
      </p>
    </div>

    <div class="hero-receipt">
      <Receipt compact={true} />
    </div>
  </div>
</section>
