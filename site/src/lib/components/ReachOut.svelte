<script>
  import { reachout, discordUrl } from '$lib/data.js';
  import DiscordIcon from './DiscordIcon.svelte';

  let copied = $state('');
  async function copy(addr) {
    try {
      await navigator.clipboard.writeText(addr);
      copied = addr;
      setTimeout(() => { if (copied === addr) copied = ''; }, 1600);
    } catch {}
  }
</script>

<section id="reach" class="section-alt">
  <div class="container">
    <div class="reach-head">
      <div>
        <div class="slabel">{reachout.label}</div>
        <h2 class="stitle">{reachout.title}</h2>
        <p class="ssub">{reachout.sub}</p>
      </div>
      <div class="reach-cta">
        {#each reachout.emails as e}
          <div class="email-btn">
            <a class="email-btn-addr" href={`mailto:${e}`}>
              <span class="email-ico" aria-hidden="true">✉</span> {e}
            </a>
            <button type="button" class="email-btn-copy" onclick={() => copy(e)} aria-label={`Copy ${e}`}>
              {copied === e ? 'Copied' : 'Copy'}
            </button>
          </div>
        {/each}
        <a class="btn btn-discord" href={discordUrl} target="_blank" rel="noopener">
          <DiscordIcon size={17} /> {reachout.discordCta}
        </a>
      </div>
    </div>

    <div class="reach-grid">
      {#each reachout.cards as c}
        <div class="reach-card">
          <span class="reach-emoji" aria-hidden="true">{c.emoji}</span>
          <h3>{c.title}</h3>
          <p>{c.body}</p>
        </div>
      {/each}
    </div>
  </div>
</section>
