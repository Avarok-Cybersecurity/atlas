<script>
  import { reachout, discordUrl } from '$lib/data.js';
  import { copyText } from '$lib/clipboard.js';
  import DiscordIcon from './DiscordIcon.svelte';

  let copied = $state('');
  async function copy(addr) {
    if ((await copyText(addr)) !== 'copied') return;
    copied = addr;
    setTimeout(() => { if (copied === addr) copied = ''; }, 1600);
  }
  import SectionHead from './SectionHead.svelte';
</script>

<section id="reach" class="section-alt sx-gold">
  <div class="container">
    <div class="reach-head">
      <div>
        <SectionHead label={reachout.label} title={reachout.title} sub={reachout.sub} />
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
