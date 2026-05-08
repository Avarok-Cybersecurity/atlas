<script>
  import { discordUrl } from '$lib/data.js';
  import DiscordIcon from './DiscordIcon.svelte';

  let copiedEmail = $state('');
  async function copyEmail(addr, event) {
    // Let modifier-clicks still open the mail client; plain clicks just copy.
    if (event?.metaKey || event?.ctrlKey || event?.shiftKey) return;
    event?.preventDefault();
    try {
      await navigator.clipboard.writeText(addr);
      copiedEmail = addr;
      setTimeout(() => {
        if (copiedEmail === addr) copiedEmail = '';
      }, 1600);
    } catch {}
  }

  const emails = ['debaterishaqui@gmail.com', 'thomas@avarok.net'];
</script>

<section id="contact">
  <div class="container">
    <div class="slabel">Contact</div>
    <h2 class="stitle">Get in touch</h2>
    <p class="ssub">
      We optimize for your use case. Reach out with model requests, hardware setups, or partnership ideas.
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem;">
      <div class="card">
        <div class="card-icon">
          <DiscordIcon size={20} />
        </div>
        <h3>Discord</h3>
        <p>Fastest way to reach us.</p>
        <a
          href={discordUrl}
          style="color: var(--cyan); text-decoration: none; font-weight: 600; font-size: 0.88rem; display: inline-block; margin-top: 0.5rem;"
        >
          discord.gg/DwF3brBMpw
        </a>
      </div>
      <div class="card">
        <div class="card-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
            <polyline points="22,6 12,13 2,6" />
          </svg>
        </div>
        <h3>Email</h3>
        <p>Partnerships and enterprise.</p>
        {#each emails as addr}
          <a
            href="mailto:{addr}"
            class="email-pill"
            title="Click to copy. Cmd/Ctrl+click to open mail client."
            onclick={(e) => copyEmail(addr, e)}
          >
            <span class="email-addr mono">{addr}</span>
            <span class="email-status">{copiedEmail === addr ? 'copied' : 'copy'}</span>
          </a>
        {/each}
      </div>
      <a class="card" href="https://github.com/Avarok-Cybersecurity/atlas" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit; display: block;">
        <div class="card-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
          </svg>
        </div>
        <h3>Open Source</h3>
        <p>AGPL-3.0. Fork it, ship it.</p>
        <span class="mono" style="color: var(--cyan); font-weight: 600; font-size: 0.85rem; display: inline-block; margin-top: 0.5rem;">github.com/Avarok-Cybersecurity/atlas</span>
      </a>
    </div>
  </div>
</section>
