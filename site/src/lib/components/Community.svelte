<script>
  import { community, discordUrl } from '$lib/data.js';
  import DiscordIcon from './DiscordIcon.svelte';
</script>

<section id="community">
  <div class="container">
    <div class="community-band">
      <div class="community-copy">
        <div class="slabel">{community.label}</div>
        <h2 class="community-title">{community.title}</h2>
        <p class="community-body">{community.body}</p>
      </div>
      <div class="community-cta">
        <a class="btn btn-discord btn-lg" href={discordUrl} target="_blank" rel="noopener">
          <DiscordIcon size={20} /> {community.cta}
        </a>
        <span class="community-sub">{community.sub}</span>
      </div>
    </div>
  </div>
</section>
