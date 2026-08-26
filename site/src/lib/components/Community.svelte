<script>
  // Ledger entry 07. One community band, not three: the star curve, the Discord
  // CTA and the testimonials were separate full-height sections that diluted
  // each other and pushed the verified numbers below three screens of social
  // proof. They are one argument, so they are one entry.
  import { stars as copy, community, githubUrl, discordUrl, testimonials } from '$lib/data.js';
  import starData from '$lib/stars.generated.json';
  import StarChart from './StarChart.svelte';
  import GithubIcon from './GithubIcon.svelte';
  import DiscordIcon from './DiscordIcon.svelte';
  import SectionHead from './SectionHead.svelte';
</script>

<section id="community" class="section-alt sx-gold">
  <div class="container">
    <SectionHead
      label={copy.label}
      title={copy.title}
      sub={copy.sub}
      prov="live from the GitHub API"
      provUrl={githubUrl}
    />

    <div class="star-wrap">
      <div>
        <div class="star-figure">{starData.count}<span class="unit"> ★</span></div>
        <div class="star-cap">GitHub stars and climbing, live from the API.</div>
        <div class="star-cta-row">
          <a class="btn btn-primary" href={githubUrl} target="_blank" rel="noopener">
            <GithubIcon size={16} /> {copy.cta}
          </a>
        </div>
      </div>
      <StarChart />
    </div>

    <div class="community-band">
      <div class="community-copy">
        <div class="slabel">{community.label}</div>
        <h3 class="community-title">{community.title}</h3>
        <p class="community-body">{community.body}</p>
      </div>
      <div class="community-cta">
        <a class="btn btn-discord btn-lg" href={discordUrl} target="_blank" rel="noopener">
          <DiscordIcon size={20} /> {community.cta}
        </a>
        <span class="community-sub">{community.sub}</span>
      </div>
    </div>

    <div class="tstrip">
      {#each testimonials as t}
        <div class="tcard">
          <blockquote>“{t.quote}”</blockquote>
          <div class="attr">{t.author}, <a href={t.sourceUrl} target="_blank" rel="noopener">{t.source}</a></div>
        </div>
      {/each}
    </div>
  </div>
</section>
