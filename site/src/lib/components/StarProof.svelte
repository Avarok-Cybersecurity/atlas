<script>
  import { stars as copy, githubUrl, testimonials } from '$lib/data.js';
  import starData from '$lib/stars.generated.json';
  import StarChart from './StarChart.svelte';
  import GithubIcon from './GithubIcon.svelte';
</script>

<section id="momentum" class="section-alt">
  <div class="container">
    <div class="slabel">{copy.label}</div>
    <h2 class="stitle">{copy.title}</h2>
    <p class="ssub">{copy.sub}</p>

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
