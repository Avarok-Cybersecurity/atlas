<script>
  import '../app.css';
  import { tagline, githubUrl, recipesUrl, discordUrl, xUrl } from '$lib/data.js';
  let { children } = $props();
</script>

<svelte:head>
  <title>Atlas, pure Rust inference for DGX Spark</title>
  <link rel="canonical" href="https://atlasinference.io/" />
  {@html `<script type="application/ld+json">${JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'Atlas Inference Engine',
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'Linux',
    description: tagline,
    url: 'https://atlasinference.io/',
    license: 'https://spdx.org/licenses/AGPL-3.0-only.html',
    sameAs: [githubUrl, recipesUrl, discordUrl, xUrl]
  })}<\/script>`}
</svelte:head>

{@render children()}
