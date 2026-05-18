<script>
  import Nav from '$lib/components/Nav.svelte';
  import Hero from '$lib/components/Hero.svelte';
  import Stats from '$lib/components/Stats.svelte';
  import Speed from '$lib/components/Speed.svelte';
  import ModelSlider from '$lib/components/ModelSlider.svelte';
  import Roadmap from '$lib/components/Roadmap.svelte';
  import Community from '$lib/components/Community.svelte';
  import Contact from '$lib/components/Contact.svelte';
  import TryIt from '$lib/components/TryIt.svelte';
  import Footer from '$lib/components/Footer.svelte';
</script>

<svelte:head>
  <title>Atlas Inference Engine</title>
  {@html `<script type="application/ld+json">${JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'Atlas Inference Engine',
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'Linux',
    description: 'Pure Rust LLM Inference.',
    url: 'https://atlasinference.io/',
    sameAs: [
      'https://sparkrun.dev/runtimes/atlas/',
      'https://github.com/Avarok-Cybersecurity/atlas-recipes'
    ]
  })}<\/script>`}
</svelte:head>

<Nav />
<Hero />
<Stats />
<Speed />
<ModelSlider />
<TryIt />
<Community />
<Contact />
<Roadmap />
<Footer />
