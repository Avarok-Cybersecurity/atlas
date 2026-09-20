<script>
  import SolutionPage from '$lib/components/avarok/SolutionPage.svelte';
  import { solutionHref } from '$lib/content/brand.js';
  let { data } = $props();
</script>

<SolutionPage path={solutionHref(data.s.slug)} s={data.s} />
