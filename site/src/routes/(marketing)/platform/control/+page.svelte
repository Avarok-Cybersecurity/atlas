<script>
  import PlatformPage from '$lib/components/avarok/PlatformPage.svelte';
  import { controlPage } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PlatformPage path={routes.control} page={controlPage} color="cyan" />
