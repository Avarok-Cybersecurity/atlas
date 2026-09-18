<script>
  import PlatformPage from '$lib/components/avarok/PlatformPage.svelte';
  import { securityPage } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PlatformPage path={routes.security} page={securityPage} color="cyan" />
