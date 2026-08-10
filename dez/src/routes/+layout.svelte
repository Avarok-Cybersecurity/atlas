<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import '../app.css';
  import SiteFooter from '$lib/components/SiteFooter.svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';

  const { children } = $props();
</script>

<a class="skip-link" href="#main">Skip to content</a>

<SiteHeader />

<main id="main" tabindex="-1">
  {@render children?.()}
</main>

<SiteFooter />

<style>
  main:focus {
    outline: none;
  }
</style>
