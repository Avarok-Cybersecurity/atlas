<script>
  // The deck is composed, not configured: each act contributes its slides in
  // DOM order and the engine numbers them. Adding a slide means adding markup
  // to an act, and there is no manifest to keep in sync.
  import Deck from '$lib/components/deck/Deck.svelte';
  import Frame from '$lib/components/deck/acts/Frame.svelte';
  import Reproduce from '$lib/components/deck/acts/Reproduce.svelte';
  import Evidence from '$lib/components/deck/acts/Evidence.svelte';
  import { stamp } from '$lib/deck/content.js';
</script>

<svelte:head>
  <meta name="robots" content="noindex" />
  <meta
    name="description"
    content="Reproduction steps for the Atlas concurrency ladder: fingerprint, parity, commands, and the artifacts behind the vLLM comparison."
  />
</svelte:head>

<Deck title="Verification steps" {stamp}>
  <Frame />
  <Reproduce />
  <Evidence />
</Deck>
