<script>
  // Page order is the ledger order. Entry 02 is Verified, so the second screen
  // a visitor reaches is the concurrency ladder against a named competitor —
  // not a Reddit chart. Everything below it is context for that claim.
  import Nav from '$lib/components/Nav.svelte';
  import GiftBanner from '$lib/components/GiftBanner.svelte';
  import Hero from '$lib/components/Hero.svelte';
  import Proof from '$lib/components/Proof.svelte';
  import Verified from '$lib/components/Verified.svelte';
  import News from '$lib/components/News.svelte';
  import Hardware from '$lib/components/Hardware.svelte';
  import ModelSlider from '$lib/components/ModelSlider.svelte';
  import GetRunning from '$lib/components/GetRunning.svelte';
  import Community from '$lib/components/Community.svelte';
  import Contribute from '$lib/components/Contribute.svelte';
  import Mission from '$lib/components/Mission.svelte';
  import Faq from '$lib/components/Faq.svelte';
  import ReachOut from '$lib/components/ReachOut.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import LaunchDialog from '$lib/components/LaunchDialog.svelte';
</script>

<Nav />
<GiftBanner />
<Hero />
<Proof />
<Verified />
<News />
<Hardware />
<ModelSlider />
<GetRunning />
<Community />
<Contribute />
<Mission />
<Faq />
<ReachOut />
<Footer />

<!-- Rendered at the page root on purpose: nested in a recipe card, any hover
     transform on an ancestor would become the containing block for the
     dialog's `position: fixed`. -->
<LaunchDialog />
