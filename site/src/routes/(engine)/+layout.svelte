<script>
  // The developer routes: /engine, /control, /diligence. The ledger design
  // system in app.css and its companions, plus the chevron field behind them.
  // Load order matters. app.css is the desktop-first design system, news.css
  // adds the news band, dashboard.css then chat.css add the two modals (chat
  // reuses dashboard pieces so it loads after), ladder.css styles the
  // concurrency section (it reuses dashboard's chart primitives, so it loads
  // after those), mobile.css is the SSOT for every viewport rule and lands last.
  import '../../app.css';
  import '../../styles/news.css';
  import '../../styles/dashboard.css';
  import '../../styles/chat.css';
  import '../../styles/ladder.css';
  import '../../styles/control.css';
  import '../../styles/bridge.css';
  import '../../styles/stage.css';
  import '../../styles/overlays.css';
  import '../../styles/mobile.css';
  import '../../styles/engine-shell.css';
  import ChevronField from '$shared/components/ChevronField.svelte';
  import SiteNav from '$lib/components/avarok/SiteNav.svelte';
  import SiteFooter from '$lib/components/avarok/SiteFooter.svelte';
  let { children } = $props();
</script>

<!-- The ambient chevron field: one fullscreen triangle, one fragment shader,
     the same code blog.atlascybernetics.ai renders. It paints the page ground
     itself, so `body`'s background sits behind it rather than beside it.

     It must stay a DIRECT child of the layout root. A `transform`, `filter`,
     `perspective`, `will-change` or `contain: paint` on any ancestor would
     make that ancestor the containing block for fixed-position descendants,
     and the background would start scrolling with the content. -->
<ChevronField />

<!-- Everything else goes above the canvas. Without this wrapper, only the
     POSITIONED elements would: a fixed canvas at z-index 0 paints above the
     block-level layer, so `footer` — which sets no position — would render
     underneath it and vanish. -->
<div class="page engine-shell">
  <SiteNav />
  {@render children()}
  <SiteFooter />
</div>
