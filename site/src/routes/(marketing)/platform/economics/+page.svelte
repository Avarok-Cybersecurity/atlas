<script>
  import PlatformPage from '$lib/components/avarok/PlatformPage.svelte';
  import { economicsPage as p } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PlatformPage path={routes.economics} page={p} color="green">
  {#snippet before()}
    <section class="av-section av-section-tight">
      <div class="av-container">
        <div class="av-cols av-reveal">
          <div class="av-col">
            <h3>{p.columns.left.title}</h3>
            <ul>{#each p.columns.left.items as it}<li>{it}</li>{/each}</ul>
          </div>
          <div class="av-cols-mid" aria-hidden="true"><span>×</span></div>
          <div class="av-col av-col-right">
            <h3>{p.columns.right.title}</h3>
            <ul>{#each p.columns.right.items as it}<li>{it}</li>{/each}</ul>
          </div>
        </div>
        <div class="av-bridge av-reveal">
          <h3>{p.columns.middle.title}</h3>
          <p>{p.columns.middle.body}</p>
        </div>
      </div>
    </section>
  {/snippet}
</PlatformPage>
