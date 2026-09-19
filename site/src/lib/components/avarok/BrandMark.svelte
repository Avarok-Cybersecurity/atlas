<!--
  Somebody else's logo, shown in its own colours on either theme.

    src       a finished raster badge (the NVIDIA Inception badge). Greyed and
              inverted on the dark theme, because it only comes in dark ink.
    file      an SVG under static/logos, for the light theme.
    fileDark  the same logo for the dark theme, when `file` is drawn in dark
              ink. Without it `file` serves both themes.
    label     a name to set beside a mark that has no wordmark of its own.
    height    pixels. The default is 26.

  Both theme files are lazy and the hidden one is display:none, so only the one
  in use is fetched. static/logos/README.md records the source of every file.
-->
<script>
  let { mark } = $props();
  const height = $derived(mark.height ?? 26);
  const alt = $derived(mark.label ? '' : mark.name);
</script>

<span class="av-mark" style:--h={`${height}px`}>
  {#if mark.src}
    <img src={mark.src} {alt} {height} width="120" loading="lazy" class="is-dark-invert" />
  {:else if mark.file}
    <img src={`/logos/${mark.file}.svg`} {alt} {height} width="120" loading="lazy" class:only-light={mark.fileDark} />
    {#if mark.fileDark}<img src={`/logos/${mark.fileDark}.svg`} {alt} {height} width="120" loading="lazy" class="only-dark" />{/if}
  {/if}
  {#if mark.label || !(mark.src || mark.file)}<span class="av-logo-text">{mark.label ?? mark.name}</span>{/if}
</span>

<style>
  .av-mark { display: inline-flex; align-items: center; gap: 0.6rem; min-height: 34px; }
  .av-mark img { height: var(--h); width: auto; max-width: 160px; }
  .av-mark .av-logo-text { font-size: 0.72rem; white-space: normal; line-height: 1.2; }
  .only-dark { display: none; }
  :global(html:not([data-theme='light'])) .only-dark { display: block; }
  :global(html:not([data-theme='light'])) .only-light { display: none; }
</style>
