<!-- An external video, embedded without letting the provider load until the
     reader actually asks for it.

     A bare <iframe> costs roughly half a megabyte of third-party JavaScript, a
     third-party cookie and several extra connections on EVERY page view,
     whether or not anyone presses play — which is most of a Lighthouse
     performance score and all of a best-practices one. So the page ships a
     local poster and a button; the provider is only contacted on click.

     The poster is a committed file rather than the provider's thumbnail CDN
     for the same reason: fetching i.ytimg.com at load would reintroduce the
     third-party connection this component exists to remove, and would need a
     hole in any future CSP. -->
<script>
  let { provider, id, title, poster, start = 0 } = $props();

  // youtube-nocookie is the no-tracking-until-play host. The map is exhaustive
  // and the validator refuses any other provider at build time, so this cannot
  // be indexed with something undefined.
  const SRC = {
    youtube: (v, s) => `https://www.youtube-nocookie.com/embed/${v}?autoplay=1&start=${s}`,
    vimeo: (v) => `https://player.vimeo.com/video/${v}?autoplay=1`
  };

  let playing = $state(false);
</script>

<figure class="video">
  {#if playing}
    <iframe
      src={SRC[provider](id, start)}
      {title}
      allow="autoplay; encrypted-media; picture-in-picture"
      referrerpolicy="strict-origin-when-cross-origin"
      allowfullscreen
      loading="lazy"
    ></iframe>
  {:else}
    <button type="button" onclick={() => (playing = true)} aria-label="Play video: {title}">
      <img src={poster} alt="" width="1280" height="720" loading="lazy" decoding="async" />
      <span class="play" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M8 5.5 18.5 12 8 18.5Z" />
        </svg>
      </span>
    </button>
  {/if}
  <figcaption>{title}</figcaption>
</figure>

<style>
  .video { margin: 2rem 0; }
  /* Reserved before anything loads, so swapping the poster for the player
     cannot shift the article underneath it. */
  .video button,
  .video iframe {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .video button {
    position: relative;
    padding: 0;
    background: var(--sunk);
    cursor: pointer;
  }
  .video img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .play {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }
  .play svg {
    width: 64px;
    height: 64px;
    padding: 16px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--sunk) 82%, transparent);
    border: 1px solid var(--border-strong);
    stroke: var(--accent);
    fill: var(--accent);
    backdrop-filter: blur(6px);
  }
  .video button:hover .play svg { border-color: var(--accent); }
  .video button:focus-visible { outline: 2.5px solid var(--accent); outline-offset: 3px; }
  figcaption {
    margin-top: 0.6rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-3);
  }
</style>
