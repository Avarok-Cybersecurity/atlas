<!--
  The PRERENDERED 404 document. adapter-static writes this route to
  `build/404.html`, and both hosts reach for that exact filename: nginx through
  `error_page 404`, Cloudflare Pages by convention.

  It is not cosmetic. Pages has no `try_files ... =404` — with no 404.html in
  the output it answers EVERY unmatched path with index.html and a 200, so a
  typo'd link returns the front page as though it were the page asked for, and
  crawlers index an unbounded set of soft-404 URLs. Measured on the first
  deployment: /definitely-not-a-real-path-zzz came back 200 with the homepage
  title. The blog never had this problem because it already ships a /404 route;
  this is the marketing site catching up to it.
-->
<script>
  import { githubUrl } from '$lib/data.js';
</script>

<svelte:head>
  <title>Not found — Atlas</title>
  <meta name="robots" content="noindex" />
  <!-- `noindex` keeps the page out of results; it does not excuse it from a
       description. Lighthouse audits the tag's presence, not indexability, so
       without this the 404 scores below 100 on SEO while every other page
       passes — the same reasoning as the blog's 404. -->
  <meta
    name="description"
    content="That page is not here. Atlas is a pure-Rust inference engine for DGX Spark and Strix Halo; the front page has the install command and the benchmarks."
  />
</svelte:head>

<div class="nf">
  <p class="code">404</p>
  <h1>That page is not here.</h1>
  <p class="lede">
    It may have been renamed, or it may never have existed. The
    <a href="/">front page</a> has the install command, the measured benchmarks and
    the model ladder; <a href="/control">the control plane</a> is where a running
    fleet shows up. The source is on <a href={githubUrl}>GitHub</a>.
  </p>
</div>

<style>
  .nf {
    max-width: 44rem;
    margin: 0 auto;
    padding: clamp(96px, 18vh, 200px) 24px 160px;
  }
  .code {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    color: var(--t3);
    margin: 0 0 12px;
  }
  h1 {
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    line-height: 1.15;
    margin: 0 0 20px;
    color: var(--t1);
  }
  .lede {
    color: var(--t2);
    line-height: 1.7;
    margin: 0;
  }
  .lede a {
    color: var(--accent);
  }
</style>
