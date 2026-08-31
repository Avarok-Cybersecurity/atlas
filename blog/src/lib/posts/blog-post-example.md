---
title: Everything a post can do
dek: The markdown pipeline exercised end to end — every block this format renders, in one draft that never ships.
categories: [engineering, design]
date: 2026-08-30
keywords: [atlasctl, markdown, katex, shiki, authoring]
og-image: ''
author: thomas-braun
draft: true
---

Posts can be written two ways. A Svelte component, when a piece needs measured
tables and inline figures that only a component can express — or markdown, when
it is an article. Both compile to the same thing and land in the same index,
the same feed and the same prev/next chain, so a reader cannot tell which is
which. This is the markdown one, and it exists to exercise every feature.

It is a `draft`: it builds in CI, where the bundle and Lighthouse gates read
it, and it never reaches the index, the feed or the sitemap.

## Headings and images

Every `##` gets an id from its own text and a chevron coloured by its position,
cycling violet, cyan, green, gold — the same rail a hand-written heading draws.

Images must be `.svg`, `.gif` or `.webp`. Anything else fails the build rather
than warning, and so does an image with no alt text or one whose file is
missing, because a broken image is invisible in review and obvious in
production.

![The Atlas chevron mark](/images/posts/blog-post-example/mark.svg)

![Decode throughput climbing to its plateau over a thirty-second window](/images/posts/blog-post-example/throughput.webp)

![The scheduler admitting a burst of twelve requests one at a time](/images/posts/blog-post-example/admission.gif)

## Running atlasctl

Fenced code is highlighted during the build, in the site's own colours — the
theme names design tokens rather than hex values, so the palette cannot drift
from the rest of the page. The filename rides on the fence.

```bash name=warm-check.sh
# Ask the engine what it already has resident before pointing traffic at it.
atlasctl status --format json | jq '.models[] | {name, quant, resident}'

# Warm the flagship and hold it there.
atlasctl load qwen3.8-27b --quant nvfp4 --pin
atlasctl bench ttft --n 10 --percentile 90
```

Inline code like `atlasctl drain --grace 30s` keeps the monospace face, and a
dollar inside it — `echo $ATLAS_HOME` — is never mistaken for mathematics,
because code is tokenised before the maths extension ever sees the line.

## Mathematics

Inline maths takes single dollars: attention costs $O(n^2 d)$ per layer, and a
price is written escaped, so \$99 stays a price. Display maths takes a fence:

```latex
\text{TTFT} = t_{\text{queue}} + \frac{n_{\text{prompt}} \cdot c_{\text{layer}}}{\text{FLOPs}_{\text{eff}}} + t_{\text{sample}}
```

<Callout label="Rendered at build time" tone="verified">
KaTeX runs in the build. The browser receives HTML and MathML plus one
stylesheet, on maths pages only — there is no maths JavaScript in the bundle,
and a test fails the build the day that stops being true.
</Callout>

## A measured table

Right-aligned columns carry the numeric treatment, so figures line up on their
digits the way they do in a hand-written table.

| percentile | TTFT (ms) | tok/s |
| ---------- | --------: | ----: |
| p50        |       412 |  41.8 |
| p90        |       688 |  39.2 |
| p99        |      1103 |  36.5 |

## Video

An external video loads nothing at all until the reader asks for it: a local
poster and a button, and only then the player. A bare embed would cost half a
megabyte of third-party JavaScript and a third-party cookie on every view,
whether or not anyone pressed play.

<Video
  provider="youtube"
  id="dQw4w9WgXcQ"
  title="Atlas on GB10: a live serve walkthrough"
  poster="/images/posts/blog-post-example/serve-walkthrough.webp"
/>

That is the whole surface: prose, headings, three image formats, highlighted
fences, inline code, both forms of mathematics, a callout, a numeric table and
a facade embed.
