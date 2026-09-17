<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Dez — placeholder site

> **Dez: The free and open-source IDE for the local inference-first paradigm**

This directory holds the public placeholder site for **Dez**, an IDE built on the
[Atlas Inference Engine](https://github.com/Avarok-Cybersecurity/atlas), WebGPU, Rust and 100%
WebAssembly. Models run locally in the browser — no server round-trip, no API key, and neither
your code nor your weights leave your machine.

**Dez is in development.** The site says so, prominently and in several places. There is no
download, no release date and no shipping product. The only calls to action are "follow
development" ones. If you edit the copy, keep it that way.

## What is on the page

- A hero carrying the tagline verbatim.
- An honest "what it is" section, including the in-development status.
- A four-card grid: local-first, WebGPU, WebAssembly, open source.
- A "how it works" section describing the Atlas → WebAssembly → WebGPU → editor pipeline.
- A footer linking to the Atlas repository, site, Discord and licence.

Every factual claim traces back to this repository. There are no invented benchmark numbers,
team members, testimonials, funding announcements or launch dates, and none should be added.

## Stack

| Piece | Choice | Why |
| --- | --- | --- |
| Framework | SvelteKit 2 + Svelte 5 (runes) | Small runtime, first-class static output |
| Adapter | `@sveltejs/adapter-static` | Emits plain files for an nginx docroot |
| Language | TypeScript, `strict` | `npm run check` must stay at 0 errors |
| Styling | Plain CSS + custom properties | No framework, no build-time CSS dependency |
| Fonts | System stacks only | Self-hosting zero bytes beats shipping a webfont |
| PWA | Hand-written manifest + `src/service-worker.ts` | Installable and fully offline after first load |

**No trackers, no analytics, no external CDN, no webfont, no third-party request of any kind.**
This is not a preference — a privacy-first product whose landing page phones home is a
contradiction. The build is verified against off-origin subresources; the only external URLs in
the output are `href`s the visitor has to click.

## Develop

Requires Node **>= 20.19** (Vite 8).

```bash
npm install
npm run dev          # http://localhost:5173
```

## Build

```bash
npm run build        # rasterises icons, then prerenders to build/
npm run preview      # serve the production build locally
npm run check        # svelte-check, TypeScript strict
```

Output lands in `build/` as a fully static tree. `npm run build` regenerates
`static/icons/*.png` from `static/icons/icon.svg` first (see `scripts/gen-icons.mjs`), so the
installed app icon can never drift from the favicon. The raster icons are generated, not committed.

### Verifying the PWA

A PWA that fails to register is a silent failure, so check the build output rather than trusting
the config:

```bash
ls build/manifest.webmanifest build/service-worker.js build/icons/
grep -c 'navigator.serviceWorker.register' build/index.html   # expect 1
```

`build/service-worker.js` precaches the hashed app bundle, everything in `static/`, and the
prerendered HTML — that whole set is what makes the site load with the network switched off.
In the browser, confirm under DevTools → Application that the service worker is *activated* and
that the manifest reports no icon errors, then toggle Offline and reload.

## Deploy to dez.rs

`build/` is the docroot. There is no server-side component, no environment variable and no
runtime — copy the tree and reload nginx.

```bash
npm ci && npm run build
rsync -av --delete build/ user@dez.rs:/var/www/dez.rs/
```

```nginx
server {
  listen 443 ssl http2;
  server_name dez.rs;

  root /var/www/dez.rs;
  index index.html;

  # Hashed assets are immutable.
  location /_app/immutable/ {
    add_header Cache-Control "public, max-age=31536000, immutable";
  }

  # The service worker and manifest must never be served stale, or visitors
  # get pinned to an old shell.
  location = /service-worker.js {
    add_header Cache-Control "no-cache";
  }
  location = /manifest.webmanifest {
    add_header Cache-Control "no-cache";
    types { } default_type application/manifest+json;
  }

  location / {
    try_files $uri $uri/ /index.html;
  }
}
```

Serve over HTTPS: service workers and the install prompt require a secure context
(`localhost` excepted).

## Accessibility and performance notes

- Semantic landmarks (`header`/`nav`/`main`/`footer`), a skip link, and headings in order.
- Visible `:focus-visible` rings everywhere; `outline: none` appears nowhere.
- AA contrast in both themes — foreground pairs are ~6:1 or better against their backgrounds.
- Dark mode follows `prefers-color-scheme`, with a manual toggle persisted to `localStorage`
  and applied before first paint so the wrong palette never flashes.
- `prefers-reduced-motion: reduce` is honoured globally and again inside the animated hero
  backdrop, which is CSS-only (no canvas, no timers, no main-thread work).
- Responsive from 320px with no horizontal page scroll.

## Licence

AGPL-3.0-only, matching Atlas. Source files carry an SPDX header.
