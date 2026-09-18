# The Avarok website

The marketing site and the developer pages at atlascybernetics.ai. SvelteKit,
prerendered to static files, deployed to Cloudflare Pages by
`.github/workflows/site.yml`.

Read `FACELIFT.md` for what is where and why. This file is the commands.

## Run it

You need [bun](https://bun.sh) and a clone of
[atlas-recipes](https://github.com/Avarok-Cybersecurity/atlas-recipes) beside
this repository. The build generates its data from both.

```sh
cd site
bun install

export AVAROK_RECIPES_ROOT=/path/to/atlas-recipes/recipes
export AVAROK_BASELINES_ROOT=/path/to/atlas/tests/baselines
export GH_TOKEN=$(gh auth token)     # optional: star history and contributors

bun x --bun vite dev                 # http://localhost:5173
bun x --bun vite build               # writes build/
bun x --bun vite preview             # serves build/ on http://localhost:4173
```

Without `GH_TOKEN` the build still succeeds. The star curve and the contributor
list fall back to the committed files.

On Windows use Git Bash, with forward slash paths in the two variables
(`/c/Users/you/atlas-recipes/recipes`).

## Test it

```sh
bun test --preload ./test-runes.js src/lib          # unit, about a second
bun x --bun playwright test e2e/marketing.spec.js   # browser, builds first
(cd .. && bun .contrast-check.mjs)                  # contrast, run from the repository root
```

## Change it

All copy, links, prices and page titles are data in `src/lib/content/`.
`FACELIFT.md` has a table of what to edit for each kind of change. The unit
suite fails when a link points nowhere, when a page has no title, or when a
sentence breaks the house voice, so run it after editing copy.

## Media

```sh
bun run media     # record and encode the procedural loops, keep the console clips
bun run og        # render static/og-image.png
```

The console clips are recordings of a private product mockup that is not in
this repository. `media-brief/README.md` explains that, and holds the prompt
pack for the imagery still to be generated. `scripts/media/README.md` explains
each script.

## Layout

```
src/lib/content/             copy, routes, nav, prices, page registry
src/lib/components/avarok/   marketing components
src/lib/components/          developer page components
src/routes/(marketing)/      marketing pages
src/routes/(engine)/         /engine, /control, /diligence
src/routes/(app)/            render pages for the media pipeline
src/styles/avarok.css        marketing design system
scripts/                     generators, run by the build
scripts/media/               media pipeline
media-brief/                 prompt pack and reel storyboard
e2e/                         browser tests
static/                      shipped as is: fonts, logos, media, icons
```
