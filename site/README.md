# The Avarok website

The marketing site and the developer pages at atlascybernetics.ai. SvelteKit,
prerendered to static files, deployed to Cloudflare Pages by
`.github/workflows/site.yml`.

Read `FACELIFT.md` for what is where and why. This file is the commands.

## Run it

You need [git](https://git-scm.com), [bun](https://bun.sh), and a clone of
[atlas-recipes](https://github.com/Avarok-Cybersecurity/atlas-recipes), which
the build reads the model list from. From an empty folder, to a running site:

```sh
git clone https://github.com/Avarok-Cybersecurity/atlas-recipes.git
git clone https://github.com/Avarok-Cybersecurity/atlas.git
cd atlas && git checkout site/avarok-facelift     # or main, once it is merged
cd site && bun install

export AVAROK_RECIPES_ROOT="$(cd ../../atlas-recipes/recipes && pwd)"
bun x --bun vite dev                              # http://localhost:5173
```

That one variable is the only setting. Everything the pages show is in the
repository: the fonts, the logos, the product clips, the film and the generated
stills are ordinary files under `static/`, not Git LFS pointers, so a plain
clone has them. This was checked from a clean clone with no GitHub login.

`GIT_LFS_SKIP_SMUDGE=1 git clone ...` makes the clone faster. The repository
keeps some large engine assets in LFS and the site needs none of them.

To build and preview the static output the way it deploys:

```sh
bun x --bun vite build               # writes build/, about two minutes
bun x --bun vite preview             # serves build/ on http://localhost:4173
```

A GitHub login is optional. With `GH_TOKEN=$(gh auth token)` the build refreshes
the star history and the contributor list. Without it those two fall back to
the committed files and the build says so without failing.

On Windows use Git Bash, and give the variable a forward slash path
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
