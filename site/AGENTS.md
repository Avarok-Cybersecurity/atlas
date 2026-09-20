# Working on the Avarok site

For a person or an AI agent opening `site/` for the first time. Read this, then open
[SITE-GUIDE.md](SITE-GUIDE.md). Between them you can change anything here without
reading the code first.

## Read in this order

1. **[SITE-GUIDE.md](SITE-GUIDE.md)**, the map. Every page, the file that draws it, the file
   that holds its words, every button and where it goes, every outside link and address and
   the pages that use it, every logo, clip and portrait, and when each last changed. It is
   generated, so it is never out of date on a green build.
2. **[FACELIFT.md](FACELIFT.md)**, the reasons. What was decided and why, the rules the site
   keeps, and the questions still open for the team.
3. **[README.md](README.md)** to run it. `media-brief/` if you are touching video or imagery.
   `static/logos/README.md` if you are touching a logo.

## How the site is put together

- **Words are data.** Every sentence on a marketing page lives in `src/lib/content/*.js`.
  Components under `src/lib/components/avarok/` only draw. Change copy in content, never in markup.
- **Facts are defined once.** Names, addresses, links and licence lines are in
  `src/lib/content/brand.js` (`company`, `links`, `contacts`, `footer`). Each has a revision
  and a date in `guide/ledger.json`. Change one and the unit suite fails until you record it.
- **Numbers are generated.** Performance figures come from `*.generated.json` through
  `src/lib/content/live.js`. Never type one.
- **Routes are named.** Every internal link comes from `routes` in `brand.js`. A route may
  carry an anchor (`/demo#book`), and a test checks the anchor exists.
- **It is two documents.** The marketing pages and the developer pages (`/engine`, `/control`,
  `/diligence`) use different design systems. A link between them is always a full page
  load. `src/lib/route-groups.js` says why. Do not "fix" that.

## The loop for any change

```sh
cd site
export AVAROK_RECIPES_ROOT=/path/to/atlas-recipes/recipes

# 1. find it in SITE-GUIDE.md, edit the source it names
# 2. prove it
bun test --preload ./test-runes.js src/lib        # about one second
bun x --bun vite build                            # about two minutes
bun run guide -- --note "what you changed"        # rewrites the guide from the build
bun x --bun playwright test                       # builds again, then the browser suite
# 3. commit the source, SITE-GUIDE.md and guide/ together
```

A build and the dev server rewrite `src/lib/*.generated.json`. Restore those with
`git checkout -- src/lib/*.generated.json` before committing, unless the data is your change.

Stop the dev server when you are done with it. A fresh `vite dev` rests at 0% of a core, with
a tab attached, through a build and through hot reloads (measured 2026-09-19). One that had
been up for ninety minutes beside five builds and two test runs was found holding four to
eight cores with nothing in its log, and that could not be reproduced. If a machine gets loud,
look for `bun` in the process list first, and restart the dev server. On Windows, stopping
the shell that started it leaves `bun.exe` holding the port, so end that process too.

## Budgets that must hold

| what | budget | what checks it |
| --- | --- | --- |
| Lighthouse, six gated pages | 100 in all four categories, no third party request | CI, `lighthouse/` |
| Accessibility, every page | 100, no skipped heading level | `e2e/marketing.spec.js` |
| Idle CPU, any page | under 5% of one core | `bun run perf:cpu -- <origin> <path>` |
| Service worker precache | under 1 MB of static files, no media | `src/lib/sw/strategy.test.js` |
| Marketing page scripts | no page imports `gates.generated.json` but benchmarks | `bundle-budget.test.js` |
| Requests before first paint | 8 to 10 scripts on a marketing page, 11 to 14 on a developer page, and no developer page loads the marketing components | `e2e/page-weight.spec.js` |

Anything that animates forever uses only `transform` and `opacity`, and stops when it is off
screen (`src/lib/reveal.js` sets `is-live`). An element reset in `avarok.css` is written with
`:where()` so it never outranks a class.

## Rules that are not negotiable

- **The repository is public.** A file pushed here is published at that moment, before any
  review. Nothing confidential goes in: no deck, no plan, no customer name, no credential.
  Commit messages and docs describe the change in engineering terms and stay general.
- **If the site says it about the company, a source says it.** Where the company is based,
  what it pays, how fast it delivers: not ours to guess. `FACELIFT.md` has the list of claims
  a first draft invented and a later audit removed.
- **Nothing is offered that is not released.** The Community Edition is a waitlist.
- **Attributed quotes are verbatim**, and a real person's entry changes only on their word.
- **The product mockup is a private project.** Only its recordings are here.
- **The voice:** plain, short sentences. No em dashes, semicolons or exclamation marks. A test
  enforces it on every string in `content/`.

## Where things are

| to change | edit |
| --- | --- |
| A sentence | the content file named on the page's entry in `SITE-GUIDE.md` |
| A name, an address, a link | `src/lib/content/brand.js`, then `bun run guide` |
| The header menu or the footer | `nav` and `footer` in `brand.js` |
| A price or a tier | `src/lib/content/pricing.js` |
| A form's fields | `demoPage.form`, `waitlistPage.form`, `careers.form` in `company.js` |
| Where forms post | `formEndpoint` in `brand.js`. Empty means they draft an email. The endpoint is the Worker in `deploy/cloudflare/forms-worker/` |
| A team member's line | `team.people` in `company.js`, and only on that person's word |
| A logo | `static/logos/`, and its row in `static/logos/README.md` |
| A clip or a still | `node scripts/media/install.mjs --from <file> --as <slot>` |
| A colour | `../web-shared/avarok-tokens.css`. Both themes. The blog reads it too |
| A page | add it to `routes`, to the registry in `content/index.js`, and a `+page.svelte` |

Use `bun`, not `npm`: `bun install`, `bun run <script>`, `bun test`, `bun x <tool>`. The site is
SvelteKit on Vite with `adapter-static`, and deploys to Cloudflare Pages from
`.github/workflows/site.yml`.
