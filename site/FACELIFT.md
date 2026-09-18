# The Avarok facelift: handoff

September 2026. Branch `site/avarok-facelift`.

This document is for whoever touches the marketing site next: a person, a
different model, a team of agents, or someone deciding to throw it all away. It
says what was built, why each decision went the way it did, where every kind of
change is made, and what is still open. Start here, then read `README.md` for
the commands.

## What changed, in one paragraph

The front page used to sell a local inference engine to developers. It now
sells an enterprise platform to the people who own GPUs: faster inference,
stronger governance, a fraction of what they pay today. The brand is Avarok.
The developer pages (`/engine`, `/control`, `/diligence`) are intact under the
new header, because they are where every number on the new pages comes from.
Around them there are now thirty marketing pages, a payback model, six product
clips recorded from a mockup of the enterprise console, a one minute product
film, and generated imagery and footage made from a prompt pack that ships
with the site.

## The order of the story

The brief asked for a site that tells a buyer or an investor an immediate
story, modelled on how artemissecurity.com is laid out. The front page runs in
that order, one component per beat, in `src/lib/components/avarok/home/`:

| Beat | Component | What it has to do |
| --- | --- | --- |
| Announcement | `Announcement` | One measured fact, linked to the proof |
| Hero | `Hero` | Say what the product is. Show the product. One call to action |
| Who built it | `LogoWall` | Prior roles of the team. No customer logos, we have none |
| The problem | `Problem` | GPUs report tokens per second. The CFO pays per workload |
| The platform | `Solution` | Three layers and how a request passes through them |
| Speed, security, governance | `ValueBand` | One card each, each linking to its page |
| Proof | `Proof` | The published ladder, live from the repository |
| The console | `Tour` | Five recorded clips, one tab each: ask, queue, fleet, economics, governance |
| Recognition | `Recognition` | NVIDIA Inception, MLCommons, Hugging Face, AMD |
| Three differences | `Differences` | Claims a buyer can test |
| The chain | `Chain` | Each layer earns the next |
| Deliveries | `Deliveries` | Week one, months one to six, at renewal |
| Voices | `Voices` | Community quotes, verbatim |
| Questions | `FaqList` | Deployment time, data ownership, pricing |
| Next step | `CtaBand` | Everything loops back to the demo |

## Where things live

```
site/src/lib/content/        every word, link and price. Edit here, not in components
  brand.js                   names, routes, nav tree, footer, contacts, form endpoint
  index.js                   the page registry: title and description for every page
  home.js why.js platform.js solutions.js pricing.js company.js resources.js faq.js
  live.js                    the generated numbers the copy prints, and fill()
  media.js  art.json         video slots, and stills installed from the prompt pack
site/src/lib/components/avarok/   the marketing components, prefix av-
site/src/lib/economics.js    the payback model, pure functions, tested
site/src/lib/broll/          procedural ambient loops
site/src/styles/avarok.css   the marketing design system
site/src/routes/(marketing)/ the marketing pages, thin: they pick content and components
site/src/routes/(engine)/    the developer pages, unchanged apart from the header
site/src/routes/(app)/       full viewport render pages for the media pipeline
site/scripts/media/          record, encode, install, film, social card, prompt sheet
site/media-brief/            the prompt pack, the takes ledger, the film's cut, the reasoning
web-shared/                  tokens, theme switch, the lockup. Shared with the blog
assets/brand/                vector masters
```

The three route groups exist so two stylesheets never meet. The developer pages
load `app.css`, the marketing pages load `avarok.css`, and SvelteKit only ships
a group's CSS to that group's pages.

## How to change things

| To change | Edit |
| --- | --- |
| Any sentence on any marketing page | The matching file in `src/lib/content/` |
| A page title or description | `src/lib/content/index.js` |
| The nav or the footer | `nav` and `footer` in `src/lib/content/brand.js` |
| A price, a tier, a feature list | `src/lib/content/pricing.js` |
| The payback model's defaults | `FLEET_DEFAULTS` and `API_DEFAULTS` in `src/lib/economics.js` |
| A contact address | `contacts` in `src/lib/content/brand.js` |
| Where the demo form posts | `formEndpoint` in `src/lib/content/brand.js`. Empty means it composes an email |
| The team's prior employers | `logoWall` in `src/lib/content/home.js`, SVGs in `static/logos/` |
| A product name | `company` in `src/lib/content/brand.js`. Copy uses `{engine}` and friends |
| An industry | `industries` in `brand.js`, and its entry in `solutions.js` |
| A video | `node scripts/media/install.mjs --from <file> --as <slot>` |
| Which clip opens a page | `heroClips` in `src/lib/content/media.js` |
| The film | `media-brief/reel.json`, then `bun run reel` |
| A page hero image | Same command with an image. It registers itself in `art.json` |
| A colour | `web-shared/avarok-tokens.css`. Both themes. The blog reads it too |
| The logo | `assets/brand/` masters, then `web-shared/components/AtlasLockup.svelte` |

To add a page: add its path to `routes`, its title to the registry, a
`+page.svelte` under `(marketing)`, and a nav entry if it should be found. The
tests in `src/lib/content/site.test.js` fail until all four agree.

## Rules the site keeps

**Numbers are generated, never typed.** Every performance figure comes from
`ladder.generated.json`, `live.generated.json`, `models.generated.json` or
`stars.generated.json`, through `live.js`. Copy says `{ratio}` and `{c}`, and
`fill()` substitutes. When the ladder is regenerated the front page follows.

**Every input to the payback model says what it is.** `MEASURED` comes from the
published ladder. `PROPOSED` is a price we have proposed and can change. `USER`
is the visitor's to edit. The 70% claim on the front page is the second
scenario with its defaults, and the page says "modeled".

**Prices are proposed.** The pricing page says so on every tier. They came from
the go to market plan and have not been approved as a public list.

**No customer logos.** The logo wall shows where the team worked before, under
a line that says so, with a note that these are not customers or endorsements.
U.S. Cyber Command and Naval Special Warfare appear as text because their
insignia are restricted. `static/logos/README.md` records each mark's source.

**Attributed quotes are verbatim.** Two community quotes use the engine's old
name. They stay as written, with a note under them. A quote is never edited to
follow a rebrand.

**The voice.** No em dashes, no semicolons, no exclamation marks, short
sentences. `site.test.js` enforces it on every string in `content/`.

**No third party requests.** Fonts, logos and media are self hosted. The
Lighthouse gate requires it and `e2e/marketing.spec.js` checks it.

**The marketing pages stay light.** `gates.generated.json` is a megabyte. Only
the benchmarks page may import it. `bundle-budget.test.js` holds that line,
after the first build of this branch put it on twelve pages by accident.

## The product mockup is not in this repository

The brief said to mock up the enterprise product, keep it out of the public
repo, work locally, and screen record it. So the console you see in the clips
is a separate private project on Alexi's machine, and only its recordings are
here. `media-brief/README.md` says where it is and how to re-record. Nothing on
the public site links to an interactive console, and a test checks that.

## The brand

The engine was named Atlas until September 2026. The rebrand reached: every
marketing page, the developer pages' visible copy, the blog's chrome, the
lockup, the social card, `llms.txt`, the JSON-LD and the web manifest.

It deliberately did not reach:

- **Blog posts.** Dated articles with permanent URLs keep the name they were
  written under. Only the blog's chrome changed.
- **The diligence deck's evidence labels.** They mirror published campaign
  records that say Atlas. The deck's stamp says so on every slide.
- **Commands, crates, images, URLs.** `atlasctl`, `avarok/atlas-gb10`, the
  repository name and the domain are what they are until someone renames them.
- **The legal entity.** Atlas Cybernetics Corp., in the footer and the corp
  lockup, until the company says otherwise.

The wordmark: the brand kit's arrow A is unchanged, byte for byte. The letters
"varok" were set in the kit's own typeface and fitted letter by letter to the
approved Avarok artwork. `assets/brand/src/wordmark-paths.json` holds the
outlines and the numbers. The site's type is IBM Plex Sans and Mono, self
hosted, because the brand kit's slide and letterhead templates use them.

## Tests and gates

| Check | Command | What it holds |
| --- | --- | --- |
| Unit | `bun test --preload ./test-runes.js src/lib` | Content integrity, the payback model, loop seams, the prompt pack, bundle budget |
| Browser | `bun x --bun playwright test e2e/marketing.spec.js` | Menus, tabs, calculator, theme, form, every page, no third party requests |
| Contrast | `bun .contrast-check.mjs` from the repository root | Text contrast in both themes |
| Titles | In `.github/workflows/site.yml` | Each route kept its own title |
| Cross links | `bun blog/e2e/check-crosslinks.mjs site/build blog/build` | The blog's links into this site resolve |
| Lighthouse | In CI | Performance, accessibility, best practices and SEO at 1.0 |

## Decisions, and why

- **Developer pages kept, not rewritten.** They are the evidence. The new pages
  point at them for every claim. The old nav became a section bar under the
  new header (`src/styles/engine-shell.css`).
- **One masterbrand, descriptive product names.** Avarok Engine, Avarok
  Control, Avarok Economics, Avarok Console. The suite name is still being
  decided, and `company` in `brand.js` is the one place to change it.
- **The form composes an email.** A static host has nowhere to post to. Set
  `formEndpoint` and the same form posts JSON instead.
- **Procedural b-roll first, generated footage over it.** The loops drawn by
  code were the placeholder of known quality. Grok's first run on 2026-09-18
  replaced them, and `media-brief/takes/TAKES.md` is the ledger of that run.
  `media-brief/RATIONALE.md` has the full reasoning.
- **The mp4 is listed before the webm.** For flat interface footage H.264 came
  out smaller than VP9 at the same legibility.
- **A mouse click never closes a hover opened menu.** People hover, then click.
  A plain toggle shut the menu under their cursor. Keyboard and touch toggle.

## Open questions for the team

1. **Prices.** Are the proposed list prices approved to be public?
2. **Sales contact.** The site uses Kyle's address from the deck. A `sales@`
   alias would keep a personal inbox off a public page. One line in `brand.js`.
3. **Form endpoint.** Do we want demo requests in a CRM? Then set `formEndpoint`.
4. **The legal name.** Does the corp lockup stay "Atlas Cybernetics Corp"?
5. **Author titles on the blog.** Changed from Atlas to Avarok. The people
   named should confirm their own.
6. **Founders section.** Skipped for now on instruction. The company page has
   the story, not the people.
7. **Repository README and docs book.** Still say Atlas. Out of scope here.
8. **GitHub social preview images** in `assets/brand/` still show the old
   wordmark. `scripts/media/og.mjs` is the pattern to regenerate them.
9. **"jev".** The brief names it beside Bend as a Labs topic. Nothing in the
   source material says what it is, so it is not on the page. Bend is. One
   line in `labs.tracks` in `src/lib/content/resources.js` adds it.
10. **Raw video takes.** About 34 MB of generated footage sits uncommitted in
    `media-brief/takes/`. Committing it means LFS. The team's call.
11. **Cloudflare Pages paths.** The build writes both `platform.html` and a
   `platform/` directory of child pages. Pages serves `/platform` from the
   file. Worth one look on the preview deployment.
12. **Domain.** `atlascybernetics.ai` is unchanged. `SITE` in `brand.js` is the
    one constant to move when DNS does.

## How to throw it away

The old front page is in git history at the commit before this branch. The new
pages are confined to `src/routes/(marketing)`, `src/lib/content`,
`src/lib/components/avarok`, `src/lib/broll`, `src/styles/avarok.css`,
`scripts/media` and `media-brief`. Deleting those and restoring
`src/routes/+page.svelte` and `+layout.svelte` from history returns the site to
where it was. The lockup, the developer page rename and the blog rename are
commits of their own for the same reason.
