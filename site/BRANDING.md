# Atlas website branding

The homepage uses the supplied Atlas brand kit: unmodified vector lockups, lavender/cyan/green/gold accents, and the kit's UI gray for body text. Light and dark themes share the chevron hues and swap ground/ink; `data-theme` is set before first paint from `localStorage` (`avarok-theme`) or `prefers-color-scheme`. The kit's JSON palette is imported by `src/lib/marketing.js`; the existing shared engineering and blog tokens remain in place.

- Navigation: the corporate lockup (mark + wordmark + "Cybernetics Corp"), 236 px on desktop and 196 px on mobile, with clear space. It is one inline SVG whose two greys are `--logo-word` and `--logo-tagline`, so the theme swap costs no second file and no second request.
- Footer: the same corporate lockup, 253 px wide, with clear space.
- Narrow bars (`/engine`, the blog header, anything under 220 px of room): the horizontal lockup, as the guidelines prescribe — the corporate tagline stops being legible below that width.
- Small engine illustration: the compact mark, appropriate below 48 px.
- Hero: original AI-generated glass artwork refined to the supplied palette. It is decorative and has an empty alternative text. Two renders: `atlas-hero.webp` (near-white ground, multiplied onto the light page; the PNG source is `assets/brand/atlas-hero.png`) and `atlas-hero-dark.webp`, a re-toned derivative (lightness inverted in Lab with hue kept, mids lifted, chroma raised, ground levelled to the dark `--bg`), because no blend mode can lift a white ground off a dark page. The page carries one `<img>`; the boot script in `app.html` preloads the render the theme calls for and an inline pick sets its `src` while the page parses, so a load fetches one render. Regenerate the dark one if the dark `--bg` token changes.
- UI icons: Lucide icon data with its ISC/MIT notice retained in `src/lib/components/marketing/icons.LICENSE`.

Brand vector masters and palette live in `assets/brand/` at the repository root. `static/brand/` links to the masters so the website ships the same bytes. The corporate lockup (`logo-full-corp.svg`, `logo-full-corp-ondark.svg`) is the Atlas Cybernetics Corp kit's own artwork with its live `<text>` tagline outlined, so it renders identically on a machine without the kit's font; the kit originals, provenance intact, are in `assets/brand/kit/`. The favicons, app icons and the social card come from the same kit — the card is a kit export, not a file this repo generates, which is why the old `og-image.svg` source is gone.

The homepage introduces Atlas and links to `/engine` (`https://atlascybernetics.ai/engine`) for the complete benchmarks, recipes, installation, and chat tools. `/control` and `/diligence` retain their existing functionality. Existing homepage fragments for verified performance, models, and getting started remain useful summaries. Other technical fragments forward to the matching engine section, with ordinary links available when JavaScript is disabled.

The performance highlight is calculated from `ladder.generated.json`, including its fastest published baseline at the highest measured concurrency. It does not contain independent throughput numbers or assume future results will show an improvement.

Full-page review images are in `review/`.
