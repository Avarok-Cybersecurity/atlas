# Avarok website branding

The homepage uses the supplied Avarok brand kit: unmodified vector lockups, lavender/cyan/green/gold accents, and the kit's UI gray for body text. Light and dark themes share the chevron hues and swap ground/ink; `data-theme` is set before first paint from `localStorage` (`avarok-theme`) or `prefers-color-scheme`. The kit's JSON palette is imported by `src/lib/marketing.js`; the existing shared engineering and blog tokens remain in place.

- Navigation: horizontal lockup, 166 px on desktop and 138 px on mobile, with clear space.
- Footer: full Avarok Inference Engine lockup, 253 px wide, with clear space.
- Small engine illustration: the compact mark, appropriate below 48 px.
- Hero: original AI-generated glass artwork refined to the supplied palette. It is decorative and has an empty alternative text.
- UI icons: Lucide icon data with its ISC/MIT notice retained in `src/lib/components/marketing/icons.LICENSE`.

Brand vector masters and palette live in `assets/brand/` at the repository root. `static/brand/` links to the masters so the website ships the same bytes.

The homepage introduces Avarok and links to `/engine` (`https://atlascybernetics.ai/engine`) for the complete benchmarks, recipes, installation, and chat tools. `/control` and `/diligence` retain their existing functionality. Existing homepage fragments for verified performance, models, and getting started remain useful summaries. Other technical fragments forward to the matching engine section, with ordinary links available when JavaScript is disabled.

The performance highlight is calculated from `ladder.generated.json`, including its fastest published baseline at the highest measured concurrency. It does not contain independent throughput numbers or assume future results will show an improvement.

Full-page review images are in `review/`.
