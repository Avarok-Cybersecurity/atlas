# Atlas website branding

The homepage uses the supplied Atlas brand kit: unmodified vector lockups, lavender/cyan/green/gold accents, white and charcoal backgrounds, and the kit's UI gray for body text. The kit's JSON palette is imported by `src/lib/marketing.js`; the existing shared engineering and blog tokens remain in place.

- Navigation: horizontal lockup, 150 px on desktop and 124 px on mobile, with clear space.
- Footer: full Atlas Inference Engine lockup, 244 px wide, with clear space.
- Small engine illustration: the compact mark, appropriate below 48 px.
- Hero: original AI-generated glass artwork refined to the supplied palette. It is decorative and has an empty alternative text.
- UI icons: Lucide icon data with its ISC/MIT notice retained in `src/lib/components/marketing/icons.LICENSE`.

Brand vector masters and palette live in `assets/brand/` at the repository root. `static/brand/` links to the masters so the website ships the same bytes.

The homepage introduces Atlas and links to `/engine.html` for the complete benchmarks, recipes, installation, and chat tools. `/control.html` and `/diligence.html` retain their existing functionality. Existing homepage fragments for verified performance, models, and getting started remain useful summaries. Other technical fragments forward to the matching engine section, with ordinary links available when JavaScript is disabled.

The performance highlight is calculated from `ladder.generated.json`, including its fastest published baseline at the highest measured concurrency. It does not contain independent throughput numbers or assume future results will show an improvement.

Full-page review images are in `review/`.
