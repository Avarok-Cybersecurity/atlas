<!--
  The stand in for a customer story until there is one: the box we can prove.
  A small ladder chart drawn from ladder.generated.json, the same numbers the
  developer page and llms.txt print.
-->
<script>
  import { proof } from '$lib/content/home.js';
  import { fill, ladderData as ladder } from '$lib/content/live.js';
  import { routes } from '$lib/content/brand.js';

  const rows = [...ladder.rows].sort((a, b) => a.c - b.c);
  const best = (r) => r.baselines.find((b) => b.id === r.best_baseline_id)?.tok_s ?? 0;
  const W = 560, H = 240, PL = 48, PR = 16, PT = 16, PB = 30;
  const maxC = Math.max(...rows.map((r) => r.c));
  const vMax = Math.max(...rows.map((r) => Math.max(r.atlas, best(r)))) * 1.08;
  const x = (c) => PL + (Math.log2(c) / Math.log2(maxC)) * (W - PL - PR);
  const y = (v) => PT + (1 - v / vMax) * (H - PT - PB);
  const line = (f) => rows.map((r, i) => `${i ? 'L' : 'M'}${x(r.c).toFixed(1)} ${y(f(r)).toFixed(1)}`).join(' ');
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => vMax * t);
</script>

<section class="av-section" id="proof">
  <div class="av-container">
    <div class="av-split av-split-wide">
      <div class="av-reveal">
        <p class="av-eyebrow av-sx-green">{proof.eyebrow}</p>
        <h2 class="av-h2">{proof.title}</h2>
        <p class="av-lede">{proof.body}</p>
        <div class="av-proof-stats">
          {#each proof.stats as s}
            <div>
              <span class="av-num av-num-plain">{fill(s.value)}<small>{s.unit ? ` ${s.unit}` : ''}</small></span>
              <span class="av-small">{fill(s.label)}</span>
            </div>
          {/each}
        </div>
        <div class="av-row" style="margin-top:1.75rem">
          <a class="av-btn av-btn-secondary" href={proof.primary.href} target="_blank" rel="noopener">{proof.primary.text} ↗</a>
          <a class="av-link" href={proof.secondary.href}>{proof.secondary.text} <span class="av-arrow">→</span></a>
        </div>
      </div>
      <div class="av-card av-reveal av-ladder-card">
        <div class="av-row" style="justify-content:space-between">
          <span class="av-kicker"><span class="av-dot"></span> Published GB10 ladder</span>
          <a class="av-small av-link-quiet" href={routes.diligence}>reproduce it</a>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} class="av-ladder" role="img" aria-label={`Aggregate tokens per second, Avarok against the matched vLLM configuration, from C=1 to C=${maxC}`}>
          {#each ticks as t}
            <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} class="grid" />
            <text x={PL - 6} y={y(t) + 4} text-anchor="end" class="axis">{Math.round(t)}</text>
          {/each}
          {#each rows as r}
            <text x={x(r.c)} y={H - 8} text-anchor="middle" class="axis">C={r.c}</text>
          {/each}
          <path d={line(best)} class="base" />
          <path d={line((r) => r.atlas)} class="atlas" />
          {#each rows as r}
            <circle cx={x(r.c)} cy={y(r.atlas)} r="4" class="atlas-dot" />
          {/each}
          <text x={W - PR} y={y(rows.at(-1).atlas) - 10} text-anchor="end" class="lab atlas-lab">Avarok {rows.at(-1).atlas.toFixed(0)} tok/s</text>
          <text x={W - PR} y={y(best(rows.at(-1))) + 18} text-anchor="end" class="lab base-lab">vLLM + MTP {best(rows.at(-1)).toFixed(0)} tok/s</text>
        </svg>
        <p class="av-small">{ladder.workload.checkpoint} · {ladder.box.gpu} · {ladder.aggregate}. ISL {ladder.workload.isl_tokens} / OSL {ladder.workload.osl_tokens.toLocaleString('en-US')}, temperature {ladder.workload.temperature}. Every rung in the campaign log.</p>
      </div>
    </div>
  </div>
</section>

<style>
  .av-proof-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 1.75rem; }
  .av-proof-stats > div { display: grid; gap: 0.3rem; padding-top: 0.9rem; border-top: 1px solid var(--border-strong); }
  .av-proof-stats .av-num { font-size: 1.8rem; }
  .av-proof-stats small { font-size: 0.85rem; font-weight: 500; color: var(--t3); letter-spacing: 0; }
  .av-ladder { width: 100%; height: auto; margin: 0.9rem 0 0.6rem; }
  .av-ladder .grid { stroke: var(--border); stroke-width: 1; }
  .av-ladder .axis { fill: var(--t3); font-family: var(--font-mono); font-size: 10px; }
  .av-ladder .base { fill: none; stroke: var(--t3); stroke-width: 2; stroke-dasharray: 5 4; }
  .av-ladder .atlas { fill: none; stroke: var(--accent); stroke-width: 3; }
  .av-ladder .atlas-dot { fill: var(--accent); }
  .av-ladder .lab { font-family: var(--font-mono); font-size: 11px; font-weight: 700; }
  .av-ladder .atlas-lab { fill: var(--accent); }
  .av-ladder .base-lab { fill: var(--t3); }
  @media (max-width: 560px) { .av-proof-stats { grid-template-columns: minmax(0, 1fr); } }
</style>
