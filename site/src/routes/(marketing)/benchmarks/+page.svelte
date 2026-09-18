<script>
  // The benchmarks tab across every architecture we have built. The ladder
  // and the gate sections are the same components the developer page and the
  // dashboard modal mount, fed by the same generated JSON, so a number here
  // is a number in the repository.
  import '../../../styles/dashboard.css';
  import '../../../styles/ladder.css';
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import FaqList from '$lib/components/avarok/FaqList.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import ConcurrencyLadder from '$lib/components/ConcurrencyLadder.svelte';
  import GateBenchSection from '$lib/components/GateBenchSection.svelte';
  import GatePointCard from '$lib/components/GatePointCard.svelte';
  import { gateData, tabs, recordsFor, benchName } from '$lib/gates.js';
  import { groupFor, groupRecords, groupedBenches } from '$lib/gate-variants.js';
  import { faqFor } from '$lib/content/faq.js';
  import { routes, links } from '$lib/content/brand.js';
  import { live, ladderData as ladder } from '$lib/content/live.js';
  import bench from '$lib/benchmarks.generated.json';
  import mlperf from '$lib/mlperf.json';
  import { hardwarePage } from '$lib/content/platform.js';

  let activeTab = $state(tabs[0]?.id);
  let selected = $state(null);
  const tab = $derived(tabs.find((t) => t.id === activeTab) ?? tabs[0]);
  const sections = $derived.by(() => {
    const out = [];
    const done = new Set();
    for (const b of tab?.benches ?? []) {
      if (done.has(b)) continue;
      const group = groupFor(b);
      if (group) {
        group.members.forEach((m) => done.add(m.bench));
        const records = groupRecords(group, recordsFor);
        if (records.length > 0) out.push({ benchId: group.primary, name: benchName(group.primary), records });
      } else {
        done.add(b);
        const records = recordsFor(b);
        if (records.length > 0) out.push({ benchId: b, name: benchName(b), records });
      }
    }
    return out;
  });
  const src = gateData.sources;
  const mlperfLine = { preparing: 'MLPerf Inference v6.1 submission in preparation.', submitted: 'MLPerf Inference v6.1 submitted, closed edge division, on both GB10 and gfx1151 from the same CUDA source. Results are under embargo until MLCommons publishes them, and they render here the moment that happens.', published: 'Published in MLPerf Inference v6.1 across NVIDIA GB10 and AMD gfx1151.' }[mlperf.status];
</script>

<PageShell path={routes.benchmarks}>
  <PageHero eyebrow="Benchmarks" title="Every number is a receipt." lede="The website is a build artifact of the repository. The ladder comes from the published campaign, the gate records from committed baselines across every branch, stamped with commit and date. If a number is not in the repo, it is not on this page." primary={{ text: 'Reproduce it yourself', href: routes.diligence }} secondary={{ text: 'The campaign log', href: links.ladderLog, external: true }} color="green">
    <div class="av-grid av-grid-4 av-bench-stats">
      <div class="av-tile"><span class="av-tile-label">At C={live.c}</span><div class="av-num">{live.ratio}</div><p>the matched vLLM configuration, same GB10</p></div>
      <div class="av-tile"><span class="av-tile-label">Aggregate</span><div class="av-num">{live.atlasTop}</div><p>tok/s at C={live.c}, {live.checkpoint.split('/').pop()}</p></div>
      <div class="av-tile"><span class="av-tile-label">Rungs</span><div class="av-num">{live.won}/{live.rungs}</div><p>won against matched vLLM + MTP</p></div>
      <div class="av-tile"><span class="av-tile-label">Gate records</span><div class="av-num">{src.committed + src.from_branches}</div><p>{src.committed} committed, {src.from_branches} from {src.branches_scanned} branches</p></div>
    </div>
  </PageHero>

  <section class="av-section av-section-alt" id="ladder">
    <div class="av-container">
      <div class="av-head av-reveal">
        <p class="av-eyebrow av-sx-green">The ladder</p>
        <h2 class="av-h2">{ladder.title}</h2>
        <p class="av-lede">{ladder.subtitle}. {ladder.box.note}.</p>
      </div>
      <div class="av-reveal"><ConcurrencyLadder embedded /></div>
    </div>
  </section>

  <section class="av-section" id="gates">
    <div class="av-container">
      <div class="av-head av-reveal">
        <p class="av-eyebrow av-sx-cyan">The gates</p>
        <h2 class="av-h2">Every gate record, across every branch.</h2>
        <p class="av-lede">{bench.methodology} Union of gate records at build time, so the newest run shows even before its pull request merges. Provenance on every point.</p>
      </div>
      <div class="av-tabs av-reveal" role="tablist" aria-label="Benchmark families">
        {#each tabs as t}
          <button type="button" role="tab" class="av-tab" aria-selected={activeTab === t.id} onclick={() => (activeTab = t.id)}>{t.label}</button>
        {/each}
      </div>
      <div class="av-reveal av-gates">
        {#each sections as s (s.benchId)}
          <GateBenchSection benchId={s.benchId} name={s.name} records={s.records} onselect={(r) => (selected = r)} />
        {/each}
        {#if sections.length === 0}<p class="av-body">No records in this family yet.</p>{/if}
      </div>
      <p class="av-small av-reveal" style="margin-top:1rem">Repro: <code class="av-mono">{bench.repro_cmd}</code> · generated at {gateData.generated_sha}, {gateData.generated_date}</p>
    </div>
  </section>

  <section class="av-section av-section-alt" id="hardware">
    <div class="av-container">
      <div class="av-head av-reveal">
        <p class="av-eyebrow">Architectures</p>
        <h2 class="av-h2">Verified, in bring up, and next.</h2>
        <p class="av-lede">{mlperfLine}</p>
      </div>
      <div class="av-grid av-grid-3 av-reveal">
        {#each [...hardwarePage.verified, ...hardwarePage.bringup] as h}
          <div class="av-card">
            <div class="av-row" style="justify-content:space-between;margin-bottom:0.5rem">
              <p class="av-card-tag" style="margin:0">{h.chip}</p>
              <span class="av-chip {h.status === 'Verified' || h.status === 'MLPerf submitted' ? 'av-chip-green' : 'av-chip-gold'}">{h.status}</span>
            </div>
            <h3 style="font-size:1rem">{h.name}</h3>
            <p style="font-size:0.9rem">{h.body}</p>
          </div>
        {/each}
      </div>
      <p class="av-small av-reveal" style="margin-top:1.2rem">The MLPerf name and logo are trademarks of MLCommons Association. See the footer.</p>
    </div>
  </section>

  <FaqList items={faqFor('benchmarks')} />
  <CtaBand title="Run the ladder on your workload." body="A side by side against your current engine on your hardware inside 48 hours, with the campaign log to match." primary={{ text: 'Book a demo', href: routes.demo }} secondary={{ text: 'Verification walkthrough', href: routes.diligence }} />
</PageShell>

{#if selected}
  <GatePointCard records={selected} onclose={() => (selected = null)} />
{/if}

<style>
  .av-bench-stats { margin-top: 2rem; max-width: 1180px; }
  .av-bench-stats .av-num { font-size: 2rem; }
  .av-gates { display: grid; gap: 1.5rem; }
</style>
