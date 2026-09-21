<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Dock tab: what the selected node says is running, and what this page has
  // asked the fleet to do this session.
  //
  // The action log is labelled *this session* because that is all it is —
  // page memory, dead with the tab. The dashed footer admits the gap
  // (durable audit) instead of letting session memory impersonate one.

  import { untrack } from 'svelte';
  import { onTarget } from '$lib/agent/verbs.js';
  import { nameOf, refusal } from '$lib/agent/refusal.js';
  import ComingSoon from './ComingSoon.svelte';

  let { fleet, node, nodes = [], log = [] } = $props();

  let running = $state(null);
  let replyOn = $state(null);
  let replyVia = $state(null);
  let problem = $state(null);
  let busy = $state(false);

  const trusted = $derived(
    Boolean(node && (node.isLocal || node.pairing === 'paired' || node.pairing === 'vouched'))
  );
  // Value-stable: `node` is a new object on every 1Hz vitals event; keying the
  // refresh effect on the id keeps it from re-asking Status once a second.
  const nodeId = $derived(node?.id ?? null);

  async function refresh() {
    if (!node || !trusted) return;
    busy = true;
    const res = await fleet.agent.status(onTarget(node));
    busy = false;
    if (res.ok) {
      running = res.reply.running ?? [];
      replyOn = res.reply.on ?? null;
      replyVia = res.reply.via ?? null;
      problem = null;
    } else {
      const r = refusal(
        { error: res.error ?? null, message: res.message ?? null },
        { target: onTarget(node), nodes }
      );
      problem = r.text;
    }
  }

  $effect(() => {
    nodeId;
    running = null;
    problem = null;
    replyOn = null;
    replyVia = null;
    // Untracked: refresh() reads the node object, which would otherwise put
    // the whole fleet list back into this effect's dependencies.
    untrack(() => refresh());
  });

  const when = (at) =>
    new Date(at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
</script>

<div class="dt">
  {#if !node}
    <p class="dt-quiet">No machine is selected.</p>
  {:else if !trusted}
    <p class="dt-quiet">Status comes over the paired channel. Pair this machine first.</p>
  {:else}
    <div class="dt-statushead">
      <h4 class="dt-h">
        Running
        <span class="dt-route-badge">
          on {replyOn ? nameOf(replyOn, nodes) : node.isLocal ? 'this machine' : node.name}{replyVia
            ? ` · via ${nameOf(replyVia, nodes)}`
            : ''}
        </span>
      </h4>
      <button type="button" class="dt-refresh" onclick={refresh} disabled={busy}>
        {busy ? 'Asking…' : 'Refresh'}
      </button>
    </div>

    {#if problem}
      <p class="dt-problem">{problem}</p>
    {:else if running === null}
      <p class="dt-quiet">Asking…</p>
    {:else if running.length === 0}
      <p class="dt-quiet">Nothing is running on this node.</p>
    {:else}
      <table class="dt-table">
        <thead>
          <tr><th>Container</th><th>Recipe</th><th>Status</th></tr>
        </thead>
        <tbody>
          {#each running as r (r.container)}
            <tr>
              <td class="mono">{r.container}</td>
              <td class="mono">{r.recipe ?? '—'}</td>
              <td>{r.status}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}

  <h4 class="dt-h dt-loghead">Actions <span class="dt-cap">this session</span></h4>
  {#if log.length === 0}
    <p class="dt-quiet">No actions taken from this page yet.</p>
  {:else}
    <ul class="dt-loglist">
      {#each log as e (e.at + e.verb + e.outcome)}
        <li class="dt-logrow" class:dt-logbad={!e.ok}>
          <span class="dt-logwhen mono">{when(e.at)}</span>
          <span class="dt-logverb mono">{e.verb}</span>
          <span class="dt-logtarget">{e.target}</span>
          {#if e.route}<span class="dt-logroute">{e.route}</span>{/if}
          <span class="dt-logoutcome">{e.outcome}</span>
        </li>
      {/each}
    </ul>
  {/if}
  <p class="dt-audit"><ComingSoon id="durable-audit" kind="chip" /></p>
</div>
