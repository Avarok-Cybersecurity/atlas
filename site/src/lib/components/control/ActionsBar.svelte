<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C4: the seven verbs, 44px.
  //
  // The bar states where every verb will run BEFORE it runs — "runs on dgx3
  // · via dgx1" at its left edge — because a refusal from dgx3 and a relay
  // failure on dgx1 send the operator to different machines, and the routing
  // should be known before the click, not learned from the error.
  //
  // A verb the target cannot honour renders disabled with the stated reason,
  // never hidden (aria-disabled so it stays focusable; the reason is its
  // description). Stop is the one verb that executes here — it is the verb
  // an operator reaches for in a hurry — and it takes two steps, with the
  // travel warning interposed when the target is reached through a peer.
  // The rest hand off to the console dock's tabs.

  import { VERBS, availability, onTarget, route, travelWarning } from '$lib/agent/verbs.js';
  import { refusal } from '$lib/agent/refusal.js';
  import { placeholdersFor } from '$lib/agent/placeholders.js';
  import ComingSoon from './ComingSoon.svelte';

  let { fleet, node, nodes = [], solo = false, onverb, onlog, onstats } = $props();

  const avail = $derived(availability(node));
  const routeText = $derived(route(node, nodes));
  const travel = $derived(travelWarning(node, nodes));
  const chips = $derived(placeholdersFor('actions', { solo }));

  let confirmingStop = $state(false);
  let stopping = $state(false);

  // The id, not the object: `node` is a new object on every 1Hz vitals event,
  // and an effect keyed on it would disarm the Stop confirm once a second —
  // an operator could never reach the second step on a live fleet.
  const nodeId = $derived(node?.id ?? null);
  $effect(() => {
    // A new selection is a new question; a half-armed Stop must not carry over.
    nodeId;
    confirmingStop = false;
  });

  function log(verb, outcome, ok) {
    onlog?.({ verb, target: node?.isLocal ? 'this machine' : (node?.name ?? ''), route: routeText, outcome, ok });
  }

  /**
   * The 's' hotkey's door: identical to pressing the Stop button, two steps
   * included — first press arms (with the travel warning when routed), the
   * second confirms. A disabled Stop stays disabled from the keyboard too.
   */
  export function armStop() {
    if (!avail.stop?.enabled) return;
    if (!confirmingStop) {
      confirmingStop = true;
      return;
    }
    doStop();
  }

  async function doStop() {
    if (!node?.running) return;
    stopping = true;
    const res = await fleet.agent.stop(node.running, onTarget(node));
    stopping = false;
    confirmingStop = false;
    if (res.ok) {
      log('stop', `stopped ${res.reply.recipe}`, true);
    } else {
      const r = refusal(
        { error: res.error ?? null, message: res.message ?? null },
        { target: onTarget(node), nodes }
      );
      log('stop', r.text, false);
    }
    onverb?.('status');
  }

  function press(verb) {
    const a = avail[verb.id];
    if (!a?.enabled) return;
    if (verb.id === 'stop') {
      if (!confirmingStop) {
        confirmingStop = true;
        return;
      }
      doStop();
      return;
    }
    if (verb.id === 'stats') {
      onstats?.();
      log('stats', 'poll requested', true);
      return;
    }
    onverb?.(verb.id);
  }
</script>

<div class="ab" aria-label="Actions">
  <span class="ab-route">
    {#if confirmingStop}
      <span class="ab-warn">
        {travel ? `${travel} ` : ''}Stop {node?.running}?
      </span>
    {:else}
      {routeText}
    {/if}
  </span>

  {#each VERBS as verb (verb.id)}
    {@const a = avail[verb.id]}
    {#if verb.id === 'stop' && confirmingStop}
      <button type="button" class="ab-btn ab-danger" onclick={() => press(verb)} disabled={stopping}>
        {stopping ? 'Stopping…' : 'Confirm stop'}
      </button>
      <button type="button" class="ab-btn" onclick={() => (confirmingStop = false)}>Cancel</button>
    {:else}
      <button
        type="button"
        class="ab-btn"
        class:ab-off={!a.enabled}
        aria-disabled={!a.enabled}
        aria-describedby={a.enabled ? undefined : `ab-why-${verb.id}`}
        title={a.enabled ? undefined : a.reason}
        onclick={() => press(verb)}
      >
        {verb.label}
      </button>
      {#if !a.enabled}
        <span class="visually-hidden" id={`ab-why-${verb.id}`}>{a.reason}</span>
      {/if}
    {/if}
  {/each}

  <span class="ab-hr" aria-hidden="true"></span>

  {#each chips as chip (chip.id)}
    {#if chip.id === 'soon-menu'}
      <!-- Solo mode: the chips collapse into one, so a single-machine console
           never reads as a roadmap. The popover still names every missing
           capability, one sentence each.

           `ComingSoon` rather than a second popover written here: this one was
           hand-rolled and skipped the whole ceremony — the dialog never took
           focus, so it was never announced; Tab left it; Escape only worked
           while the BUTTON still had focus; and nothing closed it on click-out. -->
      <ComingSoon ids={chip.collapsed.map((e) => e.id)} label={chip.label} kind="chip" />
    {:else}
      <ComingSoon id={chip.id} kind="chip" />
    {/if}
  {/each}
</div>
