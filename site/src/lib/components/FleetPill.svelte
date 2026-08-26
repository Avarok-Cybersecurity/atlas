<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Fleet awareness in the topbar.
  //
  // **Renders nothing at all unless a local agent answered.** Most visitors to
  // this site are not customers; a widget telling them it cannot reach
  // something they have never installed is worse than no widget. The page that
  // exists to explain the agent is the one place that pitches it.
  //
  // One attempt, no retry loop. A marketing page must not poll loopback
  // forever, so this asks once — the /control page is what keeps a session
  // alive, and it shares the same client, so arriving there costs no second
  // connection.
  //
  // Nothing discovered on the network leaves this machine. The strings below
  // are counts, and the page they link to holds the rest.
  import { fleet } from '$lib/agent/fleet.svelte.js';
  import { summarize } from '$lib/agent/summary.js';

  let asked = $state(false);

  $effect(() => {
    if (asked) return;
    asked = true;
    // Failure is the ordinary case — no agent installed — and is not surfaced.
    fleet.start({ watch: false }).catch(() => {});
  });

  const view = $derived(summarize(fleet));
</script>

{#if view.show}
  <a class="fp fp-{view.tone}" href="/control" title="Open the control plane">
    <span class="fp-dot" aria-hidden="true"></span>
    <span class="fp-text">{view.label}</span>
    <span class="fp-detail">{view.detail}</span>
  </a>
{/if}
