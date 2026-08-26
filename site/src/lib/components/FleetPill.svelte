<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Fleet awareness in the topbar.
  //
  // **Renders nothing at all unless a local agent answered.** Most visitors to
  // this site are not customers; a widget telling them it cannot reach
  // something they have never installed is worse than no widget. The page that
  // exists to explain the agent is the one place that pitches it.
  //
  // **It only probes at all if this browser has been paired before.** A stored
  // token is proof an operator owns this browser; a visitor has never had one.
  // Without that check the marketing homepage opened a loopback WebSocket on
  // every visit, and the failed connection logged a console error the browser
  // reports natively and no JS can catch — which cost the homepage its
  // best-practices score and told every visitor's devtools about a port they
  // do not run.
  //
  // Nothing is lost: pairing happens on /control, so an operator has a token by
  // the time the pill could tell them anything, and from then on it appears
  // everywhere.
  //
  // One attempt, no retry loop. The /control page is what keeps a session
  // alive, and it shares the same client, so arriving there costs no second
  // connection.
  //
  // Nothing discovered on the network leaves this machine. The strings below
  // are counts, and the page they link to holds the rest.
  import { fleet } from '$lib/agent/fleet.svelte.js';
  import { storedToken } from '$lib/agent/protocol.js';
  import { summarize } from '$lib/agent/summary.js';

  let asked = $state(false);

  $effect(() => {
    if (asked) return;
    asked = true;
    if (!storedToken()) return;
    // Failure is still ordinary — the agent may be stopped — and is not
    // surfaced anywhere.
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
