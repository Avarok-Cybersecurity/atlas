<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Fleet awareness in the topbar.
  //
  // **Renders nothing at all unless a local agent answered.** Most visitors to
  // this site are not customers; a widget telling them it cannot reach
  // something they have never installed is worse than no widget. The page that
  // exists to explain the agent is the one place that pitches it.
  //
  // **IT NEVER DIALS. It only reports a connection something else opened.**
  //
  // This used to probe on mount, guarded on `storedToken()` — the reasoning being
  // that a stored token proves the browser was paired, so re-dialing prompts
  // nobody. That conflates two unrelated pieces of state. The token lives in
  // localStorage forever; Chrome's Local Network Access permission is a separate
  // per-origin grant that is re-asked on a fresh profile, after a permissions
  // reset, and whenever the earlier prompt was dismissed rather than allowed.
  // So the guard held for a first-time visitor and failed for exactly the people
  // who matter: every operator hit "Access other apps and services on this
  // device" on plain https://atlasinference.io/, before touching anything.
  //
  // A permission prompt belongs to the gesture that needs the permission. There
  // is no gesture in a topbar that renders on first paint.
  //
  // Nothing is lost. `summarize()` already returns `show: false` unless
  // `fleet.mode === 'live'`, so with no probe the pill is simply invisible on the
  // marketing page — which is precisely how an unconnected pill already looked.
  // /control connects explicitly, and because `fleet` is a singleton the pill
  // lights up there and stays lit for any same-tab navigation back here.
  //
  // Nothing discovered on the network leaves this machine. The strings below
  // are counts, and the page they link to holds the rest.
  import { fleet } from '$lib/agent/fleet.svelte.js';
  import { summarize } from '$lib/agent/summary.js';

  // `.html`, not `/control`. adapter-static writes this route to control.html,
  // and the deploy target serves files literally: it does not try the
  // extension, and it has no directory index outside the document root. So
  // /control is the SPA fallback and /control/ is a 500. This is the URL that
  // resolves. If the server ever gains `try_files $uri $uri.html`, this becomes
  // '/control'.
  const CONTROL = '/control.html';

  const view = $derived(summarize(fleet));
</script>

{#if view.show}
  <a class="fp fp-{view.tone}" href={CONTROL} title="Open the control plane">
    <span class="fp-dot" aria-hidden="true"></span>
    <span class="fp-text">{view.label}</span>
    <span class="fp-detail">{view.detail}</span>
  </a>
{/if}
