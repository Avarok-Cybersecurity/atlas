<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The two commands that put an agent on a machine.
  //
  // ONE copy of this guide. It previously existed three times — the launch
  // dialog's guide phase, the control page's no-agent branch, and the literals
  // behind both. They happened to agree when they were merged; nothing kept
  // them agreeing, and a fourth was about to be added by the join guide.
  //
  // `startAgentCommand` is `agent install`, not `agent run`, deliberately: a
  // bare `run` dies with the terminal that started it, and the machine leaves
  // the fleet the next time someone closes an ssh session.
  //
  // `after` continues the numbering, so a caller can add a third step without
  // restarting the list at 1 or hard-coding "3" here.

  import CommandRow from './CommandRow.svelte';
  import { startAgentCommand } from '$lib/data.js';
  import { currentInstall } from '$lib/install/host.svelte.js';

  let { after } = $props();

  // Not `runCommand`: on Windows that line cannot run at all, and this guide is
  // the first thing a new machine's operator is asked to paste.
  const install = $derived(currentInstall());
</script>

<ol class="ld-steps">
  <li>
    <span class="ld-step-n">1</span>
    <div>
      <p class="ld-step-t">Install the launcher</p>
      <CommandRow command={install.command} />
    </div>
  </li>
  <li>
    <span class="ld-step-n">2</span>
    <div>
      <p class="ld-step-t">Start the agent in the background</p>
      <CommandRow command={startAgentCommand} />
    </div>
  </li>
  {@render after?.()}
</ol>
<!--
  The commonest reason someone is looking at this guide is NOT that they have
  never installed atlasctl: it is that they installed it, the agent is not
  running, and this page says it cannot see one. Both commands are safe to
  re-run and the second starts a stopped agent — but nothing said so, so a
  returning operator reads step 1 as "the page has not noticed".
-->
<p class="ld-steps-note">
  Already installed? Run them anyway — the first keeps the version you have,
  and the second starts an agent that is installed but stopped.
</p>
