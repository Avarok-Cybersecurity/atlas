<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The two commands that put an agent on a machine.
  //
  // ONE copy of this guide. It previously existed three times — the launch
  // dialog's guide phase, the control page's no-agent branch, and the literals
  // behind both — which is how the three drifted: two of them named
  // `atlasctl agent run`, which dies with the terminal that started it.
  //
  // `after` continues the numbering, so a caller can add a third step without
  // restarting the list at 1 or hard-coding "3" here.

  import CommandRow from './CommandRow.svelte';
  import { runCommand, startAgentCommand } from '$lib/data.js';

  let { after } = $props();
</script>

<ol class="ld-steps">
  <li>
    <span class="ld-step-n">1</span>
    <div>
      <p class="ld-step-t">Install the launcher</p>
      <CommandRow command={runCommand} />
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
