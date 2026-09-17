<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import { moveTab } from '$lib/tablist.js';
  import { guideUrl } from '$lib/data.js';
  import Icon from './Icon.svelte';
  const scenarios = [
    { id:'agents', label:'Agentic experiences', icon:'Terminal', headline:'Give your agents room to think.', description:'From the first instruction to the next tool call, keep the work moving. Build coding copilots and multi-step workflows on an engine that speaks your language.', tags:['Tool calling','Streaming responses','OpenAI-compatible API'], prompt:'Turn a big idea into a working prototype.', steps:['Understand the goal','Call the right tools','Keep the conversation moving'], output:'From “what if” to what’s next.', badge:'AGENT WORKFLOW' },
    { id:'private', label:'Private intelligence', icon:'LockKeyhole', headline:'Keep your intelligence close.', description:'Bring open models to the infrastructure you control. Create assistants for your team, your knowledge, and the work that matters to you.', tags:['Self-hosted inference','Open-weight models','Your infrastructure'], prompt:'Make our internal knowledge useful.', steps:['Connect your application','Run a model on your hardware','Bring answers to your team'], output:'Your knowledge. Your environment.', badge:'PRIVATE ASSISTANT' },
    { id:'research', label:'Unrestricted curiosity', icon:'FlaskConical', headline:'Make space for the next discovery.', description:'Explore a model. Test an idea. Go one level deeper. With an open engine and published recipes, the details are yours to understand and improve.', tags:['Source available','Model recipes','Reproducible benchmarks'], prompt:'Find out what this model can really do.', steps:['Choose a supported model','Run a published recipe','Explore, measure, iterate'], output:'More possibilities to explore.', badge:'RESEARCH WORKFLOW' }
  ];
  let selected = $state(0);
  function handleKey(event, index) {
    const next = moveTab(event.key, index, scenarios.length);
    if (next === null) return;
    event.preventDefault();
    selected = next;
    event.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next].focus();
  }
</script>
<div class="m-possibility-tabs">
  <div class="m-scenario-list" role="tablist" aria-label="Explore what you can build">
    {#each scenarios as scenario, index}
      <button type="button" class="m-scenario-tab" role="tab" id={`possibility-tab-${scenario.id}`} aria-selected={selected === index} aria-controls={`possibility-panel-${scenario.id}`} tabindex={selected === index ? 0 : -1} onclick={() => selected = index} onkeydown={event => handleKey(event, index)}>
        <Icon name={scenario.icon} />{scenario.label}<Icon name="ArrowUpRight" size={16} />
      </button>
    {/each}
  </div>
  {#each scenarios as scenario, index}
    <div class="m-scenario-panel" role="tabpanel" id={`possibility-panel-${scenario.id}`} aria-labelledby={`possibility-tab-${scenario.id}`} hidden={selected !== index} tabindex="0">
      <div class="m-scenario-copy">
        <h3>{scenario.headline}</h3><p>{scenario.description}</p>
        <ul>{#each scenario.tags as tag}<li><Icon name="Check" size={15} />{tag}</li>{/each}</ul>
        <a class="m-text-link" href={guideUrl} target="_blank" rel="noreferrer">Explore the documentation <Icon name="ArrowUpRight" size={17} /></a>
      </div>
      <div class="m-workflow">
        <div class="m-workflow-top"><span><span class="m-status-dot"></span>{scenario.badge}</span><span>ILLUSTRATIVE FLOW</span></div>
        <div class="m-prompt-bubble">{scenario.prompt}<Icon name="ArrowUpRight" size={16} /></div>
        <div class="m-workflow-engine"><img src="/brand/mark-compact.svg" alt="" width="28" height="28" /><span>Powered by Atlas</span><span class="m-engine-dot"></span></div>
        <ol>{#each scenario.steps as step}<li><span class="m-flow-check"><Icon name="Check" size={12} /></span>{step}</li>{/each}</ol>
        <div class="m-workflow-result"><Icon name="Sparkles" /><span>{scenario.output}</span><Icon name="ArrowRight" /></div>
      </div>
    </div>
  {/each}
</div>
