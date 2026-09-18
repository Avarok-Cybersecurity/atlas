<!--
  The demo request form. With `formEndpoint` set in brand.js it POSTs JSON
  there. Without one (the default on a static host) it composes a prefilled
  email to sales so nothing is stored on the site and nothing is lost. Either
  way the visitor sees the same confirmation.
-->
<script>
  import { demoPage } from '$lib/content/company.js';
  import { contacts, formEndpoint } from '$lib/content/brand.js';

  const f = demoPage.form;
  let values = $state(Object.fromEntries(f.fields.map((x) => [x.name, x.type === 'select' ? x.options[0] : ''])));
  let state = $state('idle'); // idle | sending | sent | error

  function mailto() {
    const body = f.fields.map((x) => `${x.label}: ${values[x.name] || ''}`).join('\n');
    return `mailto:${contacts.sales}?subject=${encodeURIComponent(`Avarok working session, ${values.company || values.name || ''}`)}&body=${encodeURIComponent(body + '\n\nSent from the Avarok website demo form.')}`;
  }

  async function submit(e) {
    e.preventDefault();
    if (!formEndpoint) {
      window.location.href = mailto();
      state = 'sent';
      return;
    }
    state = 'sending';
    try {
      const res = await fetch(formEndpoint, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ...values, source: 'demo' }) });
      state = res.ok ? 'sent' : 'error';
    } catch {
      state = 'error';
    }
  }
</script>

<form class="av-form av-card" onsubmit={submit} aria-labelledby="demo-form-title">
  <h2 id="demo-form-title" class="av-h3">{f.title}</h2>
  {#each f.fields as x}
    <div class="av-field">
      <label for={`demo-${x.name}`}>{x.label}{#if x.required}<span aria-hidden="true"> *</span>{/if}</label>
      {#if x.type === 'select'}
        <select id={`demo-${x.name}`} name={x.name} bind:value={values[x.name]}>{#each x.options as o}<option>{o}</option>{/each}</select>
      {:else if x.type === 'textarea'}
        <textarea id={`demo-${x.name}`} name={x.name} placeholder={x.placeholder} bind:value={values[x.name]}></textarea>
      {:else}
        <input id={`demo-${x.name}`} name={x.name} type={x.type} placeholder={x.placeholder} required={x.required} autocomplete={x.autocomplete} bind:value={values[x.name]} />
      {/if}
    </div>
  {/each}
  <button class="av-btn av-btn-primary av-btn-lg" type="submit" disabled={state === 'sending'}>{state === 'sending' ? 'Sending' : f.submit} <span class="av-arrow">→</span></button>
  {#if state === 'sent'}<p class="av-body" role="status" style="color:var(--green)">{f.thanks}</p>{/if}
  {#if state === 'error'}<p class="av-body" role="alert" style="color:var(--red)">That did not go through. Email {contacts.sales} directly and we will take it from there.</p>{/if}
  <p class="av-small">{formEndpoint ? 'Your details go to the founding team and nowhere else.' : f.fallbackNote}</p>
</form>
