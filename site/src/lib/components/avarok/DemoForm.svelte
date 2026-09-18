<!--
  The site's one form. The demo page renders it with its defaults and the
  waitlist page passes its own definition, so both behave the same way.

  With `formEndpoint` set in brand.js it POSTs JSON there, tagged with `source`
  so one endpoint can tell a demo request from a waitlist entry. Without one
  (the default on a static host) it composes a prefilled email to `to`, so
  nothing is stored on the site and nothing is lost. Either way the visitor
  sees the same confirmation.

  To add a third form: write a `{ title, fields, submit, thanks, fallbackNote }`
  object next to demoPage.form in content/company.js and pass it as `form`.
-->
<script>
  import { demoPage } from '$lib/content/company.js';
  import { contacts, formEndpoint } from '$lib/content/brand.js';

  let {
    form: f = demoPage.form,
    source = 'demo',
    to = contacts.sales,
    // the subject of the composed email, given the values typed so far
    subject = (v) => `Avarok working session, ${v.company || v.name || ''}`
  } = $props();

  let values = $state(Object.fromEntries(f.fields.map((x) => [x.name, x.type === 'select' ? x.options[0] : ''])));
  let state = $state('idle'); // idle | sending | sent | error

  function mailto() {
    const body = f.fields.map((x) => `${x.label}: ${values[x.name] || ''}`).join('\n');
    return `mailto:${to}?subject=${encodeURIComponent(subject(values))}&body=${encodeURIComponent(body + `\n\nSent from the Avarok website ${source} form.`)}`;
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
      const res = await fetch(formEndpoint, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ...values, source }) });
      state = res.ok ? 'sent' : 'error';
    } catch {
      state = 'error';
    }
  }
</script>

<form class="av-form av-card" onsubmit={submit} aria-labelledby={`${source}-form-title`}>
  <h2 id={`${source}-form-title`} class="av-h3">{f.title}</h2>
  {#each f.fields as x}
    <div class="av-field">
      <label for={`${source}-${x.name}`}>{x.label}{#if x.required}<span aria-hidden="true"> *</span>{/if}</label>
      {#if x.type === 'select'}
        <select id={`${source}-${x.name}`} name={x.name} bind:value={values[x.name]}>{#each x.options as o}<option>{o}</option>{/each}</select>
      {:else if x.type === 'textarea'}
        <textarea id={`${source}-${x.name}`} name={x.name} placeholder={x.placeholder} bind:value={values[x.name]}></textarea>
      {:else}
        <input id={`${source}-${x.name}`} name={x.name} type={x.type} placeholder={x.placeholder} required={x.required} autocomplete={x.autocomplete} bind:value={values[x.name]} />
      {/if}
    </div>
  {/each}
  <button class="av-btn av-btn-primary av-btn-lg" type="submit" disabled={state === 'sending'}>{state === 'sending' ? 'Sending' : f.submit} <span class="av-arrow">→</span></button>
  {#if state === 'sent'}<p class="av-body" role="status" style="color:var(--green)">{f.thanks}</p>{/if}
  {#if state === 'error'}<p class="av-body" role="alert" style="color:var(--red)">That did not go through. Email {to} directly and we will take it from there.</p>{/if}
  <p class="av-small">{formEndpoint ? 'Your details go to the founding team and nowhere else.' : f.fallbackNote}</p>
</form>
