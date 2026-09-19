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

  // The message the form composes, as plain text. It goes into the email, and it
  // is also shown to the visitor afterwards: a page cannot tell whether a mail
  // app opened, and on a machine without one the request would otherwise vanish
  // while the form said thank you.
  const message = () => f.fields.map((x) => `${x.label}: ${values[x.name] || ''}`).join('\n') + `\n\nSent from the Avarok website ${source} form.`;
  const mailto = () => `mailto:${to}?subject=${encodeURIComponent(subject(values))}&body=${encodeURIComponent(message())}`;
  let copied = $state(false);
  async function copyMessage() {
    try {
      await navigator.clipboard.writeText(`To: ${to}\nSubject: ${subject(values)}\n\n${message()}`);
      copied = true;
    } catch {
      copied = false;
    }
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
  {#if state === 'sent'}
    <p class="av-body" role="status" style="color:var(--green)">{formEndpoint ? f.thanks : (f.composed ?? f.thanks)}</p>
    {#if !formEndpoint}
      <div class="av-form-fallback">
        <p class="av-small"><strong>No email opened?</strong> Copy the request below and send it to <a class="av-link" href={`mailto:${to}`}>{to}</a>.</p>
        <textarea readonly rows="5" aria-label="Your request, ready to paste into an email">{message()}</textarea>
        <button type="button" class="av-btn av-btn-secondary av-btn-sm" onclick={copyMessage}>{copied ? 'Copied' : 'Copy the request'}</button>
      </div>
    {/if}
  {/if}
  {#if state === 'error'}<p class="av-body" role="alert" style="color:var(--red)">That did not go through. Email {to} directly and we will take it from there.</p>{/if}
  <p class="av-small">{formEndpoint ? 'Your details go to the founding team and nowhere else.' : f.fallbackNote}</p>
</form>

<style>
  .av-form-fallback { display: grid; gap: 0.6rem; padding: 0.9rem; border: 1px dashed var(--border-strong); border-radius: var(--av-radius-sm); justify-items: start; }
  .av-form-fallback textarea { width: 100%; font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.5; color: var(--t2); background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 0.7rem; resize: vertical; }
</style>
