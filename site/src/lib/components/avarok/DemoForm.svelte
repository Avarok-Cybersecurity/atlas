<!--
  The site's one form. The demo page renders it with its defaults, and the
  waitlist and careers pages pass their own definition, so all three behave the
  same way.

  Where a request goes is decided by `formEndpoint` in content/brand.js:

    set      POST the fields there as JSON, tagged with `source` so one endpoint
             can tell a demo request from a waitlist entry. The endpoint is the
             Worker in deploy/cloudflare/forms-worker (its README has the setup).
             If the post fails, the visitor is told, and is handed the drafted
             email below so the request is still not lost.
    empty    compose a prefilled email to `to` in the visitor's own mail app, so
             nothing is stored on the site. A page cannot tell whether a mail app
             opened, so the drafted message is also shown, ready to paste.

  `website` is a field no person can see or reach. Whatever fills it in is a bot,
  and the endpoint drops the request while answering as if it had worked.

  To add a form: write a `{ title, fields, submit, thanks, composed, fallbackNote }`
  object next to demoPage.form in content/company.js, pass it as `form`, and add
  its `source` to SOURCES in the Worker.
-->
<script>
  import { dev } from '$app/environment';
  import { demoPage } from '$lib/content/company.js';
  import { contacts, formEndpoint } from '$lib/content/brand.js';

  let {
    form: f = demoPage.form,
    source = 'demo',
    to = contacts.sales,
    // the subject of the composed email, given the values typed so far
    subject = (v) => `Avarok working session, ${v.company || v.name || ''}`
  } = $props();

  // `bun x --bun vite dev` with VITE_FORM_ENDPOINT set posts to a Worker running
  // on this machine (`wrangler dev`). Only in dev: a real build always uses the
  // address in brand.js, which is the one the site guide tracks.
  const endpoint = (dev && import.meta.env.VITE_FORM_ENDPOINT) || formEndpoint;

  let values = $state(Object.fromEntries(f.fields.map((x) => [x.name, x.type === 'select' ? x.options[0] : ''])));
  let website = $state('');
  let state = $state('idle'); // idle | sending | sent | error

  // The request as plain text. It is the body of the drafted email, and it is what
  // the visitor is handed to paste when nothing else could deliver it.
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
    if (!endpoint) {
      window.location.href = mailto();
      state = 'sent';
      return;
    }
    state = 'sending';
    try {
      const res = await fetch(endpoint, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ...values, source, page: location.pathname, website }) });
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
  <div class="av-form-trap" aria-hidden="true">
    <label for={`${source}-website`}>Leave this empty</label>
    <input id={`${source}-website`} name="website" type="text" tabindex="-1" autocomplete="off" bind:value={website} />
  </div>
  <button class="av-btn av-btn-primary av-btn-lg" type="submit" disabled={state === 'sending'}>{state === 'sending' ? 'Sending' : f.submit} <span class="av-arrow">→</span></button>
  {#if state === 'sent'}
    <p class="av-body" role="status" style="color:var(--green)">{endpoint ? f.thanks : (f.composed ?? f.thanks)}</p>
  {/if}
  {#if state === 'error'}
    <p class="av-body" role="alert" style="color:var(--red)">That did not go through. Your request is below, ready to send to {to} yourself.</p>
  {/if}
  {#if (state === 'sent' && !endpoint) || state === 'error'}
    <div class="av-form-fallback">
      <p class="av-small"><strong>{state === 'error' ? 'Send it by email.' : 'No email opened?'}</strong> Copy the request below and send it to <a class="av-link" href={mailto()}>{to}</a>.</p>
      <textarea readonly rows="5" aria-label="Your request, ready to paste into an email">{message()}</textarea>
      <button type="button" class="av-btn av-btn-secondary av-btn-sm" onclick={copyMessage}>{copied ? 'Copied' : 'Copy the request'}</button>
    </div>
  {/if}
  <p class="av-small">{endpoint ? 'Your details go to the Avarok team.' : f.fallbackNote}</p>
</form>

<style>
  .av-form-fallback { display: grid; gap: 0.6rem; padding: 0.9rem; border: 1px dashed var(--border-strong); border-radius: var(--av-radius-sm); justify-items: start; }
  .av-form-fallback textarea { width: 100%; font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.5; color: var(--t2); background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 0.7rem; resize: vertical; }
  /* Off screen, not display:none: some bots skip fields that are not rendered. */
  .av-form-trap { position: absolute; left: -9999px; top: auto; width: 1px; height: 1px; overflow: hidden; }
</style>
