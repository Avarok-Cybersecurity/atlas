# avarok-forms: where the site's forms go

The marketing site is static files, so it has nowhere to send a form. Until this Worker is
deployed and switched on, each form drafts an email in the visitor's own mail app, and a
request only arrives if the visitor then presses send. Nothing is recorded, and a visitor
with no mail app is handed the text to paste. That is honest, and it loses requests.

This Worker is the fix. The three forms (demo request, Community Edition waitlist, careers
interest) post JSON to `/lead`, and it:

1. **keeps** the request in a KV namespace, if one is bound, then
2. **tells** whoever is configured: a Discord webhook, a Slack webhook, an email through Resend.

It answers 200 if the request was kept or at least one channel took it. If neither happened
it answers 502, and the form on the site tells the visitor and hands them the drafted email,
so nobody is thanked for a request nobody will see.

One file, no dependency, no build step. `src/index.js` is what ships.

## Set it up

About ten minutes, by someone with access to the Cloudflare account. From this folder:

```sh
npx wrangler@4 login

# 1. Recommended: somewhere to keep every request. Paste the id it prints into
#    wrangler.toml and remove the comment marks on the [[kv_namespaces]] block.
npx wrangler@4 kv namespace create LEADS

# 2. At least one way to be told. Any or all of these. Each asks for the value.
npx wrangler@4 secret put DISCORD_WEBHOOK_URL    # a webhook of a PRIVATE channel
npx wrangler@4 secret put SLACK_WEBHOOK_URL      # an incoming webhook
npx wrangler@4 secret put RESEND_API_KEY         # email. See "Email" below first

# 3. Ship it. It prints the address, https://avarok-forms.<account>.workers.dev
npx wrangler@4 deploy
```

Then prove it, with the address it printed:

```sh
curl -i https://avarok-forms.<account>.workers.dev/                       # {"ok":true,...}
curl -i https://avarok-forms.<account>.workers.dev/lead \
  -H 'origin: https://atlascybernetics.ai' -H 'content-type: application/json' \
  -d '{"source":"demo","name":"Test","email":"you@atlascybernetics.ai","company":"Test","notes":"Wiring check."}'
```

A 200 with an `id` means it was kept or delivered. Look for it where you pointed it.

## Switch the site on

One line. In `site/src/lib/content/brand.js`:

```js
export const formEndpoint = 'https://avarok-forms.<account>.workers.dev/lead';
```

Then, from `site/`: `bun x --bun vite build`, `bun run guide -- --note "Forms post to the Worker"`,
commit, push. The guide tracks the endpoint as a fact, so the unit suite fails until it is
recorded. From that commit the forms post here and the email draft is only the fallback.

A custom address (`forms.atlascybernetics.ai`) is optional: add it under the Worker's
Settings, Domains and Routes, and use it in `formEndpoint` instead.

## Try it on your own machine first

Two terminals. No Cloudflare login is needed for this.

```sh
# here: the Worker, in Cloudflare's own runtime
npx wrangler@4 dev --port 8787 --var DISCORD_WEBHOOK_URL:https://your-test-webhook

# in site/: the site, posting to it. Only `vite dev` reads this variable.
VITE_FORM_ENDPOINT=http://127.0.0.1:8787/lead bun x --bun vite dev
```

`ALLOWED_ORIGINS` in `wrangler.toml` already allows `localhost:5173`.

## Reading what it kept

```sh
npx wrangler@4 kv key list --binding LEADS --prefix lead:          # newest last, keys sort by time
npx wrangler@4 kv key get "lead:2026-09-19T17:46:20.863Z:demo-..." --binding LEADS
```

A record is the fields the visitor typed, plus an id, the time and the visitor's country. The
visitor's IP address is not stored. Records expire after `LEAD_TTL_DAYS` (180). These are
people's names and addresses: keep the namespace private, and delete on request with
`kv key delete`.

## Email

Email is the one channel that needs a second account, because Workers cannot send mail by
themselves. It uses [Resend](https://resend.com): verify `atlascybernetics.ai` there (three DNS
records), create an API key, and set it as `RESEND_API_KEY`. Mail goes to the inbox for that
form (`TO_DEMO`, `TO_WAITLIST`, `TO_CAREERS` in `wrangler.toml`), from `MAIL_FROM`, with reply-to
set to the visitor, so answering is one click. A test in the site's suite keeps those inboxes
equal to `contacts` in `brand.js`. Any provider with an HTTPS API fits: `toEmail()` is ten lines.

Cloudflare's own Email Routing can send from a Worker for free, but only on a domain whose
mail Cloudflare handles. This domain's mail is on Google, so that route is closed unless the
forms get a subdomain of their own.

## What stops abuse

- Only the site's own origins may post (`ALLOWED_ORIGINS`, and Pages preview hosts by suffix).
  A browser enforces that. A script can forge it, so it is a filter, not a lock.
- A hidden field no person can reach. Whatever fills it in is answered 200 and dropped.
- Only known fields are read, each with a length limit, and the whole body is capped at 16 KB.
- With KV bound, one address may post six times a minute.
- A visitor's text cannot ping anyone in Discord: mentions are switched off in the message.

If spam still arrives, add [Turnstile](https://developers.cloudflare.com/turnstile/). It was
left out on purpose: it loads a script from Cloudflare on the page, and the site's Lighthouse
gate allows no third party request. Load it only when a visitor first touches a form.

## Options that were weighed

| option | for | against | verdict |
| --- | --- | --- | --- |
| **A standalone Worker** (this) | Stays in the company's own Cloudflare account. Free at this volume. Cannot break the site: it is deployed apart from it. Keeps a record as well as notifying. | One manual deploy. A second address, so it needs a CORS allow list. | Chosen. |
| A Pages Function at `/api/lead` | Same origin, so no CORS, and it would ship with the site. | The site is a Direct Upload project fed by a CI artifact. Functions would have to enter that deploy path, which cannot be tested without the production credentials, and in advanced mode a bad `_worker.js` stands in front of every page. | Not worth the risk to a working deploy. Revisit if the site moves to Pages' git integration. |
| A hosted form service (Formspree, Basin, Getform) | Five minutes. Paste a URL into `formEndpoint` and entries arrive by email with a dashboard. | Visitors' details sit with a third party, there is a monthly fee past the free tier, and it is one more vendor to offboard. | The right answer if nobody has ten minutes for this. `formEndpoint` takes such a URL as it is. |
| A CRM's form endpoint (HubSpot and the like) | Requests land in a sales pipeline from the first day. | Needs the CRM chosen and paid for, and their endpoints want their own field names. | Add it later as one more channel here, next to Discord and Slack, once a CRM exists. |
| Keep the email draft only | Nothing to run, nothing stored. | Loses every visitor who does not press send, and leaves no record that anyone asked. | Stays as the fallback when the post fails. |
