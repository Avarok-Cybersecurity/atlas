// SPDX-License-Identifier: AGPL-3.0-only
//
// The Worker the site's forms post to lives in deploy/cloudflare/forms-worker and
// is deployed by hand, not by CI. It is tested here because this is the suite CI
// runs, and because a form that loses a request is the one failure nobody sees:
// the visitor is thanked and the company never hears.
import { afterEach, beforeEach, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import worker, { clean, corsFor } from '../../deploy/cloudflare/forms-worker/src/index.js';
import { contacts, formEndpoint } from './content/brand.js';

const WORKER_DIR = join(import.meta.dir, '..', '..', 'deploy', 'cloudflare', 'forms-worker');
const ORIGIN = 'https://atlascybernetics.ai';
const baseEnv = () => ({ ALLOWED_ORIGINS: `${ORIGIN},http://localhost:5173`, ALLOWED_ORIGIN_SUFFIXES: '.atlas-site.pages.dev', TO_DEMO: 'sales@example.test', TO_WAITLIST: 'sales@example.test', TO_CAREERS: 'eng@example.test', MAIL_FROM: 'Forms <forms@example.test>' });
const kv = () => {
  const store = new Map();
  return { store, get: async (k) => store.get(k) ?? null, put: async (k, v) => void store.set(k, v) };
};
const post = (body, { origin = ORIGIN, type = 'application/json', method = 'POST' } = {}) =>
  new Request('https://forms.example.test/lead', { method, headers: { ...(origin ? { origin } : {}), 'content-type': type, 'cf-connecting-ip': '203.0.113.7' }, body: method === 'POST' ? (typeof body === 'string' ? body : JSON.stringify(body)) : undefined });
const demo = { source: 'demo', name: 'Ada Buyer', email: 'ada@buyer.test', company: 'Buyer Corp', hardware: '64 H100', notes: 'Show us C=128.' };

let outbound;
const realFetch = globalThis.fetch;
beforeEach(() => {
  outbound = [];
  globalThis.fetch = async (url, init) => {
    outbound.push({ url: String(url), body: JSON.parse(init.body), headers: init.headers });
    return new Response('{}', { status: 200 });
  };
});
afterEach(() => {
  globalThis.fetch = realFetch;
});

test('a request is kept first, then every configured channel is told', async () => {
  const env = { ...baseEnv(), LEADS: kv(), DISCORD_WEBHOOK_URL: 'https://discord.test/hook', SLACK_WEBHOOK_URL: 'https://slack.test/hook', RESEND_API_KEY: 're_test' };
  const res = await worker.fetch(post(demo), env);
  expect(res.status).toBe(200);
  const { ok, id } = await res.json();
  expect(ok).toBe(true);
  expect(id).toMatch(/^demo-/);
  const stored = [...env.LEADS.store].filter(([k]) => k.startsWith('lead:'));
  expect(stored).toHaveLength(1);
  expect(JSON.parse(stored[0][1])).toMatchObject({ source: 'demo', email: 'ada@buyer.test', company: 'Buyer Corp' });
  expect(outbound.map((o) => new URL(o.url).host).sort()).toEqual(['api.resend.com', 'discord.test', 'slack.test']);
  const mail = outbound.find((o) => o.url.includes('resend'));
  expect(mail.body).toMatchObject({ to: ['sales@example.test'], reply_to: 'ada@buyer.test', from: 'Forms <forms@example.test>' });
});

test('each form reaches the inbox it belongs to', async () => {
  const env = { ...baseEnv(), RESEND_API_KEY: 're_test' };
  await worker.fetch(post({ source: 'careers', email: 'dev@hire.test', role: 'Kernel Engineer' }), env);
  await worker.fetch(post({ source: 'waitlist', email: 'dev@wait.test' }), env);
  expect(outbound.map((o) => o.body.to[0])).toEqual(['eng@example.test', 'sales@example.test']);
});

test('a visitor cannot make the Discord message ping anyone', async () => {
  const env = { ...baseEnv(), DISCORD_WEBHOOK_URL: 'https://discord.test/hook' };
  await worker.fetch(post({ ...demo, notes: '@everyone free money <@&123>' }), env);
  expect(outbound[0].body.allowed_mentions).toEqual({ parse: [] });
});

test('it is a success if the request was kept, even when every channel is down', async () => {
  globalThis.fetch = async () => new Response('no', { status: 500 });
  const env = { ...baseEnv(), LEADS: kv(), DISCORD_WEBHOOK_URL: 'https://discord.test/hook' };
  expect((await worker.fetch(post(demo), env)).status).toBe(200);
});

test('it is a failure the site can see when nothing kept it and nobody was told', async () => {
  globalThis.fetch = async () => {
    throw new Error('network down');
  };
  const down = await worker.fetch(post(demo), { ...baseEnv(), SLACK_WEBHOOK_URL: 'https://slack.test/hook' });
  expect(down.status).toBe(502);
  const unconfigured = await worker.fetch(post(demo), baseEnv());
  expect(unconfigured.status).toBe(502);
});

test('a hidden field filled in means a bot: it is thanked, and nothing is kept or sent', async () => {
  const env = { ...baseEnv(), LEADS: kv(), DISCORD_WEBHOOK_URL: 'https://discord.test/hook' };
  const res = await worker.fetch(post({ ...demo, website: 'https://spam.test' }), env);
  expect(res.status).toBe(200);
  expect([...env.LEADS.store.keys()].filter((k) => k.startsWith('lead:'))).toEqual([]);
  expect(outbound).toEqual([]);
});

test('only our own pages may post, and the answer names that origin and no other', async () => {
  const env = { ...baseEnv(), LEADS: kv() };
  expect((await worker.fetch(post(demo, { origin: 'https://evil.test' }), env)).status).toBe(403);
  expect((await worker.fetch(post(demo, { origin: '' }), env)).status).toBe(403);
  const preview = await worker.fetch(post(demo, { origin: 'https://facelift-preview.atlas-site.pages.dev' }), env);
  expect(preview.status).toBe(200);
  expect(preview.headers.get('access-control-allow-origin')).toBe('https://facelift-preview.atlas-site.pages.dev');
  expect(corsFor('http://atlascybernetics.ai', env)).toBeNull(); // plain http is not us
  expect(corsFor('https://atlas-site.pages.dev.evil.test', env)).toBeNull(); // a suffix must end the host
  const preflight = await worker.fetch(post(null, { method: 'OPTIONS' }), env);
  expect(preflight.status).toBe(204);
  expect(preflight.headers.get('access-control-allow-methods')).toContain('POST');
});

test('what it refuses, and with which status', async () => {
  const env = { ...baseEnv(), LEADS: kv() };
  expect((await worker.fetch(post({ ...demo, email: 'not an address' }), env)).status).toBe(400);
  expect((await worker.fetch(post({ ...demo, source: 'newsletter' }), env)).status).toBe(400);
  expect((await worker.fetch(post('{ not json', {}), env)).status).toBe(400);
  expect((await worker.fetch(post(demo, { type: 'text/plain' }), env)).status).toBe(415);
  expect((await worker.fetch(post({ ...demo, notes: 'x'.repeat(17000) }), env)).status).toBe(413);
  expect((await worker.fetch(post({ ...demo, name: 'x'.repeat(300) }), env)).status).toBe(400);
  expect((await worker.fetch(new Request('https://forms.example.test/elsewhere', { method: 'POST' }), env)).status).toBe(404);
  expect((await worker.fetch(new Request('https://forms.example.test/lead', { method: 'GET', headers: { origin: ORIGIN } }), env)).status).toBe(405);
});

test('a field it does not know is dropped, not stored', () => {
  const { lead } = clean({ ...demo, isAdmin: true, __proto__: { x: 1 }, ssn: '000' });
  expect(Object.keys(lead).sort()).toEqual(['company', 'email', 'hardware', 'name', 'notes', 'source']);
});

test('one address cannot post more than a few times a minute', async () => {
  const env = { ...baseEnv(), LEADS: kv() };
  const codes = [];
  for (let i = 0; i < 8; i++) codes.push((await worker.fetch(post({ ...demo, notes: `try ${i}` }), env)).status);
  expect(codes.slice(0, 6)).toEqual([200, 200, 200, 200, 200, 200]);
  expect(codes.slice(6)).toEqual([429, 429]);
});

test('the health check says it is alive and nothing else', async () => {
  const res = await worker.fetch(new Request('https://forms.example.test/'), baseEnv());
  expect(await res.json()).toEqual({ ok: true, service: 'avarok-forms' });
});

// The Worker's inboxes and the site's contact addresses are two copies of one fact.
test('the inboxes in wrangler.toml are the addresses the site publishes', () => {
  const toml = readFileSync(join(WORKER_DIR, 'wrangler.toml'), 'utf8');
  const v = (name) => toml.match(new RegExp(`^${name}\\s*=\\s*"([^"]*)"`, 'm'))?.[1];
  expect(v('TO_DEMO')).toBe(contacts.sales);
  expect(v('TO_WAITLIST')).toBe(contacts.sales);
  expect(v('TO_CAREERS')).toBe(contacts.technical);
  expect(toml).not.toMatch(/^\s*(DISCORD_WEBHOOK_URL|SLACK_WEBHOOK_URL|RESEND_API_KEY)\s*=/m); // secrets are never committed
});

// Empty means not switched on yet, and the forms draft an email instead.
test('the site names no endpoint, or this Worker\'s /lead over https', () => {
  expect(formEndpoint).toMatch(/^(|https:\/\/[^/\s]+\/lead)$/);
});
