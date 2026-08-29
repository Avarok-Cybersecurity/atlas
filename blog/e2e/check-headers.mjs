#!/usr/bin/env bun
/**
 * Post-deploy check for blog.atlasinference.io.
 *
 *   bun blog/e2e/check-headers.mjs [base-url]
 *
 * Asserts the four response headers the vhost promises, on all three response
 * classes that nginx routes differently: a document, a content-hashed asset,
 * and a 404. It exists because the defect it guards against is invisible from
 * inside the config file — `add_header` does not accumulate across contexts, so
 * one location block setting Cache-Control silently discards every inherited
 * security header, and nginx reports nothing. See
 * blog/deploy/nginx/blog.atlasinference.io.conf.
 *
 * This is a live check, deliberately: the thing being tested is the deployed
 * server's behaviour, and nothing short of a request observes it.
 *
 * To watch it FAIL — which you should, before trusting it — point it at a host
 * that still has the location-level form:
 *
 *   bun blog/e2e/check-headers.mjs https://docs.atlasinference.io
 */

const base = (process.argv[2] ?? 'https://blog.atlasinference.io').replace(/\/$/, '');

const SECURITY = {
  'x-content-type-options': 'nosniff',
  'x-frame-options': 'SAMEORIGIN',
  'referrer-policy': 'strict-origin-when-cross-origin'
};

let failures = 0;
const log = [];

function check(what, ok, detail) {
  log.push(`${ok ? '  ok  ' : '  FAIL'}  ${what}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
}

async function get(path) {
  const res = await fetch(base + path, { redirect: 'manual' });
  return { res, h: (n) => res.headers.get(n) ?? '' };
}

function expectSecurityHeaders(label, h) {
  for (const [name, value] of Object.entries(SECURITY)) {
    check(`${label}: ${name}`, h(name).toLowerCase() === value.toLowerCase(), h(name) || 'absent');
  }
}

console.log(`checking ${base}`);

/* 1. A document. This is the response class the add_header defect hits, and the
      one where these headers actually do something. */
{
  const { res, h } = await get('/');
  check('document: 200', res.status === 200, `status ${res.status}`);
  check('document: cache-control is short', /max-age=300/.test(h('cache-control')), h('cache-control') || 'absent');
  check('document: not cached forever', !/immutable/.test(h('cache-control')), h('cache-control'));
  expectSecurityHeaders('document', h);
}

/* 2. A content-hashed asset, discovered from the document rather than
      hardcoded — the hash changes on every build. */
{
  const html = await (await fetch(base + '/')).text();
  const m = html.match(/\/_app\/immutable\/[^"'\s>]+\.(?:js|css)/);
  if (!m) {
    check('asset: found a hashed asset to test', false, 'no /_app/immutable/ URL in the document');
  } else {
    const { res, h } = await get(m[0]);
    check(`asset ${m[0]}: 200`, res.status === 200, `status ${res.status}`);
    check('asset: immutable', /immutable/.test(h('cache-control')), h('cache-control') || 'absent');
    check('asset: year-long max-age', /max-age=31536000/.test(h('cache-control')), h('cache-control'));
    expectSecurityHeaders('asset', h);
  }
}

/* 3. A 404. `error_page 404 /404.html` serves a real document here, so this
      also proves the built 404 page landed at the path nginx expects. */
{
  const { res, h } = await get('/this-path-does-not-exist-' + 'x'.repeat(8));
  check('missing page: 404', res.status === 404, `status ${res.status}`);
  const body = await res.text();
  check('missing page: serves the built 404 document', /Not found/i.test(body), `${body.length} bytes`);
  expectSecurityHeaders('missing page', h);
}

/* 4. Hidden files stay hidden. */
{
  const { res } = await get('/.env');
  check('dotfile: refused', res.status === 404 || res.status === 403, `status ${res.status}`);
}

console.log(log.join('\n'));
if (failures) {
  console.error(`\n${failures} check(s) failed against ${base}`);
  process.exit(1);
}
console.log(`\nall ${log.length} checks passed against ${base}`);
