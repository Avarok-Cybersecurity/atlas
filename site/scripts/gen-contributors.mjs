#!/usr/bin/env node
// =============================================================================
// gen-contributors.mjs — src/lib/contributors.generated.json from GitHub
// -----------------------------------------------------------------------------
// The contributors page calls everyone out by name. The list comes from the
// GitHub contributors API at build time, like gen-stars.mjs, and degrades the
// same way: a network failure keeps the committed file rather than failing
// the build, because a stale list is better than no site.
//
// No avatars are stored or hotlinked: the page renders initials, so it makes
// no third party request and the Lighthouse third-party budget stays at zero.
// =============================================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, '..', 'src/lib/contributors.generated.json');
const REPO = 'Avarok-Cybersecurity/atlas';
const BOTS = /\[bot\]$|^dependabot|^github-actions/i;

async function fetchAll() {
  const headers = { 'User-Agent': 'avarok-site-gen', Accept: 'application/vnd.github+json' };
  if (process.env.GH_TOKEN) headers.Authorization = `Bearer ${process.env.GH_TOKEN}`;
  const all = [];
  for (let page = 1; page <= 5; page++) {
    const res = await fetch(`https://api.github.com/repos/${REPO}/contributors?per_page=100&page=${page}`, { headers });
    if (!res.ok) throw new Error(`GitHub ${res.status}`);
    const batch = await res.json();
    all.push(...batch);
    if (batch.length < 100) break;
  }
  return all
    .filter((c) => c.type === 'User' && !BOTS.test(c.login))
    .map((c) => ({ login: c.login, url: c.html_url, contributions: c.contributions }))
    .sort((a, b) => b.contributions - a.contributions);
}

try {
  const contributors = await fetchAll();
  if (contributors.length === 0) throw new Error('empty contributor list');
  const json = {
    generated_date: new Date().toISOString().slice(0, 10),
    repo: REPO,
    total_contributions: contributors.reduce((n, c) => n + c.contributions, 0),
    contributors
  };
  const prev = existsSync(out) ? JSON.parse(readFileSync(out, 'utf8')) : null;
  if (!prev || JSON.stringify(prev.contributors) !== JSON.stringify(contributors)) {
    writeFileSync(out, JSON.stringify(json, null, 2) + '\n');
    console.log(`gen-contributors: wrote ${out} (${contributors.length} contributors)`);
  } else {
    console.log(`gen-contributors: ${out} unchanged (${contributors.length} contributors)`);
  }
} catch (err) {
  if (existsSync(out)) {
    console.warn(`gen-contributors: ${err.message}; keeping the committed file`);
  } else {
    throw new Error(`gen-contributors: ${err.message} and no committed file to fall back on`);
  }
}
