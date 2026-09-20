#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// =============================================================================
// cpu.mjs — how much CPU does a page burn while nobody touches it?
// -----------------------------------------------------------------------------
// Lighthouse scores how fast a page ARRIVES. It says nothing about what the page
// costs once it is there, and a page can score 100 and still keep a laptop's fan
// running: this site did, for a day. A gradient ring animated through a custom
// property repainted and re-blurred itself sixty times a second, 38% of a core
// on an idle front page. Nothing in CI noticed, a person's fan did.
//
// This opens a page in Chromium with the GPU on, lets it settle, and reads the
// operating system's CPU time for every browser process over a fixed window:
// the renderer, and the GPU and raster work the page's own metrics leave out.
// That is the number Chrome's task manager shows. Then it repeats with one
// suspect switched off at a time, so the difference is that suspect's cost.
//
//   bun x --bun vite dev                          (or preview, in another shell)
//   bun run perf:cpu -- http://127.0.0.1:5173 /
//   bun run perf:cpu -- http://127.0.0.1:5173 / "#tour"      scroll there first
//
// Budget: an idle page should sit under 5% of one core. Anything that animates
// forever must use only `transform` and `opacity`, and must stop when it is off
// screen ($lib/reveal.js sets `is-live` for that). See FACELIFT.md.
// =============================================================================

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';

const [origin, path = '/', target] = process.argv.slice(2);
if (!origin) {
  console.error('usage: node scripts/perf/cpu.mjs <origin> [path] [selector to scroll to]');
  process.exit(2);
}
const WINDOW_MS = 10_000;
const TOKEN = 'avarokprobe' + Date.now();

/** CPU seconds used so far by this browser's processes, keyed by process type. */
function cpuByType() {
  let rows = [];
  if (process.platform === 'win32') {
    const ps = `Get-CimInstance Win32_Process -Filter "Name LIKE 'chrome%' OR Name LIKE 'headless%'" | ForEach-Object { "$($_.ProcessId)|$($_.ParentProcessId)|$([double]($_.KernelModeTime + $_.UserModeTime) / 1e7)|$($_.CommandLine)" }`;
    const out = execFileSync('powershell.exe', ['-NoProfile', '-Command', ps], { encoding: 'utf8', maxBuffer: 1 << 24 });
    rows = out.split(/\r?\n/).filter(Boolean).map((l) => {
      const [pid, ppid, cpu, ...cmd] = l.split('|');
      return { pid: +pid, ppid: +ppid, cpu: +cpu, cmd: cmd.join('|') };
    });
  } else {
    const out = execFileSync('ps', ['-eo', 'pid=,ppid=,cputime=,args='], { encoding: 'utf8', maxBuffer: 1 << 24 });
    rows = out.split('\n').filter(Boolean).map((l) => {
      const m = l.trim().match(/^(\d+)\s+(\d+)\s+([\d:.-]+)\s+(.*)$/);
      if (!m) return null;
      const parts = m[3].replace(/^(\d+)-/, (_, d) => `${+d * 24}:`).split(':').map(Number);
      const cpu = parts.reduce((s, v) => s * 60 + v, 0);
      return { pid: +m[1], ppid: +m[2], cpu, cmd: m[4] };
    }).filter(Boolean);
  }
  const root = rows.find((r) => r.cmd.includes(TOKEN) && !/--type=/.test(r.cmd));
  if (!root) return {};
  const mine = new Set([root.pid]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const r of rows) if (mine.has(r.ppid) && !mine.has(r.pid)) { mine.add(r.pid); grew = true; }
  }
  const acc = {};
  for (const r of rows) if (mine.has(r.pid)) { const type = (r.cmd.match(/--type=([a-z-]+)/) || [, 'browser'])[1]; acc[type] = (acc[type] || 0) + r.cpu; }
  return acc;
}

const STILL = '*, *::before, *::after { animation-play-state: paused !important; }';
const variants = [
  ['as shipped', '', false],
  ['css animations paused', STILL, false],
  ['videos paused', '', true],
  ['both', STILL, true]
];

const results = [];
for (const [name, css, pause] of variants) {
  const browser = await chromium.launch({ headless: true, args: [`--${TOKEN}`, '--headless=new', '--enable-gpu', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu-rasterization'] });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, serviceWorkers: 'block' });
  const page = await ctx.newPage();
  await page.goto(origin + path, { waitUntil: 'load' });
  await page.waitForTimeout(2500);
  if (target) {
    await page.locator(target).first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(1500);
  }
  if (css) await page.addStyleTag({ content: css });
  if (pause) await page.evaluate(() => { document.querySelectorAll('video').forEach((v) => v.pause()); HTMLMediaElement.prototype.play = () => Promise.resolve(); });
  await page.waitForTimeout(2500);
  const cdp = await ctx.newCDPSession(page);
  await cdp.send('Performance.enable');
  const metrics = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map((m) => [m.name, m.value]));
  const m0 = await metrics();
  const c0 = cpuByType();
  const t0 = Date.now();
  await page.waitForTimeout(WINDOW_MS);
  const dt = (Date.now() - t0) / 1000;
  const c1 = cpuByType();
  const m1 = await metrics();
  const pct = (k) => (((c1[k] || 0) - (c0[k] || 0)) / dt) * 100;
  results.push({
    variant: name,
    'renderer %': +pct('renderer').toFixed(1),
    'gpu %': +pct('gpu-process').toFixed(1),
    'all processes %': +Object.keys(c1).reduce((s, k) => s + pct(k), 0).toFixed(1),
    'main thread busy %': +(((m1.TaskDuration - m0.TaskDuration) / dt) * 100).toFixed(1),
    'style and layout ms/s': +(((m1.RecalcStyleDuration - m0.RecalcStyleDuration + m1.LayoutDuration - m0.LayoutDuration) / dt) * 1000).toFixed(1)
  });
  await browser.close();
}
console.log(`${path}${target ? ' at ' + target : ''}: percent of one core, page idle, ${WINDOW_MS / 1000} s window`);
console.table(results);
const shipped = results[0]['all processes %'];
console.log(shipped <= 5 ? `ok: ${shipped}% idle, within the 5% budget` : `OVER BUDGET: ${shipped}% idle. The variants above show which suspect owns it.`);
process.exit(shipped <= 5 ? 0 : 1);
