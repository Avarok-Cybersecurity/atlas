#!/usr/bin/env node
// =============================================================================
// gen-models.mjs — generate src/lib/models.generated.json from the recipe SSOT
// -----------------------------------------------------------------------------
// SSOT: https://github.com/Avarok-Cybersecurity/atlas-recipes
//   (read-only mirror expected at /workspace/atlas-recipes/recipes on the host
//    that runs this script — that public repo is the single source of truth for
//    every supported model + its canonical `sparkrun run` command).
//
// Regenerate with:   node site/scripts/gen-models.mjs
//
// Output is a 3-level tree consumed by the model navigation UI:
//   [{ vendor, icon, subfamilies: [{ name, recipes: [{...}] }] }]
//   level 1: vendor  = top-level brand (Qwen/Gemma/Nemotron/Mistral/MiniMax)
//   level 2: subfamily = the recipe directory (e.g. qwen3.6, gemma4)
//   level 3: recipe  = one recipes/**/*.yaml file
//
// Every recipes/**/*.yaml MUST appear in the output. The generated tree's
// total recipe count is asserted to equal the number of recipe YAML files.
// No third-party deps: a tiny hand-rolled reader parses the (deliberately
// simple) recipe schema — top-level scalars, a `metadata:` block of scalars
// plus a `description: |` literal block, and a `defaults:` scalar block.
// =============================================================================

import { readdirSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const RECIPES_ROOT = process.env.ATLAS_RECIPES_ROOT || '/workspace/atlas-recipes/recipes';
const SSOT_URL = 'https://github.com/Avarok-Cybersecurity/atlas-recipes';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '..', 'src', 'lib', 'models.generated.json');

// --- recursive YAML file discovery ------------------------------------------
function walkYaml(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walkYaml(full));
    else if (entry.endsWith('.yaml') || entry.endsWith('.yml')) out.push(full);
  }
  return out;
}

// --- minimal recipe reader ---------------------------------------------------
// Returns: { top: {scalars...}, metadata: {scalars + description}, defaults: {} }
function parseRecipe(text) {
  const lines = text.split('\n');
  const top = {};
  const metadata = {};
  const defaults = {};
  let section = 'top'; // 'top' | 'metadata' | 'defaults'
  let i = 0;

  const stripComment = (v) => {
    // strip an unquoted trailing comment, keep quoted/literal values intact
    if (v.startsWith('"') || v.startsWith("'")) return v;
    const h = v.indexOf(' #');
    return (h === -1 ? v : v.slice(0, h)).trim();
  };
  const unquote = (v) => {
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'")))
      return v.slice(1, -1);
    return v;
  };

  while (i < lines.length) {
    const raw = lines[i];
    const line = raw.replace(/\s+$/, '');
    i++;
    if (line.trim() === '' || line.trim().startsWith('#')) continue;

    // section headers (no indentation, key with empty value)
    if (/^metadata:\s*$/.test(line)) { section = 'metadata'; continue; }
    if (/^defaults:\s*$/.test(line)) { section = 'defaults'; continue; }
    if (/^[a-zA-Z_]/.test(line) && section !== 'top' && /^[a-zA-Z_][\w.-]*:\s*\S/.test(line)) {
      // a new top-level scalar after a block ends a block section
      section = 'top';
    }

    const m = line.match(/^(\s*)([\w.\-]+):\s*(.*)$/);
    if (!m) continue;
    const [, indent, key, rest0] = m;
    const rest = rest0.trim();

    const bucket = section === 'metadata' ? metadata : section === 'defaults' ? defaults : top;

    if (rest === '|' || rest === '|-' || rest === '>' || rest === '>-') {
      // literal/folded block scalar — collect more-indented lines
      const baseIndent = indent.length;
      const block = [];
      while (i < lines.length) {
        const bl = lines[i];
        if (bl.trim() === '') { block.push(''); i++; continue; }
        const blIndent = bl.match(/^(\s*)/)[1].length;
        if (blIndent <= baseIndent) break;
        block.push(bl.slice(baseIndent + 2));
        i++;
      }
      bucket[key] = block.join('\n').replace(/\n+$/, '').trim();
      continue;
    }

    if (rest === '') continue; // nested map header we don't need
    bucket[key] = unquote(stripComment(rest));
  }

  return { top, metadata, defaults };
}

// --- subfamily display names -------------------------------------------------
// Keyed by recipe directory name (the SSOT family). This is the 2nd nav level.
const FAMILY_DISPLAY = {
  'qwen3.5': 'Qwen3.5',
  'qwen3.6': 'Qwen3.6',
  'qwen3-next': 'Qwen3-Next',
  'qwen3-coder-next': 'Qwen3-Coder-Next',
  'qwen3-vl': 'Qwen3-VL',
  'gemma4': 'Gemma-4',
  'nemotron-3-nano': 'Nemotron-3 Nano',
  'nemotron-3-super': 'Nemotron-3 Super',
  'mistral-small-4': 'Mistral-Small-4',
  'minimax-m2.7': 'MiniMax-M2.7',
  'deepseek-v4': 'DeepSeek-V4',
  'diffusion-gemma': 'Gemma Diffusion'
};
function familyDisplay(fam) {
  if (FAMILY_DISPLAY[fam]) return FAMILY_DISPLAY[fam];
  return fam.replace(/[-.]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// --- vendor (top-level brand) mapping ----------------------------------------
// The 1st nav level. Every recipe directory MUST map to exactly one vendor;
// an unmapped directory is a hard error (PCND — no silent default bucket).
// `icon` is a stable key the Svelte component resolves to an inline SVG;
// the SVG markup itself is NOT emitted into JSON (kept inline in the UI).
const VENDOR_OF_FAMILY = {
  'qwen3.5': 'Qwen',
  'qwen3.6': 'Qwen',
  'qwen3-next': 'Qwen',
  'qwen3-coder-next': 'Qwen',
  'qwen3-vl': 'Qwen',
  'gemma4': 'Gemma',
  'nemotron-3-nano': 'Nemotron',
  'nemotron-3-super': 'Nemotron',
  'mistral-small-4': 'Mistral',
  'minimax-m2.7': 'MiniMax',
  'deepseek-v4': 'DeepSeek',
  'diffusion-gemma': 'Gemma'
};
// Display + icon key + stable sort order, keyed by vendor brand.
const VENDOR_META = {
  Qwen: { icon: 'qwen', order: 0 },
  Gemma: { icon: 'gemma', order: 1 },
  Nemotron: { icon: 'nemotron', order: 2 },
  Mistral: { icon: 'mistral', order: 3 },
  MiniMax: { icon: 'minimax', order: 4 },
  DeepSeek: { icon: 'deepseek', order: 5 }
};
function vendorOf(fam) {
  const v = VENDOR_OF_FAMILY[fam];
  if (!v) {
    console.error(
      `Unmapped recipe family "${fam}" — add it to VENDOR_OF_FAMILY. SSOT: ${SSOT_URL}`
    );
    process.exit(1);
  }
  return v;
}

// --- topology inference ------------------------------------------------------
// The recipe's *own* topology is encoded in (a) the filename stem suffix
// (`-ep2` / `-tp2`) and (b) the declared node count. We deliberately do NOT
// scan the prose description: several single-node recipes mention "Use --tp 2
// / EP=2 ..." as advisory text, which would false-positive.
function inferTopology(stem, top) {
  const s = stem.toLowerCase();
  if (/(^|-)ep2($|-)/.test(s)) return 'EP=2';
  if (/(^|-)tp2($|-)/.test(s)) return 'TP=2';
  const maxN = parseInt(top.max_nodes ?? '1', 10);
  const minN = parseInt(top.min_nodes ?? top.max_nodes ?? '1', 10);
  if (maxN >= 2 || minN >= 2) return 'EP=2';
  return 'single';
}

// --- per-recipe display label ------------------------------------------------
function recipeDisplay(stem) {
  // humanize the file stem into a short variant label
  const parts = stem.replace(/-atlas$/, '').split('-');
  const out = parts.map((p) => {
    const lp = p.toLowerCase();
    if (lp === 'nvfp4a16' || lp === 'nvfp4') return 'NVFP4';
    if (lp === 'fp8') return 'FP8';
    if (lp === 'bf16') return 'BF16';
    if (lp === 'ep2') return 'EP=2';
    if (lp === 'tp2') return 'TP=2';
    if (lp === 'mtp') return 'MTP';
    if (lp === 'vl') return 'VL';
    if (lp === 'it') return 'IT';
    if (lp === 'dense' || lp === 'single') return p[0].toUpperCase() + p.slice(1);
    // param-style tokens: 80b, a3b, a10b, a12b, 0.8b, 122b -> uppercase
    if (/^a?\d+(\.\d+)?b$/.test(lp)) return p.toUpperCase();
    // version-bearing family tokens stay as-is (qwen3.5, gemma, minimax...)
    return p[0].toUpperCase() + p.slice(1);
  });
  return out.join(' ');
}

// --- main --------------------------------------------------------------------
const files = walkYaml(RECIPES_ROOT).sort();
if (files.length === 0) {
  console.error(`No recipe YAML files found under ${RECIPES_ROOT}`);
  process.exit(1);
}

// Build a 3-level tree: vendor -> subfamily (recipe dir) -> recipes.
const vendorMap = new Map(); // vendor -> { subfamilies: Map<famKey, {name,recipes[]}> }
let recipeCount = 0;

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  const { top, metadata } = parseRecipe(text);
  const fam = basename(dirname(file)); // recipe directory == subfamily key
  const stem = basename(file).replace(/\.(ya?ml)$/, '');
  const topology = inferTopology(stem, top);
  const vendor = vendorOf(fam);

  // sparkrun requires --hosts (it errors "No hosts specified" otherwise).
  // Single-node recipes target one Spark -> localhost. EP=2/TP=2 recipes
  // span two nodes, so show a two-host placeholder the user fills in.
  const hostsArg =
    topology === 'single' ? '--hosts localhost' : '--hosts <spark-1>,<spark-2>';
  const recipe = {
    displayName: recipeDisplay(stem),
    hfId: top.model || '',
    params: metadata.model_params || '',
    quant: metadata.quantization || '',
    topology,
    recipeStem: stem,
    command: `sparkrun run @atlas/${stem} ${hostsArg}`
  };

  if (!vendorMap.has(vendor)) vendorMap.set(vendor, new Map());
  const subs = vendorMap.get(vendor);
  if (!subs.has(fam)) subs.set(fam, { name: familyDisplay(fam), recipes: [] });
  subs.get(fam).recipes.push(recipe);
  recipeCount++;
}

// Stable ordering: vendors by VENDOR_META.order, subfamilies by their dir key,
// recipes by stem. This keeps the JSON (and the rendered nav) deterministic.
const vendors = [...vendorMap.entries()]
  .map(([vendor, subs]) => {
    const subfamilies = [...subs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, sf]) => {
        sf.recipes.sort((a, b) => a.recipeStem.localeCompare(b.recipeStem));
        return sf;
      });
    return { vendor, icon: VENDOR_META[vendor].icon, subfamilies };
  })
  .sort((a, b) => VENDOR_META[a.vendor].order - VENDOR_META[b.vendor].order);

const json = JSON.stringify(vendors, null, 2) + '\n';
writeFileSync(OUT, json);

const emitted = vendors.reduce(
  (n, v) => n + v.subfamilies.reduce((m, s) => m + s.recipes.length, 0),
  0
);
if (emitted !== recipeCount || emitted !== files.length) {
  console.error(
    `Recipe count mismatch: yaml files=${files.length}, emitted=${emitted}. SSOT: ${SSOT_URL}`
  );
  process.exit(1);
}

const subCount = vendors.reduce((n, v) => n + v.subfamilies.length, 0);
console.log(
  `Wrote ${OUT}\n  ${files.length} recipes across ${subCount} subfamilies` +
    ` / ${vendors.length} vendors (SSOT: ${SSOT_URL})`
);
for (const v of vendors) {
  const n = v.subfamilies.reduce((m, s) => m + s.recipes.length, 0);
  console.log(`  - ${v.vendor} (${n}):`);
  for (const s of v.subfamilies) console.log(`      · ${s.name}: ${s.recipes.length}`);
}
