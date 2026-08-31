// SPDX-License-Identifier: AGPL-3.0-only

// postmd.js — everything that can be decided about a markdown post by reading
// its source, with no bundler, no parser and no filesystem.
//
// It exists so the authoring rules are TESTABLE. A rule enforced only inside a
// Vite plugin can only be checked by building the whole site; a rule here is a
// function `bun test` can call with a two-line fixture, which is what makes it
// cheap to prove that each rule both fires and can be satisfied.
//
// Every rule here also fails the BUILD, not just a linter: `posts.js` calls
// `analysePost` on each post's raw source at module scope, and the prerender
// imports that module. A post that breaks a rule cannot be published, which is
// the only enforcement that actually holds.

/** Post body images. Raster formats are refused: svg/gif/webp only. */
export const IMAGE_RE = /^\/images\/posts\/[a-z0-9-]+\/[A-Za-z0-9._-]+\.(svg|gif|webp)$/;

/**
 * Social card override. PNG and WEBP only, and NOT svg or gif — every major
 * scraper (X, Slack, LinkedIn, iMessage) refuses to rasterise an SVG card and
 * renders nothing at all, which looks identical to having no card.
 */
export const OG_RE = /^\/images\/og\/[A-Za-z0-9._-]+\.(png|webp)$/;

export const SLUG_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
export const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
export const VIDEO_PROVIDERS = ['youtube', 'vimeo'];

const REQUIRED = ['title', 'dek', 'categories', 'date', 'keywords', 'og-image'];
const OPTIONAL = ['author', 'slug', 'updated', 'canonical', 'draft'];
const KNOWN = new Set([...REQUIRED, ...OPTIONAL]);

const LIST_KEYS = new Set(['categories', 'keywords']);
const BOOL_KEYS = new Set(['draft']);

/** Words per minute for the reading estimate. */
const WPM = 220;

const unquote = (v) => {
  const t = v.trim();
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
    return t.slice(1, -1);
  }
  return t;
};

/**
 * Split the frontmatter block from the body.
 * @param {string} src
 * @returns {{front: string|null, body: string}}
 */
export function splitFrontmatter(src) {
  const text = src.replace(/^﻿/, '');
  if (!text.startsWith('---')) return { front: null, body: text };
  const end = text.indexOf('\n---', 3);
  if (end === -1) return { front: null, body: text };
  const front = text.slice(text.indexOf('\n') + 1, end);
  const after = text.indexOf('\n', end + 1);
  return { front, body: after === -1 ? '' : text.slice(after + 1) };
}

/**
 * Parse the frontmatter dialect: `key: value` lines, flow arrays `[a, b]`,
 * booleans, optional quotes. Deliberately NOT general YAML.
 *
 * The dialect is a strict subset because TWO parsers read it — this one, and
 * mdsvex's own — and a construct they disagree about would mean the validator
 * checked something different from what was published. The subset is the part
 * on which they cannot disagree.
 *
 * @param {string} src full post source
 * @returns {{meta: Record<string, unknown>, errors: string[]}}
 */
export function parseFrontmatter(src) {
  const { front } = splitFrontmatter(src);
  const errors = [];
  if (front === null) return { meta: {}, errors: ['no frontmatter block: a post must open with `---`'] };

  const meta = {};
  for (const [i, raw] of front.split('\n').entries()) {
    const line = raw.trimEnd();
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    if (/^\s/.test(line)) {
      errors.push(`frontmatter line ${i + 1}: indented lines are not supported (keep every field on one line)`);
      continue;
    }
    const at = line.indexOf(':');
    if (at === -1) {
      errors.push(`frontmatter line ${i + 1}: expected \`key: value\`, got ${JSON.stringify(line)}`);
      continue;
    }
    const key = line.slice(0, at).trim();
    const rest = line.slice(at + 1).trim();
    if (key in meta) errors.push(`frontmatter: duplicate key \`${key}\``);

    if (LIST_KEYS.has(key)) {
      if (!(rest.startsWith('[') && rest.endsWith(']'))) {
        errors.push(`frontmatter: \`${key}\` must be a flow list, e.g. [engineering, design]`);
        continue;
      }
      meta[key] = rest
        .slice(1, -1)
        .split(',')
        .map((s) => unquote(s))
        .filter((s) => s !== '');
    } else if (BOOL_KEYS.has(key)) {
      if (rest !== 'true' && rest !== 'false') {
        errors.push(`frontmatter: \`${key}\` must be true or false, got ${JSON.stringify(rest)}`);
        continue;
      }
      meta[key] = rest === 'true';
    } else {
      meta[key] = unquote(rest);
    }
  }
  return { meta, errors };
}

/**
 * Body with fenced blocks and inline code spans blanked out.
 *
 * Used by the math and reading-time rules so that a `$` inside `echo $PATH`,
 * or a whole shell script full of them, is structurally invisible — which is
 * exactly how remark-math sees it, because code is tokenised before text.
 * Newlines are preserved so reported line numbers stay true.
 */
export function stripCode(body) {
  const blanked = body.replace(/^([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\2[^\n]*$/gm, (m) =>
    m.replace(/[^\n]/g, ' ')
  );
  return blanked.replace(/`+[^`\n]*`+/g, (m) => ' '.repeat(m.length));
}

/** Fenced blocks in the body, as `{lang, meta, code}`. */
export function fences(body) {
  const out = [];
  const re = /^[ \t]*(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*([^\n]*)\n([\s\S]*?)^[ \t]*\1[ \t]*$/gm;
  let m;
  while ((m = re.exec(body)) !== null) out.push({ lang: m[2] || '', meta: m[3].trim(), code: m[4] });
  return out;
}

function checkMeta(meta, { tags, authors }, errors) {
  for (const k of Object.keys(meta)) {
    if (!KNOWN.has(k)) {
      errors.push(`frontmatter: unknown field \`${k}\` (known: ${[...KNOWN].sort().join(', ')})`);
    }
  }
  for (const k of REQUIRED) {
    if (!(k in meta)) errors.push(`frontmatter: missing required field \`${k}\``);
  }
  if (typeof meta.title === 'string' && meta.title.trim() === '') errors.push('frontmatter: `title` is empty');
  if (typeof meta.dek === 'string' && meta.dek.trim() === '') errors.push('frontmatter: `dek` is empty');

  if (Array.isArray(meta.categories)) {
    if (meta.categories.length === 0) errors.push('frontmatter: `categories` needs at least one entry');
    for (const c of meta.categories) {
      if (!(c in tags)) {
        errors.push(`frontmatter: unknown category \`${c}\` (known: ${Object.keys(tags).join(', ')})`);
      }
    }
  }
  if (Array.isArray(meta.keywords) && meta.keywords.length === 0) {
    errors.push('frontmatter: `keywords` needs at least one entry');
  }
  for (const k of ['date', 'updated']) {
    const v = meta[k];
    if (v === undefined) continue;
    if (!DATE_RE.test(String(v)) || Number.isNaN(Date.parse(`${v}T00:00:00Z`))) {
      errors.push(`frontmatter: \`${k}\` must be a real YYYY-MM-DD date, got ${JSON.stringify(v)}`);
    }
  }
  if (meta['og-image'] !== undefined && meta['og-image'] !== '' && !OG_RE.test(String(meta['og-image']))) {
    errors.push(
      `frontmatter: \`og-image\` must be '' or /images/og/<name>.png|webp, got ${JSON.stringify(meta['og-image'])}`
    );
  }
  if (meta.author !== undefined && !(meta.author in authors)) {
    errors.push(`frontmatter: unknown author \`${meta.author}\` (known: ${Object.keys(authors).join(', ')})`);
  }
  if (meta.slug !== undefined && !SLUG_RE.test(String(meta.slug))) {
    errors.push(`frontmatter: \`slug\` must be kebab-case, got ${JSON.stringify(meta.slug)}`);
  }
  if (meta.canonical !== undefined && !/^https?:\/\/\S+$/.test(String(meta.canonical))) {
    errors.push('frontmatter: `canonical` must be an absolute http(s) URL');
  }
}

function checkBody(body, errors) {
  const images = [...body.matchAll(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)].map((m) => ({
    alt: m[1],
    src: m[2]
  }));
  for (const img of images) {
    if (!IMAGE_RE.test(img.src)) {
      errors.push(
        `image \`${img.src}\` is not allowed: use /images/posts/<slug>/<name>.svg|gif|webp ` +
          '(raster jpg/png are refused for weight; svg, gif and webp are the accepted three)'
      );
    }
    if (img.alt.trim() === '') errors.push(`image \`${img.src}\` has empty alt text`);
  }

  const noCode = stripCode(body);
  for (const [tag, why] of [
    ['img', 'use markdown `![alt](/images/posts/…)` so the format allowlist and the size injection apply'],
    ['iframe', 'use the <Video> component so the embed stays behind a facade'],
    ['script', 'posts may not carry script']
  ]) {
    const re = new RegExp(`<${tag}[\\s>]`, 'i');
    if (re.test(noCode)) errors.push(`raw <${tag}> in the body: ${why}`);
  }

  for (const m of noCode.matchAll(/<Video\b([^>]*)>/g)) {
    const attrs = m[1];
    for (const need of ['provider', 'id', 'title', 'poster']) {
      if (!new RegExp(`\\b${need}\\s*=`).test(attrs)) errors.push(`<Video> is missing \`${need}\``);
    }
    const prov = /\bprovider\s*=\s*"([^"]*)"/.exec(attrs)?.[1];
    if (prov !== undefined && !VIDEO_PROVIDERS.includes(prov)) {
      errors.push(`<Video provider="${prov}"> is not allowed (known: ${VIDEO_PROVIDERS.join(', ')})`);
    }
    const poster = /\bposter\s*=\s*"([^"]*)"/.exec(attrs)?.[1];
    if (poster !== undefined && !IMAGE_RE.test(poster)) {
      errors.push(`<Video poster="${poster}"> must be a local /images/posts/<slug>/<name>.webp`);
    }
  }

  // Currency looks exactly like inline math to remark-math, and the failure is
  // silent: "$5 and $6" renders as a formula instead of two prices.
  const dollars = [...noCode.matchAll(/(^|[^\\])\$/g)].length;
  if (dollars % 2 === 1) {
    errors.push('an odd number of unescaped `$` in the prose — write a price as \\$5, or close the math');
  }
  const money = /(^|[^\\])\$\d[\d,.]*\s[\s\S]*?[^\\]\$\d/.exec(noCode);
  if (money) errors.push('two `$` amounts look like currency but read as math — escape them as \\$');

  return images;
}

/**
 * Words outside fenced code, as minutes.
 *
 * Code is excluded on purpose: a post that is mostly a shell session is not a
 * twenty-minute read, and the number is shown to a reader deciding whether to
 * start.
 */
export const readingMinutes = (body) =>
  Math.max(1, Math.round(stripCode(body).split(/\s+/).filter((w) => /[A-Za-z0-9]/.test(w)).length / WPM));

/** Whether the post uses math at all — drives the per-page KaTeX stylesheet. */
export const hasMath = (body) => {
  const noCode = stripCode(body);
  return /(^|[^\\])\$[^$\n]+\$/.test(noCode) || fences(body).some((f) => f.lang === 'latex');
};

/**
 * Everything derivable from one post's source. Throws on any violation, with
 * every problem listed at once rather than one per build.
 *
 * @param {string} src
 * @param {{tags: object, authors: object}} vocab
 * @param {string} [name] file name, for the error message
 */
export function analysePost(src, vocab, name = 'post') {
  const { meta, errors } = parseFrontmatter(src);
  const { body } = splitFrontmatter(src);
  checkMeta(meta, vocab, errors);
  const images = checkBody(body, errors);
  if (errors.length) {
    throw new Error(`${name}:\n  - ${errors.join('\n  - ')}`);
  }
  return {
    meta,
    images,
    readingMinutes: readingMinutes(body),
    hasMath: hasMath(body)
  };
}

/** Non-throwing form, for tests and for reporting every post at once in CI. */
export function validatePost(src, vocab) {
  try {
    analysePost(src, vocab);
    return [];
  } catch (e) {
    return String(e.message).split('\n  - ').slice(1);
  }
}
