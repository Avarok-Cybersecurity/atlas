// SPDX-License-Identifier: AGPL-3.0-only

// compile.js — markdown source in, Svelte component source out.
//
// Markdown is an ADDITION to the Svelte post format, not a replacement: a `.md`
// post compiles to the same kind of component a hand-written post already is —
// `meta` from `<script module>`, body built from H2/Code/Table/Callout — so the
// two formats land in one index, one feed, one prev/next chain, and look
// identical on the page.
//
// Everything here runs in the BUILD. The compiled component contains no parser,
// no highlighter and no math engine; `e2e/check-bundle.mjs` asserts that,
// because "build-time only" is a claim that rots the first time someone imports
// this module from a component.
//
// -- Escaping, which is where this kind of compiler usually goes wrong --------
// Svelte treats `{`, `}` and backticks as syntax. Markdown output is full of
// them, and highlighted code is worse. A global escape pass over the finished
// markup would corrupt our own `code={...}` props two files away. So component
// markup is emitted as an opaque PLACEHOLDER, the escape pass runs over what is
// left (which is only prose), and the placeholders are substituted afterwards.

import { Marked } from 'marked';
import { analysePost, fences, splitFrontmatter } from '../postmd.js';
import { math } from './highlight.js';

/** Prose-safe text: HTML entities plus the three characters Svelte parses. */
export const svelteEscape = (s) =>
  String(s)
    .replace(/&(?![a-zA-Z#][a-zA-Z0-9]*;)/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\{/g, '&#123;')
    .replace(/\}/g, '&#125;')
    .replace(/`/g, '&#96;');

/** GitHub-style heading slug, so in-post anchors match what readers expect. */
export const slugify = (text) =>
  String(text)
    .toLowerCase()
    // No tag-stripping pass: `<...>` removal by regex is incomplete
    // sanitisation (`<scr<script>ipt>` survives it), and it is unnecessary
    // here — the next rule already drops every character that is not a word
    // character, whitespace or a hyphen, angle brackets included.
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-');

/**
 * Placeholder marker for component markup.
 *
 * Printable rather than a control character so the emitted source stays
 * greppable and diffable, and improbable enough that a post containing it is
 * almost certainly a mistake — `compileMarkdown` refuses such a source rather
 * than silently mangling it.
 */
export const MARK = '@@ATLASMD';

/** Raw HTML the body may contain: exactly the two components, nothing else. */
const ALLOWED_TAGS = /^<\/?(Callout|Video)\b/;

/**
 * Compile one markdown post.
 *
 * @param {string} src raw file contents
 * @param {object} opts
 * @param {string} [opts.filename] for error messages
 * @param {object} opts.tags declared categories (content.js)
 * @param {object} opts.authors declared authors (content.js)
 * @param {{code: (code: string, lang: string) => string}} opts.highlight injected so
 *   the pure tests run without shiki — the repo's SBIO habit
 * @param {(src: string) => {width: number, height: number}|null} [opts.measure] image
 *   sizer, injected for the same reason
 * @returns {{meta: object, svelte: string}}
 */
export function compileMarkdown(src, { filename, tags, authors, highlight, measure }) {
  const name = filename ? filename.split('/').pop() : 'post';
  if (src.includes(MARK)) {
    throw new Error(`${name}: source contains the reserved marker ${MARK}`);
  }
  // Validation first: a post that breaks a rule never reaches the emitter, so
  // the emitter may assume its input is well formed.
  const { meta: front, readingMinutes, hasMath } = analysePost(src, { tags, authors }, name);
  const { body } = splitFrontmatter(src);

  const parts = [];
  const hold = (markup) => {
    parts.push(markup);
    return `${MARK}${parts.length - 1}@@`;
  };

  let h2 = 0;
  let images = 0;

  const marked = new Marked({ gfm: true });
  marked.use({
    extensions: [
      {
        name: 'inlineMath',
        level: 'inline',
        start: (s) => s.indexOf('$'),
        tokenizer(s) {
          // Non-space either side, as in every $...$ convention. A `\$` never
          // reaches here: marked resolves backslash escapes first.
          const m = /^\$(?![\s$])((?:[^$\\\n]|\\.)+?)(?<![\s\\])\$/.exec(s);
          return m ? { type: 'inlineMath', raw: m[0], text: m[1] } : undefined;
        },
        renderer: (t) => hold(`{@html ${JSON.stringify(math(t.text, false))}}`)
      }
    ],
    renderer: {
      heading(token) {
        const text = this.parser.parseInline(token.tokens);
        const id = slugify(token.text);
        if (token.depth === 2) {
          // index drives the violet -> cyan -> green -> gold chevron cycle,
          // exactly as a hand-written <H2 index={n}> does.
          return hold(`<H2 id="${id}" index={${h2++}}>${text}</H2>\n`);
        }
        return hold(`<h${token.depth} id="${id}">${text}</h${token.depth}>\n`);
      },
      code(token) {
        const info = String(token.lang ?? '');
        const lang = info.split(/\s+/)[0] ?? '';
        const label = /name=(\S+)/.exec(info)?.[1] ?? '';
        if (lang === 'latex') return hold(`{@html ${JSON.stringify(math(token.text, true))}}\n`);
        const html = highlight.code(token.text, lang);
        return hold(
          `<Code lang=${JSON.stringify(lang)} name=${JSON.stringify(label)} ` +
            `code={${JSON.stringify(token.text)}} html={${JSON.stringify(html)}} />\n`
        );
      },
      image(token) {
        const dim = measure ? measure(token.href) : null;
        const size = dim ? ` width="${dim.width}" height="${dim.height}"` : '';
        // The first image is the likely LCP element; lazy-loading it is a
        // measurable performance loss, so only later images defer.
        const loading = images++ === 0 ? 'eager' : 'lazy';
        return hold(
          `<img src=${JSON.stringify(token.href)} alt=${JSON.stringify(token.text)}${size}` +
            ` loading="${loading}" decoding="async" />`
        );
      },
      table(token) {
        const cell = (c, tag) => {
          const cls = c.align === 'right' ? ' class="num"' : c.align ? ` class="${c.align}"` : '';
          return `<${tag}${cls}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        };
        const head = token.header.map((c) => cell(c, 'th')).join('');
        const rows = token.rows.map((r) => `<tr>${r.map((c) => cell(c, 'td')).join('')}</tr>`).join('\n');
        return hold(`<Table><thead><tr>${head}</tr></thead><tbody>\n${rows}\n</tbody></Table>\n`);
      },
      html(token) {
        const text = String(token.text);
        if (!ALLOWED_TAGS.test(text.trim())) {
          throw new Error(`${name}: raw HTML is not allowed in a post: ${text.trim().slice(0, 60)}`);
        }
        return hold(text);
      },
      codespan: (token) => `<code>${svelteEscape(token.text)}</code>`,
      // Escaping happens HERE, on the text leaves, rather than over the
      // finished markup. Escaping the whole document and then un-escaping the
      // tags by name meant turning `&amp;` back into `&` — a double-unescape
      // that is both a real hazard and impossible to reason about. Overriding
      // `text` hands us the raw leaf before marked escapes anything, so the
      // structure marked emits is never touched and prose is escaped exactly
      // once.
      text: (token) => svelteEscape(token.text)
    }
  });

  const html = marked.parse(body);
  const restored = html.replace(new RegExp(`${MARK}(\\d+)@@`, 'g'), (_, i) => parts[Number(i)]);

  const meta = {
    format: 'md',
    title: front.title,
    dek: front.dek,
    date: front.date,
    tag: front.categories[0],
    categories: front.categories,
    keywords: front.keywords,
    ogImage: front['og-image'] ?? '',
    author: front.author ?? 'thomas-braun',
    readingMinutes,
    hasMath,
    draft: front.draft === true,
    updated: front.updated ?? null,
    canonical: front.canonical ?? null,
    slug: front.slug ?? null
  };

  const svelte = `<script module>
  export const meta = ${JSON.stringify(meta)};
</script>

<script>
  import H2 from '$lib/components/H2.svelte';
  import Code from '$lib/components/Code.svelte';
  import Table from '$lib/components/Table.svelte';
  import Callout from '$lib/components/Callout.svelte';
  import Video from '$lib/components/Video.svelte';
</script>

${restored}`;

  return { meta, svelte };
}

/** Languages a post's fences use, so a build can preload only what it needs. */
export const langsUsed = (src) => [...new Set(fences(src).map((f) => f.lang).filter(Boolean))];
