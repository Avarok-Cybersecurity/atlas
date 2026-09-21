// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { authors, tags } from '../content.js';
import { MARK, compileMarkdown, slugify, svelteEscape } from './compile.js';

/** A stub highlighter, so these tests measure the emitter and not shiki. */
const highlight = { code: (c, l) => `<pre class="shiki" data-lang="${l}"><code>${c}</code></pre>` };
const V = { tags, authors, highlight };

const FRONT = `---
title: A post
dek: One line.
categories: [engineering]
date: 2026-08-30
keywords: [atlasctl]
og-image: ''
---
`;
const compile = (body, front = FRONT) => compileMarkdown(`${front}\n${body}`, { ...V, filename: 't.md' });

describe('the emitted component', () => {
  test('exports meta and imports the shared post components', () => {
    const { svelte, meta } = compile('Hello.\n');
    expect(svelte).toContain('<script module>');
    expect(svelte).toContain('export const meta =');
    for (const c of ['H2', 'Code', 'Table', 'Callout', 'Video']) {
      expect(svelte).toContain(`import ${c} from '$lib/components/${c}.svelte'`);
    }
    expect(meta.tag).toBe('engineering');
    expect(meta.format).toBe('md');
  });

  test('h2 headings get sequential indices so the chevron cycle works', () => {
    const { svelte } = compile('## One\n\ntext\n\n## Two\n\n## Three\n');
    expect(svelte).toContain('<H2 id="one" index={0}>');
    expect(svelte).toContain('<H2 id="two" index={1}>');
    expect(svelte).toContain('<H2 id="three" index={2}>');
  });

  test('h3 stays a plain heading with an id', () => {
    expect(compile('### Deeper\n').svelte).toContain('<h3 id="deeper">');
  });

  test('a fence becomes a Code component carrying both forms', () => {
    const { svelte } = compile('```bash name=run.sh\necho hi\n```\n');
    expect(svelte).toContain('<Code lang="bash" name="run.sh"');
    // marked hands the fence body over without its trailing newline
    expect(svelte).toContain('code={"echo hi"}');
    expect(svelte).toContain('data-lang=\\"bash\\"');
  });

  test('right-aligned table columns keep the numeric class', () => {
    const { svelte } = compile('| a | b |\n|---|--:|\n| 1 | 2 |\n');
    expect(svelte).toContain('<Table>');
    expect(svelte).toContain('<th class="num">b</th>');
    expect(svelte).toContain('<td class="num">2</td>');
  });

  test('the two allowed components pass through untouched', () => {
    const { svelte } = compile('<Callout label="L" tone="verified">Body.</Callout>\n');
    expect(svelte).toContain('<Callout label="L" tone="verified">');
  });

  test('the first image loads eagerly and later ones lazily', () => {
    const { svelte } = compile(
      '![one](/images/posts/p/a.webp)\n\n![two](/images/posts/p/b.webp)\n'
    );
    // The first image is the likely LCP element; deferring it is a real loss.
    expect(svelte.indexOf('loading="eager"')).toBeLessThan(svelte.indexOf('loading="lazy"'));
    expect(svelte.match(/loading="eager"/g)).toHaveLength(1);
  });

  test('image dimensions are written when a measurer supplies them', () => {
    const { svelte } = compileMarkdown(`${FRONT}\n![a](/images/posts/p/a.webp)\n`, {
      ...V,
      measure: () => ({ width: 800, height: 400 })
    });
    expect(svelte).toContain('width="800" height="400"');
  });
});

describe('escaping — where this kind of compiler usually breaks', () => {
  test('braces and backticks in prose cannot become Svelte syntax', () => {
    const { svelte } = compile('A brace { here } and a tick ` there.\n');
    expect(svelte).toContain('&#123;');
    expect(svelte).toContain('&#125;');
    expect(svelte).not.toMatch(/<p>[^<]*\{ here \}/);
  });

  test('a literal Svelte block in prose survives as text', () => {
    const { svelte } = compile('Write {#if x}foo{/if} to branch.\n');
    expect(svelte).not.toContain('{#if x}');
    expect(svelte).toContain('&#123;#if x&#125;');
  });

  test('NEGATIVE CONTROL: the escape pass does not corrupt component props', () => {
    // A global brace-escape would mangle `code={...}` two files away — the
    // exact bug the placeholder indirection exists to prevent.
    const { svelte } = compile('```js\nconst a = {b: 1};\n```\n');
    expect(svelte).toContain('code={"const a = {b: 1};"}');
  });

  test('a codespan keeps its dollars and angle brackets', () => {
    const { svelte } = compile('Run `echo $PATH` and `Vec<T>` today.\n');
    expect(svelte).toContain('<code>echo $PATH</code>');
    expect(svelte).toContain('&lt;T&gt;');
  });

  test('a source using the reserved marker is refused, not mangled', () => {
    expect(() => compile(`text ${MARK}0@@ more\n`)).toThrow(/reserved marker/);
  });
});

describe('maths', () => {
  test('inline maths renders to KaTeX with MathML', () => {
    const { svelte, meta } = compile('Cost is $O(n)$ here.\n');
    expect(svelte).toContain('class=\\"katex\\"');
    expect(svelte).toContain('<math');
    expect(meta.hasMath).toBe(true);
  });

  test('a latex fence renders as display maths, not as code', () => {
    const { svelte } = compile('```latex\ne = mc^2\n```\n');
    expect(svelte).toContain('katex-display');
    expect(svelte).not.toContain('<Code lang="latex"');
  });

  test('NEGATIVE CONTROL: a dollar in code is never maths', () => {
    const { svelte, meta } = compile('Use `echo $HOME` now.\n');
    expect(svelte).not.toContain('class=\\"katex\\"');
    expect(meta.hasMath).toBe(false);
  });

  test('an escaped dollar stays a price', () => {
    const { svelte } = compile('It costs \\$99 today.\n');
    expect(svelte).toContain('$99');
    expect(svelte).not.toContain('katex');
  });

  test('a broken formula fails the build rather than rendering an error box', () => {
    expect(() => compile('Bad $\\frac{1}$ here.\n')).toThrow();
  });
});

describe('rules the compiler enforces before emitting', () => {
  test('a disallowed image format stops the build', () => {
    expect(() => compile('![a](/images/posts/p/a.png)\n')).toThrow(/not allowed/);
  });

  test('raw html other than the two components is refused', () => {
    expect(() => compile('<div>nope</div>\n')).toThrow(/raw/i);
  });

  test('an unknown category stops the build', () => {
    const bad = FRONT.replace('[engineering]', '[kernels]');
    expect(() => compile('x\n', bad)).toThrow(/kernels/);
  });
});

describe('slugify', () => {
  test('matches the anchors readers expect', () => {
    expect(slugify('Running atlasctl')).toBe('running-atlasctl');
    expect(slugify('A/B, then C?')).toBe('ab-then-c');
    expect(slugify('  Spaced   Out  ')).toBe('spaced-out');
  });
});

describe('svelteEscape', () => {
  test('covers the three characters Svelte parses, and entities survive', () => {
    expect(svelteEscape('{a}`b`')).toBe('&#123;a&#125;&#96;b&#96;');
    expect(svelteEscape('&amp; stays')).toBe('&amp; stays');
    expect(svelteEscape('a & b')).toBe('a &amp; b');
  });
});
