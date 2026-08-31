// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { tags, authors } from './content.js';
import {
  analysePost, fences, hasMath, parseFrontmatter, readingMinutes,
  splitFrontmatter, stripCode, validatePost
} from './postmd.js';

const V = { tags, authors };
const FRONT = {
  title: 'A post',
  dek: 'One line about it.',
  categories: '[engineering, design]',
  date: '2026-08-30',
  keywords: '[atlasctl, nvfp4]',
  'og-image': "''"
};
/** Build a post source, overriding or deleting frontmatter fields. */
const post = (over = {}, body = 'Some prose.\n') => {
  const f = { ...FRONT, ...over };
  const lines = Object.entries(f)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}: ${v}`);
  return `---\n${lines.join('\n')}\n---\n\n${body}`;
};
const errs = (over, body) => validatePost(post(over, body), V);

describe('frontmatter', () => {
  test('a well-formed post validates and yields its metadata', () => {
    const { meta } = analysePost(post(), V);
    expect(meta.title).toBe('A post');
    expect(meta.categories).toEqual(['engineering', 'design']);
    expect(meta.keywords).toEqual(['atlasctl', 'nvfp4']);
    expect(meta['og-image']).toBe('');
  });

  test('every required field is actually required', () => {
    for (const k of ['title', 'dek', 'categories', 'date', 'keywords', 'og-image']) {
      const e = errs({ [k]: undefined });
      expect(e.join(' ')).toContain(`missing required field \`${k}\``);
    }
  });

  test('a post with no frontmatter at all is rejected', () => {
    expect(validatePost('# just a heading\n', V).join(' ')).toContain('no frontmatter');
  });

  test('unknown fields are rejected rather than ignored', () => {
    // `og_image` instead of `og-image` must be loud: silently falling back to
    // the default card is the failure this catches.
    expect(errs({ og_image: '/images/og/x.png' }).join(' ')).toContain('unknown field `og_image`');
  });

  test('categories must come from the declared set', () => {
    expect(errs({ categories: '[kernels]' }).join(' ')).toContain('unknown category `kernels`');
    expect(errs({ categories: '[]' }).join(' ')).toContain('at least one entry');
    expect(errs({ categories: 'engineering' }).join(' ')).toContain('must be a flow list');
  });

  test('NEGATIVE CONTROL: every declared category is accepted', () => {
    // Without this the category test would pass on a rule that rejects
    // everything.
    for (const c of Object.keys(tags)) expect(errs({ categories: `[${c}]` })).toEqual([]);
  });

  test('dates must be real and strictly formatted', () => {
    for (const bad of ['last tuesday', '2026-8-1', '2026-13-01', '30-08-2026']) {
      expect(errs({ date: bad }).join(' ')).toContain('`date` must be a real YYYY-MM-DD');
    }
    expect(errs({ date: '2026-02-28' })).toEqual([]);
  });

  test('og-image is empty or a png/webp card', () => {
    expect(errs({ 'og-image': '/images/og/a.png' })).toEqual([]);
    expect(errs({ 'og-image': '/images/og/a.webp' })).toEqual([]);
    // svg and gif are body formats; scrapers will not rasterise them
    expect(errs({ 'og-image': '/images/og/a.svg' }).join(' ')).toContain('og-image');
    expect(errs({ 'og-image': 'images/og/a.png' }).join(' ')).toContain('og-image');
  });

  test('author and slug are checked against reality', () => {
    expect(errs({ author: 'ghost' }).join(' ')).toContain('unknown author `ghost`');
    expect(errs({ author: 'thomas-braun' })).toEqual([]);
    expect(errs({ slug: 'Has Spaces' }).join(' ')).toContain('kebab-case');
    expect(errs({ slug: 'fine-slug' })).toEqual([]);
  });

  test('draft is a real boolean', () => {
    expect(errs({ draft: 'yes' }).join(' ')).toContain('must be true or false');
    expect(errs({ draft: 'true' })).toEqual([]);
  });

  test('duplicate keys are caught', () => {
    const src = `---\ntitle: a\ntitle: b\ndek: d\ncategories: [design]\ndate: 2026-01-01\nkeywords: [k]\nog-image: ''\n---\nx\n`;
    expect(validatePost(src, V).join(' ')).toContain('duplicate key');
  });
});

describe('images', () => {
  test('only svg, gif and webp are accepted', () => {
    for (const ext of ['svg', 'gif', 'webp']) {
      expect(errs({}, `![alt](/images/posts/p/a.${ext})\n`)).toEqual([]);
    }
    for (const ext of ['png', 'jpg', 'jpeg', 'avif']) {
      expect(errs({}, `![alt](/images/posts/p/a.${ext})\n`).join(' ')).toContain('is not allowed');
    }
  });

  test('the path shape is enforced, not just the extension', () => {
    expect(errs({}, '![alt](https://cdn.example/a.webp)\n').join(' ')).toContain('is not allowed');
    expect(errs({}, '![alt](/a.webp)\n').join(' ')).toContain('is not allowed');
  });

  test('empty alt text is rejected', () => {
    expect(errs({}, '![](/images/posts/p/a.webp)\n').join(' ')).toContain('empty alt text');
  });

  test('raw html that bypasses the rules is rejected', () => {
    expect(errs({}, '<img src="/x.png">\n').join(' ')).toContain('raw <img>');
    expect(errs({}, '<iframe src="https://y"></iframe>\n').join(' ')).toContain('raw <iframe>');
    expect(errs({}, '<script>alert(1)</script>\n').join(' ')).toContain('raw <script>');
  });

  test('NEGATIVE CONTROL: those tags inside code are prose, not markup', () => {
    expect(errs({}, 'Use `<img>` sparingly.\n')).toEqual([]);
    expect(errs({}, '```html\n<iframe src="x"></iframe>\n```\n')).toEqual([]);
  });
});

describe('video', () => {
  const vid = (attrs) => `<Video ${attrs} />\n`;
  const OK = 'provider="youtube" id="abc" title="T" poster="/images/posts/p/a.webp"';

  test('a complete embed is accepted', () => {
    expect(errs({}, vid(OK))).toEqual([]);
  });

  test('every attribute is required', () => {
    for (const drop of ['provider', 'id', 'title', 'poster']) {
      const attrs = OK.split(' ').filter((a) => !a.startsWith(`${drop}=`)).join(' ');
      expect(errs({}, vid(attrs)).join(' ')).toContain(`missing \`${drop}\``);
    }
  });

  test('the provider allowlist and a local poster are enforced', () => {
    expect(errs({}, vid(OK.replace('youtube', 'dailymotion'))).join(' ')).toContain('not allowed');
    expect(errs({}, vid(OK.replace('/images/posts/p/a.webp', 'https://i.ytimg.com/x.jpg'))).join(' '))
      .toContain('must be a local');
  });
});

describe('math and code', () => {
  test('a shell `$` is never mistaken for math', () => {
    expect(errs({}, 'Run `echo $PATH` first.\n')).toEqual([]);
    expect(errs({}, '```bash\nexport A=$HOME\ncd $A\n```\n')).toEqual([]);
  });

  test('currency that reads as math is refused with the fix in the message', () => {
    const e = errs({}, 'It costs $5 and $6 more.\n').join(' ');
    expect(e).toContain('\\$');
  });

  test('escaped currency passes', () => {
    expect(errs({}, 'It costs \\$5 and \\$6 more.\n')).toEqual([]);
  });

  test('an unclosed `$` is caught', () => {
    expect(errs({}, 'The cost is $x for each.\n').join(' ')).toContain('odd number');
  });

  test('real inline math passes', () => {
    expect(errs({}, 'Attention is $O(n^2 d)$ per layer.\n')).toEqual([]);
  });

  test('hasMath sees both forms and ignores code', () => {
    expect(hasMath('cost $O(n)$ here')).toBe(true);
    expect(hasMath('```latex\ne=mc^2\n```')).toBe(true);
    expect(hasMath('`$PATH` and ```bash\n$X\n```')).toBe(false);
    expect(hasMath('plain prose')).toBe(false);
  });

  test('fences are parsed with their language and meta', () => {
    const f = fences('```bash name=run.sh\necho hi\n```\n');
    expect(f).toHaveLength(1);
    expect(f[0].lang).toBe('bash');
    expect(f[0].meta).toBe('name=run.sh');
    expect(f[0].code).toBe('echo hi\n');
  });

  test('stripCode blanks code but keeps line numbering', () => {
    const src = 'a\n```\nx\n```\nb';
    expect(stripCode(src).split('\n')).toHaveLength(src.split('\n').length);
    expect(stripCode(src)).toContain('a');
    expect(stripCode(src)).not.toContain('x');
  });
});

describe('derived fields', () => {
  test('reading time excludes fenced code', () => {
    // The fence must start its own line — a fixture that ran it onto the end
    // of the prose was not testing a fence at all.
    const prose = `${'word '.repeat(220)}\n\n`;
    const code = '```bash\n' + 'echo hello world\n'.repeat(400) + '```\n';
    expect(readingMinutes(prose)).toBe(1);
    expect(readingMinutes(prose + code)).toBe(1);
  });

  test('NEGATIVE CONTROL: prose does count', () => {
    expect(readingMinutes('word '.repeat(2200))).toBeGreaterThan(5);
  });

  test('analysePost returns the images it validated', () => {
    const { images, hasMath: m } = analysePost(
      post({}, '![a](/images/posts/p/a.webp)\n![b](/images/posts/p/b.svg)\n'),
      V
    );
    expect(images.map((i) => i.src)).toEqual(['/images/posts/p/a.webp', '/images/posts/p/b.svg']);
    expect(m).toBe(false);
  });

  test('analysePost throws with every problem at once, naming the file', () => {
    let msg = '';
    try {
      analysePost(post({ date: 'nope', categories: '[bogus]' }), V, 'bad-post.md');
    } catch (e) {
      msg = e.message;
    }
    expect(msg).toContain('bad-post.md');
    expect(msg).toContain('`date`');
    expect(msg).toContain('bogus');
  });

  test('splitFrontmatter keeps the body intact', () => {
    const { body } = splitFrontmatter(post({}, 'line one\nline two\n'));
    expect(body).toBe('\nline one\nline two\n');
  });
});
