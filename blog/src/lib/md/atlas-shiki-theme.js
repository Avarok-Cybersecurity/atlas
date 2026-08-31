// SPDX-License-Identifier: AGPL-3.0-only

// atlas-shiki-theme.js — the code-block palette, expressed as the site's own
// design tokens rather than as a second set of hex values.
//
// Shiki themes must name concrete colours, and it inlines them into `style`
// attributes. Writing real hexes here would put a second copy of the palette in
// the repo, which `web-shared/atlas-tokens.css` exists to prevent. So the theme
// names SENTINELS — colours chosen only because nothing else uses them — and
// `colorReplacements` swaps each one for a `var()` reference as the HTML is
// emitted. The built pages therefore READ the token file at runtime and cannot
// drift from it, and changing the brand changes the code blocks with it.

/** sentinel -> the token that actually paints it. */
export const COLOR_REPLACEMENTS = {
  // The surface belongs to `.codeblock pre` (--bg-sunk). Mapping shiki's own
  // background to `transparent` is what stops it fighting that rule — cheaper
  // and less brittle than stripping the inline style afterwards.
  '#000000': 'transparent',
  '#000001': 'var(--t2)', //           plain text, punctuation
  '#000002': 'var(--ch-violet)', //    keywords, storage, operators
  '#000003': 'var(--ch-cyan)', //      functions, methods, tags
  '#000004': 'var(--green)', //        strings
  '#000005': 'var(--amber)', //        numbers, constants, booleans
  '#000006': 'var(--ch-gold)', //      types, classes, attribute names
  '#000007': 'var(--t3)', //           comments — the quietest ink, as in prose
  '#000008': 'var(--accent-deep)' //   regex, escapes
};

export const atlasTheme = {
  name: 'atlas',
  type: 'dark',
  colors: { 'editor.background': '#000000', 'editor.foreground': '#000001' },
  settings: [
    { scope: ['keyword', 'storage', 'keyword.operator', 'keyword.control'], settings: { foreground: '#000002' } },
    {
      scope: ['entity.name.function', 'support.function', 'entity.name.tag', 'variable.function'],
      settings: { foreground: '#000003' }
    },
    { scope: ['string', 'punctuation.definition.string'], settings: { foreground: '#000004' } },
    { scope: ['constant.numeric', 'constant.language', 'constant.other'], settings: { foreground: '#000005' } },
    {
      scope: ['entity.name.type', 'support.type', 'support.class', 'entity.other.attribute-name'],
      settings: { foreground: '#000006' }
    },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#000007' } },
    { scope: ['string.regexp', 'constant.character.escape'], settings: { foreground: '#000008' } }
  ]
};

/** Languages a post may use. An unknown language falls back to plain text. */
export const LANGS = [
  'bash', 'console', 'rust', 'javascript', 'json', 'toml', 'yaml',
  'python', 'nginx', 'glsl', 'html', 'css', 'diff', 'sql'
];
