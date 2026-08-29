/**
 * Everything about the blog that is copy rather than code.
 *
 * Kept apart from the components for the same reason the marketing site keeps
 * `data.js` apart: one place to change a link or a label, and no hunting
 * through markup for the third copy of a URL.
 */

/**
 * adapter-static writes /posts/foo to `posts/foo.html`, and nginx serves BOTH
 * that path and the extensionless one. The extensionless form is canonical, but
 * the `.html` URL is real, shareable, and what someone lands on if they copy a
 * link out of a directory listing or a crawler's index — and on that URL the
 * client router hands `load` a slug of "foo.html", which matches no post. So
 * every route strips it, and the layout canonicalises to the clean form.
 */
export const cleanSlug = (s) => s.replace(/\.html$/, '');

export const SITE = 'https://blog.atlasinference.io';
export const MAIN_SITE = 'https://atlasinference.io';
export const DOCS_SITE = 'https://docs.atlasinference.io';
export const githubUrl = 'https://github.com/Avarok-Cybersecurity/atlas';
export const discordUrl = 'https://discord.gg/PGUMSSpU';
export const xUrl = 'https://x.com/AtlasInferenceX';

export const blog = {
  name: 'Atlas blog',
  kicker: 'blog.atlasinference.io',
  title: 'Notes from the inference layer',
  lede:
    'Kernel work, measured benchmarks, and what it takes to run frontier models on hardware you own. ' +
    'Everything we publish is reproducible from a commit.',
  description:
    'Engineering notes from the Atlas inference engine: CUDA kernels, quantisation, ' +
    'speculative decoding, and benchmarks you can reproduce.'
};

/**
 * The four chevron colours carry fixed meanings on atlasinference.io — violet
 * = engine, cyan = silicon, green = verified, gold = community. Categories
 * inherit those meanings rather than inventing a fifth palette.
 */
export const tags = {
  engineering: { name: 'Engineering', color: 'var(--ch-cyan)', blurb: 'Kernels, memory, and the parts of the engine that decide the number.' },
  benchmarks: { name: 'Benchmarks', color: 'var(--ch-gold)', blurb: 'What we measured, on what hardware, with the harness attached.' },
  releases: { name: 'Releases', color: 'var(--ch-green)', blurb: 'What shipped, what it changes, and what it does not.' },
  design: { name: 'Design', color: 'var(--ch-violet)', blurb: 'The interface and the brand, measured the same way the engine is.' }
};

export const authors = {
  'thomas-braun': {
    name: 'Thomas Braun',
    initials: 'TB',
    role: 'Founder, Avarok Cybersecurity',
    bio: 'Works on the Atlas inference engine — kernels, scheduling, and the benchmarks that decide whether any of it was worth it.'
  }
};

export const nav = [
  { href: '/', label: 'Latest' },
  { href: '/tags/engineering', label: 'Engineering' },
  { href: '/tags/benchmarks', label: 'Benchmarks' },
  { href: '/tags/design', label: 'Design' }
];

export const footerCols = [
  {
    heading: 'Blog',
    links: [
      { text: 'Latest', href: '/' },
      { text: 'Engineering', href: '/tags/engineering' },
      { text: 'Benchmarks', href: '/tags/benchmarks' },
      { text: 'RSS feed', href: '/rss.xml' }
    ]
  },
  {
    heading: 'Atlas',
    links: [
      { text: 'atlasinference.io', href: MAIN_SITE },
      { text: 'Documentation', href: DOCS_SITE },
      { text: 'Benchmarks', href: `${MAIN_SITE}/#benchmarks` },
      { text: 'Download', href: `${MAIN_SITE}/#get-running` }
    ]
  },
  {
    heading: 'Community',
    links: [
      { text: 'GitHub', href: githubUrl },
      { text: 'Discord', href: discordUrl },
      { text: 'X', href: xUrl }
    ]
  }
];

/**
 * Is `href` the nav entry the current path belongs to?
 *
 * Two things this has to get right, and the inline version got neither:
 *   - `/posts/*` belongs to "Latest", so a reader on an article does not see
 *     an unlit bar;
 *   - the path may still carry `.html`, because adapter-static writes that
 *     file and nginx serves both forms — the chip row already handled this
 *     (its state comes from the load function, which strips it) and the nav
 *     did not, so on `/tags/engineering.html` the chip lit and the nav did not.
 */
export function navCurrent(pathname, href) {
  const p = cleanSlug(pathname).replace(/\/+$/, '') || '/';
  return href === '/' ? p === '/' || p.startsWith('/posts') : p === href;
}
