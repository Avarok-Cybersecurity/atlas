// SPDX-License-Identifier: AGPL-3.0-only
//
// The site is three documents that share a header, not one.
//
// The marketing pages load avarok.css. The developer pages (/engine, /control,
// /diligence) load app.css and its companions, a different design system whose
// rules are written against bare elements: `nav`, `header`, `section`. Each
// group's layout imports only its own stylesheets, and on a full page load that
// keeps them apart.
//
// The client router does not. It never removes a stylesheet, so moving from one
// group to the other by a client side navigation leaves both systems in the
// page. Worse, the router PRELOADS a link's code and styles when the pointer
// rests on it, so merely passing over "Developers" on the way to "Company"
// injected app.css into a marketing page and collapsed its header. Nothing had
// been clicked. A refresh cured it, which is what made it look random.
//
// So a link that crosses groups is always a full page load, and is never
// preloaded. The root layout marks such links `data-sveltekit-reload` the moment
// the pointer, the focus or a finger reaches them, which is before the router
// looks, and refuses any navigation that still crosses (a `goto`, the back
// button). This file is the rule. src/routes/+layout.svelte applies it.

/** Pages that carry the developer design system. */
export const DEVELOPER_PATHS = ['/engine', '/control', '/diligence'];
/** Pages with no site chrome at all. */
export const BARE_PATHS = ['/broll'];

const clean = (pathname) => pathname.replace(/\.html$/, '').replace(/\/+$/, '') || '/';
const under = (path, roots) => roots.some((r) => path === r || path.startsWith(r + '/'));

/**
 * Which document a path belongs to.
 * @param {string} pathname
 * @returns {'developer' | 'bare' | 'marketing'}
 */
export function groupOf(pathname) {
  const path = clean(pathname);
  if (under(path, DEVELOPER_PATHS)) return 'developer';
  if (under(path, BARE_PATHS)) return 'bare';
  return 'marketing';
}

/**
 * Whether following `href` from `fromPathname` leaves the current document.
 * Links to other origins are not ours to manage and never cross.
 * @param {string} fromPathname
 * @param {string} href        absolute or relative
 * @param {string} origin      the page's own origin
 */
export function crossesGroup(fromPathname, href, origin) {
  let url;
  try {
    url = new URL(href, origin + fromPathname);
  } catch {
    return false;
  }
  if (url.origin !== origin) return false;
  if (!/^https?:$/.test(url.protocol)) return false;
  return groupOf(url.pathname) !== groupOf(fromPathname);
}
