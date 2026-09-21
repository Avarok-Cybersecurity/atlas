export const prerender = true;
export const ssr = true;
// Every route is a static document; the client router still handles in-site
// navigation, which is what keeps the WebGL context alive across a click
// instead of tearing it down and recompiling the shader on every page.
export const trailingSlash = 'never';
