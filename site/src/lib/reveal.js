// SPDX-License-Identifier: AGPL-3.0-only
//
// Scroll reveal for `.av-reveal` elements. One IntersectionObserver per page,
// attached to the shell, so sections fade up as they enter. Elements are
// visible without JavaScript (the class only hides once the observer exists)
// and under prefers-reduced-motion (the CSS ignores the class).
const LIVE = '.av-frame, .av-diagram';

export function reveal(root) {
  if (typeof IntersectionObserver === 'undefined') return;
  const targets = root.querySelectorAll('.av-reveal');
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.08 }
  );
  targets.forEach((t) => io.observe(t));

  // Anything that animates forever only does so while it is on screen. A ring
  // turning or wires marching below the fold still cost style and paint work
  // on every frame, which is how an idle page keeps a fan running.
  const live = new IntersectionObserver((entries) => {
    for (const e of entries) e.target.classList.toggle('is-live', e.isIntersecting);
  });
  root.querySelectorAll(LIVE).forEach((t) => live.observe(t));

  return {
    destroy: () => {
      io.disconnect();
      live.disconnect();
    }
  };
}
