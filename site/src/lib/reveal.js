// SPDX-License-Identifier: AGPL-3.0-only
//
// Scroll reveal for `.av-reveal` elements. One IntersectionObserver per page,
// attached to the shell, so sections fade up as they enter. Elements are
// visible without JavaScript (the class only hides once the observer exists)
// and under prefers-reduced-motion (the CSS ignores the class).
export function reveal(root) {
  if (typeof IntersectionObserver === 'undefined') return;
  const targets = root.querySelectorAll('.av-reveal');
  if (!targets.length) return;
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
  return { destroy: () => io.disconnect() };
}
