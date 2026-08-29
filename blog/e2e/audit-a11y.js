/**
 * Accessibility diagnostic for the blog — contrast, accessible names, heading
 * order, h1 count.
 *
 * NOT wired into CI, deliberately. The marketing site is audited by
 * lighthouse.yml against `minScore: 1`, and the honest way to extend that to
 * the blog is to observe a first real score and set the budget from it — not
 * to declare a threshold in the same change that first measures one. Until
 * that score exists this is a manual tool, and it is the tool to run after any
 * change to the palette, the type ramp or the heading structure.
 *
 * Usage — build, serve, then inject and dump:
 *
 *   cd blog && bun x --bun vite build
 *   (cd build && python3 -m http.server 8199 &)
 *   # then, for each page, insert this file before </body> and read #a11y-audit
 *
 * It reproduces four of Lighthouse's accessibility audits closely enough to be
 * useful and cheap enough to run against every route:
 *
 *   color-contrast   composites through transparent ancestors to the first
 *                    opaque background, applies the 3:1 large-text rule
 *   link-name        counts aria-label, text, <img alt> and <svg><title>
 *   heading-order    flags any jump of more than one level
 *   page-has-heading-one
 *
 * It does NOT rasterise the WebGL canvas — neither does Lighthouse. The field's
 * contribution to contrast is bounded separately and exhaustively by
 * .contrast-check.mjs at the repo root, which is the stronger statement.
 *
 * Proved able to fail: with three defects planted in a page (body copy at
 * 1.77:1, metadata at 3.78:1, an anchor wrapping only an aria-hidden svg, and
 * an h5 after an h2) it reports all four. A checker that has only ever printed
 * "all pass" has not been tested.
 */
(() => {
  const lin = c => c <= 0.03928 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4);
  const parse = s => {
    const m = s.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
    return m ? [+m[1], +m[2], +m[3], m[4] === undefined ? 1 : +m[4]] : null;
  };
  const L = ([r,g,b]) => 0.2126*lin(r/255) + 0.7152*lin(g/255) + 0.0722*lin(b/255);
  const over = (fg, bg) => { // composite fg (with alpha) over opaque bg
    const a = fg[3];
    return [0,1,2].map(i => fg[i]*a + bg[i]*(1-a));
  };
  const contrast = (a, b) => { const [h,l] = [L(a), L(b)].sort((x,y)=>y-x); return (h+0.05)/(l+0.05); };

  const effBg = el => {
    let n = el, acc = null;
    while (n && n !== document.documentElement.parentNode) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c[3] > 0) {
        acc = acc === null ? c.slice() : [...over(acc, c), 1];
        if (c[3] === 1) return acc.slice(0,3);
      }
      n = n.parentElement;
    }
    return acc ? acc.slice(0,3) : [255,255,255];
  };

  const hasText = el => [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length);
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    if (!hasText(el)) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const fg = parse(cs.color); if (!fg) continue;
    const bg = effBg(el);
    const c = contrast(fg[3] < 1 ? over(fg, bg) : fg.slice(0,3), bg);
    const px = parseFloat(cs.fontSize);
    const bold = (parseInt(cs.fontWeight,10) || 400) >= 700;
    const large = px >= 24 || (px >= 18.66 && bold);
    const need = large ? 3 : 4.5;
    if (c < need) {
      out.push(`${c.toFixed(2)} (need ${need})  <${el.tagName.toLowerCase()}${el.className && typeof el.className==='string' ? '.'+el.className.trim().split(/\s+/).slice(0,2).join('.') : ''}>  ${px}px${bold?' bold':''}  "${el.textContent.trim().slice(0,42)}"`);
    }
  }
  const pre = document.createElement('pre');
  pre.id = 'a11y-audit';
  pre.textContent = out.length ? `CONTRAST FAILURES (${out.length}):\n` + out.join('\n') : 'CONTRAST: all pass';

  // link/button name
  const noname = [];
  for (const el of document.querySelectorAll('a[href], button')) {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const imgAlt = [...el.querySelectorAll('img[alt]')].map(i => i.alt).join(' ');
    const svgTitle = [...el.querySelectorAll('svg > title')].map(s => s.textContent).join(' ');
    const name = (el.getAttribute('aria-label') || el.textContent || imgAlt || svgTitle || el.title || '').trim();
    if (!name) noname.push(`<${el.tagName.toLowerCase()}> ${el.outerHTML.slice(0,90)}`);
  }
  pre.textContent += noname.length ? `\n\nNO ACCESSIBLE NAME (${noname.length}):\n` + noname.join('\n') : '\n\nNAMES: all pass';

  // heading order
  const hs = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(h => +h.tagName[1]);
  const skips = [];
  for (let i = 1; i < hs.length; i++) if (hs[i] > hs[i-1] + 1) skips.push(`h${hs[i-1]} -> h${hs[i]} at #${i}`);
  pre.textContent += skips.length ? `\n\nHEADING SKIPS: ${skips.join(', ')}` : '\n\nHEADINGS: no skips';
  pre.textContent += `\n\nH1 COUNT: ${document.querySelectorAll('h1').length}`;

  document.body.prepend(pre);
})();
