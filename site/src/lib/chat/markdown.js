// =============================================================================
// markdown.js — escape-first mini markdown renderer for assistant answers.
// XSS-safe by construction: ALL input text is HTML-escaped before any markup
// is generated, and the only attributes ever emitted are a fixed rel/target
// pair and an https?-validated href. Pure function — no DOM, no state.
//
// Supported: fenced code blocks, inline code, **bold**, [text](https?://…)
// links, unordered/ordered lists, paragraphs, and [n] citations rendered as
// <sup class="cc-cite"> for the UI to style/wire.
// =============================================================================

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Inline markup on an already-escaped, non-code text segment.
function renderSegment(escaped) {
  let out = escaped;
  // Links first so citation rewriting cannot eat a [text](url) opener.
  // href is escaped text and restricted to http(s), so it cannot break out of
  // the attribute or smuggle a javascript: scheme.
  // The URL charset excludes `[`, `]` and `*` — exactly the characters the two
  // passes BELOW rewrite. Those passes run over a string that already contains
  // the anchor emitted here, so a URL carrying them had its own href rewritten:
  //
  //   [x](https://e.com/[1]z)   ->  href="https://e.com/<sup class="cc-cite">[1]…
  //
  // which closes the attribute early and drops rel and target. Not an XSS —
  // only fixed strings ever reach tag context, and the attacker's text stays
  // escaped — but the header's claim about which attributes are emitted was
  // untrue in the DOM. A URL containing them is now left as plain text, which
  // is the safe direction: a link that does not render is visibly wrong, an
  // anchor with its rel silently stripped is not.
  out = out.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s*[\]]+)\)/g,
    '<a href="$2" rel="noopener nofollow" target="_blank">$1</a>'
  );
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // [n] citations (not link openers — those were consumed above).
  out = out.replace(/\[(\d{1,3})\](?!\()/g, '<sup class="cc-cite">[$1]</sup>');
  return out;
}

// Inline pass: protect `code` spans, format everything else.
function renderInline(text) {
  const escaped = escapeHtml(text);
  return escaped
    .split(/(`[^`\n]+`)/)
    .map((part) => {
      if (part.length > 2 && part.startsWith('`') && part.endsWith('`')) {
        return `<code>${part.slice(1, -1)}</code>`;
      }
      return renderSegment(part);
    })
    .join('');
}

function renderList(lines, ordered) {
  const marker = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*]\s+/;
  const items = lines.map((l) => `<li>${renderInline(l.replace(marker, ''))}</li>`).join('');
  return ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
}

/**
 * Render markdown `src` to an HTML string.
 * @param {string} src
 * @returns {string}
 */
export function renderMarkdown(src) {
  if (typeof src !== 'string' || src.length === 0) return '';
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const html = [];
  let block = [];

  const flushBlock = () => {
    if (block.length === 0) return;
    const isUl = block.every((l) => /^\s*[-*]\s+/.test(l));
    const isOl = block.every((l) => /^\s*\d+[.)]\s+/.test(l));
    if (isUl) html.push(renderList(block, false));
    else if (isOl) html.push(renderList(block, true));
    else html.push(`<p>${block.map(renderInline).join('<br>')}</p>`);
    block = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^```([\w+-]*)\s*$/);
    if (fence) {
      flushBlock();
      const lang = fence[1];
      const code = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      // (An unterminated fence swallows to EOF — matches common renderers.)
      const cls = lang ? ` class="language-${lang}"` : '';
      html.push(`<pre class="cc-fence"><code${cls}>${escapeHtml(code.join('\n'))}</code></pre>`);
      continue;
    }
    if (line.trim() === '') {
      flushBlock();
      continue;
    }
    block.push(line);
  }
  flushBlock();

  return html.join('');
}
