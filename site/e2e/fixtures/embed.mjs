// =============================================================================
// embed.mjs — deterministic fake embedder shared by the corpus generator and
// the mocked OpenRouter route handlers (SSOT: both sides MUST embed with the
// same function, so retrieval in tests behaves like a real semantic search).
// Not a real model: a seeded character-trigram hash bagged into `dim` buckets,
// L2-normalized. Same input -> same vector, on any runtime.
// =============================================================================

export function embedText(text, dim = 8) {
  const v = new Array(dim).fill(0);
  for (let i = 0; i + 2 < text.length; i++) {
    let h = 2166136261 >>> 0; // FNV-1a over a char trigram
    for (let j = i; j < i + 3; j++) {
      h = (h ^ text.charCodeAt(j)) >>> 0;
      h = Math.imul(h, 16777619) >>> 0;
    }
    v[h % dim] += ((h >>> 8) % 2000) / 1000 - 1;
  }
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => Number((x / norm).toFixed(6)));
}
