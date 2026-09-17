// =============================================================================
// rag.js — the retrieval pipeline: embed question -> vector search -> rerank
// -> code-aware context -> chat. Pure orchestration over openrouter.js (all
// network) and lattice.js (all wasm); no DOM, no state mutation.
// =============================================================================

import { getEmbedding, rerank, chat } from './openrouter.js';
import { searchVectors } from './lattice.js';
import { TOP_K, RERANK_MULTIPLIER } from './config.js';

// LatticeDB `score` is a cosine DISTANCE: 0.0 = identical, lower = better.
function relevancePct(score) {
  return Math.max(0, Math.min(100, (1 - score) * 100));
}

function sourceUrl(repo, commit, path, startLine, endLine) {
  return `https://github.com/${repo}/blob/${commit}/${path}#L${startLine}-L${endLine}`;
}

function contextBlock(n, { path, startLine, endLine, language, text }) {
  const fence = '```';
  return `[${n}] ${path} lines ${startLine}–${endLine}\n${fence}${language || ''}\n${text}\n${fence}`;
}

function systemPrompt(repo, commit, context) {
  return (
    `You are the code assistant for the Atlas inference engine codebase ` +
    `(GitHub repository ${repo}, commit ${commit}). Answer questions about this ` +
    `codebase using ONLY the numbered code context below.\n\n` +
    `Rules:\n` +
    `- Cite the context blocks you draw from inline with bracketed numbers, e.g. [1] or [2].\n` +
    `- If the context does not contain what is needed to answer, say plainly that the ` +
    `retrieved code does not cover it. Never invent file names, functions, or behavior.\n` +
    `- Be concise and technical. Prefer short code excerpts over prose when they answer better.\n\n` +
    `Context:\n${context}`
  );
}

/**
 * Answer `question` against the indexed corpus.
 *
 * @param {string} question
 * @param {Array<{role: string, content: string}>} history prior chat turns
 * @param {object} opts
 * @param {string} opts.apiKey       visitor's OpenRouter key
 * @param {object} opts.corpus      { commit, dim, repo, ... } from state
 * @param {(phase: 'retrieving'|'reranking'|'thinking') => void} [opts.onPhase]
 *   'writing' is not announced here — the caller flips to it from its onDelta
 *   when the first answer token arrives (the chat model reasons first).
 * @param {(delta: {reasoning?: string, content?: string}) => void} [opts.onDelta]
 *   streamed thinking/answer tokens, forwarded verbatim from openrouter.chat
 * @returns {Promise<{answer: string, sources: Array<object>}>}
 */
export async function askCodebase(question, history, { apiKey, corpus, onPhase, onDelta, chatModel }) {
  onPhase?.('retrieving');

  const vector = await getEmbedding(question, apiKey);
  if (!Array.isArray(vector) || vector.length !== corpus.dim) {
    throw new Error(
      `The embedding model returned ${vector?.length ?? 0} dimensions but this corpus ` +
        `was built with ${corpus.dim}. The site's embedding model and the published ` +
        `corpus are out of sync — the corpus needs a rebuild against the current model.`
    );
  }

  // Stage 1 — cheap recall: over-fetch candidates by cosine distance
  // (ascending: 0.0 = identical, lower = better).
  const candidates = searchVectors(vector, TOP_K * RERANK_MULTIPLIER)
    .map((r) => ({ id: Number(r.id), score: r.score, payload: r.payload ?? {} }))
    .filter((c) => typeof c.payload.text === 'string')
    .sort((a, b) => a.score - b.score);

  // Stage 2 — precise reordering: the cross-encoder picks the final TOP_K.
  // Nothing to reorder with <= 1 candidate: skip the rerank round-trip.
  let picked;
  if (candidates.length > 1) {
    onPhase?.('reranking');
    const ranked = await rerank(
      question,
      candidates.map((c) => c.payload.text),
      apiKey,
      TOP_K
    );
    // `index` is a position into the documents WE sent, but it arrives from the
    // rerank service, so it is not ours to trust. The `sources` map below reads
    // `.payload` off whatever comes back, so a bad index costs the whole answer
    // rather than the one result.
    //
    // Checked as an INDEX, not for truthiness. `candidates` is an Array, so a
    // string index reaches its properties: `"length"` returns a number and
    // `"map"` a function, both of which survive a `.filter(Boolean)` and then
    // throw on `.payload`. That was the first version of this guard, and it
    // reinstated the exact TypeError it was written to remove.
    picked = ranked
      .slice(0, TOP_K)
      .filter(
        ({ index }) => Number.isInteger(index) && index >= 0 && index < candidates.length
      )
      .map(({ index }) => candidates[index]);
  } else {
    picked = candidates.slice(0, TOP_K);
  }

  const sources = picked.map((c, i) => {
    const p = c.payload;
    const startLine = Number(p.start_line);
    const endLine = Number(p.end_line);
    return {
      n: i + 1,
      path: p.path,
      startLine,
      endLine,
      score: c.score,
      relevancePct: relevancePct(c.score),
      url: sourceUrl(corpus.repo, corpus.commit, p.path, startLine, endLine)
    };
  });

  const context =
    picked.length > 0
      ? picked
          .map((c, i) =>
            contextBlock(i + 1, {
              path: c.payload.path,
              startLine: Number(c.payload.start_line),
              endLine: Number(c.payload.end_line),
              language: c.payload.language,
              text: c.payload.text
            })
          )
          .join('\n\n')
      : 'No relevant code was found in the corpus for this question.';

  onPhase?.('thinking');
  const answer = await chat(
    [...history, { role: 'user', content: question }],
    systemPrompt(corpus.repo, corpus.commit, context),
    apiKey,
    { onDelta, model: chatModel }
  );

  return { answer, sources };
}
