// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask};

use super::engine::GrammarError;

// ── GrammarState ───────────────────────────────────────────────────────

/// How many of the costliest token-masks to pre-warm when a grammar
/// state is created (xgrammar Tier 2, overlapped mask generation).
///
/// `CompiledGrammar::compile_top_k_masks` ranks reachable scanable
/// parser states by first-character scan breadth and eagerly populates
/// the JIT mask cache for the top `k`. Called from [`GrammarState::new`]
/// — which runs during the prefill phase of a grammar-constrained
/// request, while the GPU is busy with the prompt — so the first decode
/// steps never pay a cold mask-computation stall.
///
/// `8` is a deliberately small constant: a tool-call structural-tag
/// grammar has only a handful of genuinely expensive states (the JSON
/// string / value scanners), and the call is `<= ranked.len()` work, so
/// over-provisioning `k` is harmless. The mask cache is per-grammar and
/// shared (`Arc`) across every request that reuses the cached
/// `CompiledGrammar`, so this warm-up amortizes across requests too.
const FORCED_TOKEN_TOP_K: usize = 8;

/// Per-request grammar matching state.
///
/// Wraps a [`GrammarMatcher`] with its own bitmask buffer. The bitmask
/// is reused across decode steps to avoid re-allocation.
pub struct GrammarState {
    matcher: GrammarMatcher,
    /// Bitmask buffer: `Box<[i32]>` of shape `(1, ceil(vocab_size / 32))`.
    bitmask_data: Box<[i32]>,
    vocab_size: usize,
    /// Model stop/EOS token IDs (e.g. `<|im_end|>`). These are control
    /// tokens that terminate generation, NOT part of the grammar's content
    /// language — [`Self::accept_token`] accepts them unconditionally rather
    /// than feeding them to the xgrammar matcher (which refuses a stop token
    /// in a non-accepting state, desyncing the NPDA and aborting the turn).
    /// Empty unless set via [`Self::with_stop_tokens`] — production wires the
    /// request's `eos_tokens` in; unit tests that exercise only grammar
    /// structure leave it empty.
    stop_tokens: Box<[u32]>,
}

impl GrammarState {
    /// Create a new per-request grammar state from a compiled grammar.
    ///
    /// `vocab_size` must match the tokenizer vocabulary used during compilation.
    ///
    /// As part of construction this pre-warms the [`FORCED_TOKEN_TOP_K`]
    /// costliest token-masks via [`CompiledGrammar::compile_top_k_masks`]
    /// (xgrammar Tier 2). Construction happens during prefill, so the
    /// warm-up overlaps the prompt forward pass and the first decode
    /// steps never stall on a cold mask computation. The warm-up only
    /// populates a cache — it cannot change matcher behavior — so it is
    /// safe unconditionally.
    pub fn new(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        let matcher = GrammarMatcher::new(
            compiled, None,  // use stop tokens from compiled grammar
            false, // require stop token for proper termination
            -1,    // unlimited rollback
        )
        .map_err(GrammarError::Compilation)?;

        // Tier 2 (overlapped mask generation): eagerly compute the
        // costliest masks so they are warm before the first decode
        // step. Pure cache population — no behavioral effect.
        let warmed = compiled.compile_top_k_masks(FORCED_TOKEN_TOP_K);
        tracing::debug!(
            warmed,
            requested = FORCED_TOKEN_TOP_K,
            "Grammar: pre-warmed top-k token masks during prefill"
        );

        let bitmask_data = allocate_token_bitmask(1, vocab_size);

        Ok(Self {
            matcher,
            bitmask_data,
            vocab_size,
            stop_tokens: Box::new([]),
        })
    }

    /// Register the model's stop/EOS token IDs so [`Self::accept_token`]
    /// exempts them from grammar refusal. See the `stop_tokens` field doc.
    ///
    /// Builder form so the existing two-arg [`Self::new`] (used widely by
    /// unit tests of pure grammar structure) is unchanged; the single
    /// production creation site chains this with the request's eos tokens.
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: &[u32]) -> Self {
        self.stop_tokens = stop_tokens.to_vec().into_boxed_slice();
        self
    }

    /// Fill the allowed-token bitmask for the next decode step.
    ///
    /// Returns `true` if the bitmask constrains at least one token (i.e., is not
    /// all-ones). When `false`, the caller can skip bitmask application.
    ///
    /// Optimized for structural-tag grammars: in preamble state (before trigger),
    /// fill_bitmask() is called every 4 tokens instead of every token, saving
    /// fill_bitmask MUST be called every token to keep the xgrammar NPDA
    /// stacks synchronized with accept_token(). Skipping calls desynchronizes
    /// the FSM and causes fill_next_token_bitmask to hang (~47 tokens in).
    pub fn fill_bitmask(&mut self) -> bool {
        // Guard: calling fill_next_token_bitmask after the matcher has accepted
        // its stop token throws xgrammar::LogFatalError, which std::terminate()s
        // the whole process. Return false so callers skip bitmask application —
        // the grammar is already satisfied and imposes no further constraint.
        if self.matcher.is_terminated() {
            return false;
        }
        reset_token_bitmask(&mut self.bitmask_data);
        self.matcher
            .fill_next_token_bitmask(&mut self.bitmask_data, 0, false)
    }

    /// Raw bitmask data: `ceil(vocab_size / 32)` i32 words.
    ///
    /// Bit `token_id` is at `data[token_id / 32] & (1 << (token_id % 32))`.
    /// A set bit means the token is allowed.
    pub fn bitmask_data(&self) -> &[i32] {
        &self.bitmask_data
    }

    /// Check if a specific token is allowed by the current bitmask.
    pub fn is_token_allowed(&self, token_id: u32) -> bool {
        let word = (token_id / 32) as usize;
        let bit = token_id % 32;
        if word >= self.bitmask_data.len() {
            return false;
        }
        (self.bitmask_data[word] & (1i32 << bit)) != 0
    }

    /// Accept a sampled token and advance the grammar state.
    ///
    /// Returns `true` if the token was accepted by the grammar.
    /// Returns `false` if the token violates the grammar (should not happen
    /// if the bitmask was applied correctly).
    ///
    /// Short-circuits with `true` once the matcher has reached its
    /// terminated (accepting) state — feeding tokens past the stop into
    /// xgrammar emits a `grammar_matcher.cc:493` warning ("matcher has
    /// terminated, but is trying to accept new token") for every trailing
    /// token in spec-decode draft runs (Discord 2026-05-08 universe06608).
    /// Returning `true` keeps the spec-decode boundary heuristic in
    /// `truncate_drafts_at_grammar_boundary` consistent — drafts past a
    /// completed grammar are not "rejected by grammar"; they are simply
    /// past the stop, which the EOS handler will terminate independently.
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        // Stop/EOS tokens (e.g. `<|im_end|>`, 248046) are control tokens that
        // terminate the turn — they are never part of the grammar's content
        // language. The model legitimately ends a turn with one (after a
        // text-only reply, or after a complete tool call) while the
        // structural-tag NPDA is still in a non-accepting state; feeding it to
        // `matcher.accept_token` then returns false ("refusal"), which callers
        // mis-read as a grammar desync and abort the response mid-stream
        // (`emit_step`: "Ending response to prevent cascading grammar-mask
        // corruption"). That truncates agentic turns and is the dominant cause
        // of the opencode webserver_ok gap vs vLLM (which parses tools
        // post-hoc and never constrains the stop token). Accept it
        // unconditionally and let the EOS handler terminate. This cannot
        // corrupt tool-call STRUCTURE — only non-stop tokens drive the matcher.
        if self.stop_tokens.contains(&token_id) {
            return true;
        }
        self.matcher.accept_token(token_id as i32)
    }

    /// The single grammar-forced next token, if the current state
    /// admits exactly one legal token (xgrammar Tier 3b, Coalescence).
    ///
    /// Returns `Some(token_id)` when the constrained grammar leaves no
    /// choice — the token is fully determined, so the model sampling
    /// step (and the full vocab-wide mask fill) for this position can be
    /// skipped and `token_id` emitted directly. Returns `None` when the
    /// continuation is a genuine choice, the state is dead, or the
    /// matcher has terminated.
    ///
    /// CORRECTNESS: `forced_token` computes the same authoritative
    /// next-token bitmask [`Self::fill_bitmask`] would and reports a
    /// token only when it is the *sole* set bit — so it is, by
    /// construction, the only grammar-legal token. The normal path could
    /// only ever have sampled that exact token (every other token is
    /// masked to `-inf`). The matcher state is left unchanged: the caller
    /// must still feed the returned token back through
    /// [`Self::accept_token`], exactly as for a sampled token.
    ///
    /// Returns `None` once the matcher has terminated (no further
    /// constraint) — symmetric with [`Self::fill_bitmask`]'s guard.
    pub fn forced_token(&mut self) -> Option<i32> {
        if self.matcher.is_terminated() {
            return None;
        }
        self.matcher.forced_token()
    }

    /// Whether the grammar has been fully matched (all required structure generated).
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    /// Whether a model stop/EOS token is grammar-legal at the current
    /// position — i.e. the response may end *here* with parseable output.
    ///
    /// Used by budget-aware graceful close (#144) to decide whether a close
    /// is needed: a length-truncated structured-output response whose stop
    /// token is *not* legal here ends mid-structure (e.g. inside an open
    /// JSON string) with unparseable output. Returns `true` (no close
    /// needed) once terminated or when the grammar imposes no constraint.
    pub fn stop_legal(&mut self, eos_tokens: &[u32]) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        // `fill_bitmask` returning false means the grammar is all-accepting
        // here, so stopping is legal. When it constrains, the stop token is
        // present in the mask iff the root rule can complete now.
        if !self.fill_bitmask() {
            return true;
        }
        eos_tokens.iter().any(|&e| self.is_token_allowed(e))
    }

    /// The shortest grammar-legal close as content token ids (#144), or
    /// `None` if no close is found within `max_bytes`. `Some(empty)` means
    /// the grammar can already stop. Leaves matcher state unchanged.
    pub fn completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        self.matcher.find_completion_token_ids(max_bytes)
    }

    /// Number of actual matcher history steps (== tokens `rollback` can undo).
    ///
    /// BUG#3 (2026-06-02): `accept_token` returns `true` for stop/EOS tokens and
    /// in the terminated state WITHOUT advancing the matcher (no history step).
    /// Spec/verify rollback accounting must therefore count actual advances via
    /// the delta of this value across a draft span — NOT the number of
    /// `accept_token`→true calls — or it over-rewinds (corrupt state / rollback
    /// panic) when a stop or terminated token lands inside the span.
    pub fn num_history_steps(&self) -> usize {
        self.matcher.num_history_steps()
    }

    /// Rollback the grammar state by `n` tokens.
    ///
    /// Used for MTP speculative decode: when draft tokens are rejected,
    /// the grammar state must be rewound to match.
    pub fn rollback(&mut self, n: usize) {
        self.matcher.rollback(n as i32);
    }

    /// Reset the grammar state to the initial position.
    pub fn reset(&mut self) {
        self.matcher.reset();
    }

    /// Apply the current bitmask to a slice of f32 logits in-place.
    ///
    /// Masked tokens (disallowed by grammar) are set to `f32::NEG_INFINITY`.
    /// This is the CPU-side application; for GPU-side, a CUDA kernel would
    /// be needed (future optimization).
    pub fn apply_bitmask_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        for token_id in 0..n {
            let word = token_id / 32;
            let bit = token_id % 32;
            if word < self.bitmask_data.len() && (self.bitmask_data[word] & (1i32 << bit)) == 0 {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Vocabulary extraction helper ───────────────────────────────────────

// F72 helpers (`decoded_vocab_bytes`, `compute_trigger_breakers`)
// were removed in F73 / fix42. The byte-level partial-trigger anchor
// at the sampler level hung the server in production despite passing
// isolated tests. The xgrammar non-anchored TagDispatch limitation is
// now handled at the streaming-sanitizer + parser layer (envelope_open
// markers in `LeakMarkers`, plus `<minimax:_call>` → `<tool_call>`
// normalisation in `parse_tool_calls`).
