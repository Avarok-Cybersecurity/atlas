// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher — per-request grammar-matching state.
//
// Port of `class GrammarMatcher` / `GrammarMatcher::Impl` from
// `cpp/grammar_matcher.cc` + `include/xgrammar/matcher.h`. This file
// holds the struct, construction, token/string acceptance, rollback,
// termination and reset. The bitmask fill and jump-forward search
// (the long methods) live in the sibling `fill.rs`.
//
// In the C++ the `Impl` *inherits* `EarleyParser`; the Rust port
// composes one instead (`parser` field) — the parser already exposes
// `advance`, `pop_last_states`, `push_one_state_to_check`,
// `is_completed`, `latest_scanable_states` and `reset`.
//
// Termination: the C++ tracks `stop_token_is_accepted_` on the parser
// base. The Rust `EarleyParser` clears its own such flag on every
// `pop_last_states`, which is *not* what the matcher wants (a rollback
// past a stop token un-terminates, but a rollback that does not reach
// the stop step must keep it). The matcher therefore owns its own
// `stop_token_accepted` flag, undone precisely by `rollback`.

use std::collections::{HashMap, VecDeque};

use crate::compiler::CompiledGrammar;
use crate::earley::{cache_key, EarleyParser, ParserState};
use crate::tokenizer::TokenizerInfo;

/// A stateful matcher that matches sampled tokens against a compiled
/// grammar — the core of grammar-guided generation.
///
/// One `GrammarMatcher` tracks one request's decoding state. Each
/// decode step Atlas calls [`Self::fill_next_token_bitmask`] to learn
/// which tokens are legal, samples one, then [`Self::accept_token`].
/// [`Self::rollback`] undoes accepted tokens (e.g. for speculative
/// decoding); [`Self::reset`] returns to the initial state.
#[derive(Debug)]
pub struct GrammarMatcher {
    /// The compiled grammar (shared, cheap to clone).
    compiled_grammar: CompiledGrammar,
    /// Cache-key index over the compiled adaptive token masks.
    ///
    /// The compiler keys `adaptive_token_mask` by a canonical
    /// `ParserState` (`rule_start_pos = -1`, `repeat_count = 0`), but
    /// the parser's *live* scanable states carry real positions. The
    /// C++ `adaptive_token_mask_cache` is hashed by `StateHashForCache`
    /// — which ignores exactly those fields ([`cache_key`]). This map
    /// reproduces that: cache-key tuple -> canonical state, so a live
    /// state resolves to its mask via [`Self::mask_for_state`].
    mask_index: HashMap<(i32, i32, i32, i32), ParserState>,
    /// The Earley parser driving byte-level matching.
    pub(super) parser: EarleyParser,
    /// Token ids that terminate generation. Either the override set or
    /// the tokenizer's detected stop tokens.
    stop_token_ids: Vec<i32>,
    /// When true the matcher completes (terminates) as soon as the
    /// root rule is matched, without requiring a stop token.
    terminate_without_stop_token: bool,
    /// True once a stop token has been accepted — the matcher is then
    /// terminated. Undone by [`Self::rollback`] past the stop step.
    stop_token_accepted: bool,
    /// Byte length consumed by each accepted token / string, in order.
    /// One entry per `accept_token`/`accept_string` call; the rollback
    /// unit. A stop token contributes a `0`-length entry.
    token_length_history: VecDeque<usize>,
}

impl GrammarMatcher {
    /// Construct a matcher from a [`CompiledGrammar`].
    ///
    /// * `override_stop_tokens` — when `Some`, replaces the tokenizer's
    ///   detected stop tokens; must be non-empty (panics otherwise,
    ///   matching the C++ `XGRAMMAR_CHECK`).
    /// * `terminate_without_stop_token` — terminate on root-rule
    ///   completion without needing a stop token.
    /// * `max_rollback_tokens` — accepted for API faithfulness; the C++
    ///   deprecated and ignores it (rollback is unbounded).
    ///
    /// Port of the `GrammarMatcher` constructor.
    #[must_use]
    pub fn new(
        compiled_grammar: CompiledGrammar,
        override_stop_tokens: Option<Vec<i32>>,
        terminate_without_stop_token: bool,
        max_rollback_tokens: i32,
    ) -> Self {
        let _ = max_rollback_tokens; // deprecated & unused, kept for ABI parity.
        if let Some(ref ovr) = override_stop_tokens {
            assert!(
                !ovr.is_empty(),
                "override_stop_tokens must not be empty"
            );
        }
        let stop_token_ids = override_stop_tokens
            .unwrap_or_else(|| compiled_grammar.tokenizer_info().stop_token_ids().to_vec());
        let parser = EarleyParser::from_grammar(std::sync::Arc::new(
            compiled_grammar.grammar().clone(),
        ));
        let mask_index = compiled_grammar
            .adaptive_token_mask()
            .keys()
            .map(|s| (cache_key(s), *s))
            .collect();
        Self {
            compiled_grammar,
            mask_index,
            parser,
            stop_token_ids,
            terminate_without_stop_token,
            stop_token_accepted: false,
            token_length_history: VecDeque::new(),
        }
    }

    /// Convenience constructor with default options (no stop-token
    /// override, requires a stop token, unbounded rollback).
    #[must_use]
    pub fn from_compiled_grammar(compiled_grammar: CompiledGrammar) -> Self {
        Self::new(compiled_grammar, None, false, -1)
    }

    /// The compiled grammar this matcher runs.
    #[must_use]
    pub fn compiled_grammar(&self) -> &CompiledGrammar {
        &self.compiled_grammar
    }

    /// The tokenizer info the grammar was compiled against.
    #[must_use]
    pub fn tokenizer_info(&self) -> &TokenizerInfo {
        self.compiled_grammar.tokenizer_info()
    }

    /// Resolve a *live* parser state to the canonical [`ParserState`]
    /// the compiler keyed its adaptive token mask under, via the
    /// cache-key index. Returns `None` when the compiler produced no
    /// mask for that state (e.g. an empty vocabulary).
    ///
    /// The caller looks the canonical state up in
    /// [`CompiledGrammar::mask_for_state`] to get the actual mask — a
    /// two-step lookup keeps the `&AdaptiveTokenMask` borrow tied to
    /// the (clonable) grammar rather than to `self`.
    pub(super) fn canonical_mask_state(&self, state: &ParserState) -> Option<ParserState> {
        self.mask_index.get(&cache_key(state)).copied()
    }

    /// The stop token ids that terminate this matcher.
    #[must_use]
    pub fn stop_token_ids(&self) -> &[i32] {
        &self.stop_token_ids
    }

    /// The maximum number of rollback tokens — always `-1` (unbounded),
    /// matching the deprecated C++ behavior.
    #[must_use]
    pub fn max_rollback_tokens(&self) -> i32 {
        -1
    }

    /// True if the matcher has terminated. When
    /// `terminate_without_stop_token` is set this is "root rule
    /// completed"; otherwise it is "a stop token was accepted".
    ///
    /// Port of `IsTerminated`.
    #[must_use]
    pub fn is_terminated(&self) -> bool {
        if self.terminate_without_stop_token {
            self.parser.is_completed()
        } else {
            self.stop_token_accepted
        }
    }

    /// True once an actual stop token has been accepted — distinct
    /// from [`Self::is_terminated`], which also fires on root-rule
    /// completion under `terminate_without_stop_token`. The bitmask
    /// fill and jump-forward search guard on this flag specifically.
    #[must_use]
    pub(super) fn stop_token_accepted(&self) -> bool {
        self.stop_token_accepted
    }

    /// True if the root rule has matched at the current position — a
    /// stop token may be emitted now. Exposed for callers that need the
    /// raw completion state independently of termination policy.
    #[must_use]
    pub fn is_grammar_completed(&self) -> bool {
        self.parser.is_completed()
    }

    /// Try to accept the stop token, terminating the matcher.
    /// Returns whether the stop token was acceptable. Port of
    /// `AcceptStopToken`.
    fn accept_stop_token(&mut self) -> bool {
        if self.terminate_without_stop_token {
            return false;
        }
        if !self.parser.is_completed() {
            return false;
        }
        debug_assert!(!self.stop_token_accepted);
        self.token_length_history.push_back(0);
        self.stop_token_accepted = true;
        true
    }

    /// Accept one sampled token and advance the matcher.
    ///
    /// Returns `false` (without mutating state) if the token is not
    /// grammar-legal, is out of range, is a special token, or the
    /// matcher has already terminated. A stop token terminates the
    /// matcher. Port of `AcceptToken`.
    pub fn accept_token(&mut self, token_id: i32, debug: bool) -> bool {
        let _ = debug;
        if self.stop_token_accepted {
            return false;
        }
        let vocab_size = self.tokenizer_info().vocab_size() as i32;
        if token_id < 0 || token_id >= vocab_size {
            return false;
        }
        if self.stop_token_ids.contains(&token_id) {
            return self.accept_stop_token();
        }
        if self
            .tokenizer_info()
            .special_token_ids()
            .contains(&token_id)
        {
            return false;
        }
        // Decode the token to bytes and feed them one by one.
        let token = self.tokenizer_info().decoded_vocab()[token_id as usize].clone();
        self.advance_bytes_as_step(&token)
    }

    /// Accept a literal string, advancing the matcher byte by byte.
    /// The whole string counts as one rollback step. Used to seed a
    /// matcher or in tests. Port of `AcceptString`.
    pub fn accept_string(&mut self, s: &str, debug: bool) -> bool {
        self.accept_bytes(s.as_bytes(), debug)
    }

    /// Accept a literal byte string — the byte-level form of
    /// [`Self::accept_string`] (token bytes are not always UTF-8).
    pub fn accept_bytes(&mut self, bytes: &[u8], debug: bool) -> bool {
        let _ = debug;
        if self.stop_token_accepted {
            return false;
        }
        self.advance_bytes_as_step(bytes)
    }

    /// Advance the parser over `bytes` as a single rollback step.
    ///
    /// On a rejected byte the bytes already consumed are rolled back so
    /// the matcher state is unchanged — exactly as the C++
    /// `AcceptToken` / `AcceptString` do. Returns whether all bytes
    /// were accepted; on success one history entry is recorded.
    fn advance_bytes_as_step(&mut self, bytes: &[u8]) -> bool {
        for (consumed, &byte) in bytes.iter().enumerate() {
            if !self.parser.advance(byte) {
                if consumed > 0 {
                    self.parser.pop_last_states(consumed);
                }
                return false;
            }
        }
        self.token_length_history.push_back(bytes.len());
        true
    }

    /// Roll back the matcher by `n` accepted tokens / strings.
    ///
    /// Each prior `accept_token`/`accept_string` is one unit; a stop
    /// token is a zero-byte unit whose rollback un-terminates the
    /// matcher. Panics if `n` exceeds the recorded history. Port of
    /// `Rollback`.
    pub fn rollback(&mut self, n: i32) {
        let n = n.max(0) as usize;
        assert!(
            n <= self.token_length_history.len(),
            "cannot roll back {n} tokens: only {} steps recorded",
            self.token_length_history.len()
        );
        for _ in 0..n {
            let steps = self
                .token_length_history
                .pop_back()
                .expect("history checked non-empty");
            if steps == 0 {
                // The stop-token step: un-terminate, no bytes consumed.
                self.stop_token_accepted = false;
            } else {
                self.parser.pop_last_states(steps);
            }
        }
    }

    /// Reset the matcher to its initial state.
    pub fn reset(&mut self) {
        self.parser.reset();
        self.stop_token_accepted = false;
        self.token_length_history.clear();
    }

    /// Number of accepted tokens / strings currently in the rollback
    /// history.
    #[must_use]
    pub fn num_history_steps(&self) -> usize {
        self.token_length_history.len()
    }
}
