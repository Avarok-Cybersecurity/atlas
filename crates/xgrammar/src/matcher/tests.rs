// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMatcher tests — adapted from xgrammar v0.1.32
// `tests/python/test_grammar_matcher_*.py` + `tests/cpp/`.
//
// The upstream tests load real HuggingFace tokenizers; here a small
// hand-built `TokenizerInfo` exercises the same matcher code paths
// (accept/reject, bitmask fill, rollback, termination, jump-forward,
// uncertain-token resolution). The fixtures live here; the cases are
// split into submodules so every file stays under the 250-line cap.

use crate::compiler::{CompiledGrammar, GrammarCompiler};
use crate::tokenizer::{TokenizerInfo, VocabType};

use super::bitmask::bitmask_size;
use super::{BatchGrammarMatcher, GrammarMatcher, TokenBitmask};

mod accept_tests;
mod bitmask_tests;
mod coalesce_tests;
mod completion_tests;
mod jump_rollback_tests;

/// A tiny RAW-vocab tokenizer covering the bytes / multi-char tokens
/// used by the test grammars. Index 0 (`</s>`) is the stop token; the
/// rest are content tokens. `<pad>` is an empty-decode special token.
pub(super) fn tok() -> TokenizerInfo {
    let vocab: Vec<String> = [
        "</s>", // 0  — stop token
        "a",    // 1
        "b",    // 2
        "c",    // 3
        "d",    // 4
        "ab",   // 5
        "abc",  // 6
        "bc",   // 7
        "x",    // 8
        "{",    // 9
        "}",    // 10
        "\"",   // 11
        ":",    // 12
        " ",    // 13
        "1",    // 14
        "yes",  // 15
        "no",   // 16
        "abcd", // 17
        "",     // 18 — empty -> special token
        "cd",   // 19
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // Stop token id 0 (`</s>`); explicit so detection is deterministic.
    TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![0]), false)
}

/// The stop token id of [`tok`].
pub(super) const STOP_ID: i32 = 0;

/// Compile an EBNF grammar against [`tok`].
pub(super) fn compile_ebnf(ebnf: &str) -> CompiledGrammar {
    let c = GrammarCompiler::new(tok(), 1, true, -1);
    c.compile_grammar_from_ebnf(ebnf, "root")
        .expect("grammar should compile")
}

/// A matcher for an EBNF grammar compiled against [`tok`].
pub(super) fn matcher(ebnf: &str) -> GrammarMatcher {
    GrammarMatcher::from_compiled_grammar(compile_ebnf(ebnf))
}

/// Resolve a content token's id by its decoded bytes.
pub(super) fn id_of(token: &str) -> i32 {
    tok()
        .decoded_vocab()
        .iter()
        .position(|t| t.as_slice() == token.as_bytes())
        .map(|p| p as i32)
        .unwrap_or_else(|| panic!("token {token:?} not in vocab"))
}

/// Fill a fresh bitmask for `m` and return the set of accepted token
/// ids. Panics if the matcher has terminated.
pub(super) fn accepted_ids(m: &mut GrammarMatcher) -> Vec<i32> {
    let vocab = m.tokenizer_info().vocab_size();
    let mut buf = vec![0i32; bitmask_size(vocab)];
    m.fill_next_token_bitmask(&mut buf, 0, false)
        .expect("fill should succeed");
    let mask = wrap(&buf, vocab);
    (0..vocab as i32)
        .filter(|&t| mask.is_set(t as usize))
        .collect()
}

/// Wrap a raw word buffer into a `TokenBitmask` for assertions.
pub(super) fn wrap(words: &[i32], vocab: usize) -> TokenBitmask {
    let mut b = TokenBitmask::new(vocab);
    b.as_words_mut()
        .copy_from_slice(&words[..bitmask_size(vocab)]);
    b
}

// --------------------------------------------------------------------
// Construction / option tests (kept in the parent file — small).
// --------------------------------------------------------------------

#[test]
fn construct_default_matcher() {
    let m = matcher("root ::= \"abc\"\n");
    assert!(!m.is_terminated());
    assert_eq!(m.max_rollback_tokens(), -1);
    assert_eq!(m.stop_token_ids(), &[STOP_ID]);
    assert_eq!(m.num_history_steps(), 0);
}

#[test]
fn override_stop_tokens_replaces_detected() {
    let cg = compile_ebnf("root ::= \"a\"\n");
    let m = GrammarMatcher::new(cg, Some(vec![3, 4]), false, -1);
    assert_eq!(m.stop_token_ids(), &[3, 4]);
}

#[test]
#[should_panic(expected = "override_stop_tokens must not be empty")]
fn empty_override_stop_tokens_panics() {
    let cg = compile_ebnf("root ::= \"a\"\n");
    let _ = GrammarMatcher::new(cg, Some(vec![]), false, -1);
}

#[test]
fn terminate_without_stop_token_terminates_on_completion() {
    let cg = compile_ebnf("root ::= \"a\"\n");
    let mut m = GrammarMatcher::new(cg, None, true, -1);
    assert!(!m.is_terminated());
    assert!(m.accept_token(id_of("a"), false));
    // Root rule complete and `terminate_without_stop_token` set.
    assert!(m.is_terminated());
}

#[test]
fn batch_constructor_thread_caps() {
    let auto = BatchGrammarMatcher::new(None);
    assert!(auto.max_threads() >= 1);
    let one = BatchGrammarMatcher::new(Some(1));
    assert_eq!(one.max_threads(), 1);
    let four = BatchGrammarMatcher::new(Some(4));
    assert_eq!(four.max_threads(), 4);
}

#[test]
#[should_panic(expected = "max_threads must be >= 1")]
fn batch_zero_threads_panics() {
    let _ = BatchGrammarMatcher::new(Some(0));
}

#[test]
fn bitmask_size_matches_cpp_formula() {
    // GetBitmaskSize == ceil(vocab / 32).
    assert_eq!(bitmask_size(0), 0);
    assert_eq!(bitmask_size(1), 1);
    assert_eq!(bitmask_size(32), 1);
    assert_eq!(bitmask_size(33), 2);
    assert_eq!(bitmask_size(64), 2);
    assert_eq!(bitmask_size(65), 3);
}

#[test]
fn reset_returns_to_initial_state() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert_eq!(m.num_history_steps(), 1);
    m.reset();
    assert_eq!(m.num_history_steps(), 0);
    assert!(!m.is_terminated());
    // After reset the matcher accepts "a" again.
    assert!(m.accept_token(id_of("a"), false));
}
