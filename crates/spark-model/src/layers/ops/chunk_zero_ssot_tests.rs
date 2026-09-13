// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT pin for the CHUNK-ZERO batched-prefill admission predicate.
//!
//! The predicate lives in exactly one function,
//! [`super::prefill_batched_chunk_zero_allowed`]. Every other site that needs
//! "may a wave of fresh prompts co-admit into one forward?" calls it.
//!
//! Why a source-scanning test rather than a behavioural one: the predicate
//! reads process-global `OnceLock`s seeded from the environment, so a
//! behavioural test can only observe whatever the first reader in the test
//! binary happened to resolve. What actually broke (#927 cell E) was not the
//! predicate's value — it was one of four readers computing its OWN disjunction
//! and getting a different answer: admission said yes to a `--prefill-varlen-batch`
//! chunk-0 wave, `Qwen3AttentionLayer::prefill_inner` said no, and the refusal
//! landed mid-forward with KV blocks already allocated, failing all sixteen
//! requests in the wave. The defect was a duplicated expression, so the test
//! is on the expressions.

/// Files that decide chunk-zero admission, and the call each must make.
const READERS: &[(&str, &str)] = &[
    (
        "eligible.rs (admission)",
        include_str!("../../model/trait_impl/prefill_b/batch_kernel/eligible.rs"),
    ),
    (
        "batch_kernel.rs (paged upload)",
        include_str!("../../model/trait_impl/prefill_b/batch_kernel.rs"),
    ),
    (
        "prefill_inner.rs (attention layer body)",
        include_str!("../qwen3_attention/trait_impl/prefill_inner.rs"),
    ),
    (
        "paged_attn_batched.rs (batched attention entry)",
        include_str!("../qwen3_attention/prefill/paged_attn_batched.rs"),
    ),
];

/// The shape that must never come back: a reader OR-ing the two levers itself.
fn rederives_the_predicate(src: &str) -> bool {
    // Strip comment lines so this file's own narration (and the fix's SSOT
    // comments, which name the old expression) does not trip the scan.
    src.lines()
        .map(str::trim_start)
        .filter(|l| !l.starts_with("//"))
        .any(|l| {
            l.contains("prefill_batched_first_chunk_enabled")
                || (l.contains("prefill_varlen_enabled") && l.contains("chunk"))
        })
}

#[test]
fn every_chunk_zero_reader_calls_the_one_predicate() {
    for (name, src) in READERS {
        assert!(
            src.contains("prefill_batched_chunk_zero_allowed"),
            "{name} decides chunk-zero admission but does not call \
             ops::prefill_batched_chunk_zero_allowed()",
        );
    }
}

#[test]
fn no_chunk_zero_reader_rederives_the_disjunction() {
    for (name, src) in READERS {
        assert!(
            !rederives_the_predicate(src),
            "{name} re-derives the chunk-zero predicate instead of calling \
             ops::prefill_batched_chunk_zero_allowed() — that divergence is \
             exactly the #927 cell-E failure",
        );
    }
}

#[test]
fn the_predicate_is_defined_once() {
    let helpers = include_str!("dispatch_helpers.rs");
    assert_eq!(
        helpers
            .matches("pub fn prefill_batched_chunk_zero_allowed")
            .count(),
        1,
        "the chunk-zero predicate must have exactly one definition",
    );
}
