// SPDX-License-Identifier: AGPL-3.0-only

//! Sub-block prefix matching is OPT-IN.
//!
//! The two sub-block arms in `inner::walk` let a lookup match FEWER tokens
//! than a cached block holds, by reusing a block whose KV was computed for a
//! LONGER key. The reused tail is the previous turn's generated tokens, and
//! the new sequence writes its own tokens over them inside a block the radix
//! tree still shares with the original key.
//!
//! Measured on gx10-9959 (qwen3.8-flash-next EXL3 2.05bpw, 490-token prompt,
//! `--enable-prefix-caching`, temp 0, `warmrepro.py`): with the arms ON, six
//! identical requests returned three distinct completions; with them OFF, six
//! returned one, equal to the cold answer. Hence: off by default.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::{RadixTree, subblock_matching_from_value};

/// The regression. A warm turn repeating only the prompt must NOT be handed
/// the block that also holds the previous turn's generation.
#[test]
fn a_default_lookup_never_reaches_into_a_longer_cached_block() {
    // Turn 1 cached 31 tokens: one full block (10) plus a 15-token partial
    // block (20) whose tail is generation.
    let tokens_31: Vec<u32> = (0..31).collect();
    let tokens_22: Vec<u32> = (0..22).collect();

    // Construct the DEFAULT explicitly rather than through `new()`, which
    // reads `ATLAS_PREFIX_SUBBLOCK` from the ambient process env: a test that
    // asserts the default must not flip when it runs under the very A/B this
    // change documents (it did — `ATLAS_PREFIX_SUBBLOCK=1` in the shell made
    // this fail).
    let tree = RadixTree::with_subblock_matching(false);
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);

    // Turn 2 repeats the 22-token prompt. Block-aligned by default.
    let m = tree.lookup(&tokens_22, 16, 0, 0);
    assert_eq!(m.matched_tokens, 16, "match must be block-aligned");
    assert_eq!(
        m.matched_blocks,
        vec![10],
        "block 20 holds turn 1's generation past token 22 and must not be reused"
    );
    assert!(m.matched_tokens.is_multiple_of(16));
    tree.release(&tokens_22, 16, 0);
}

/// The opt-in still works, and it is what produced the defect — so this test
/// also pins exactly what the default is protecting against.
#[test]
fn the_opt_in_restores_the_reach_into_the_longer_block() {
    let tokens_31: Vec<u32> = (0..31).collect();
    let tokens_22: Vec<u32> = (0..22).collect();

    let tree = RadixTree::with_subblock_matching(true);
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);

    let m = tree.lookup(&tokens_22, 16, 0, 0);
    assert_eq!(m.matched_tokens, 22, "sub-block arm reaches past the block");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    assert!(!m.matched_tokens.is_multiple_of(16));
    tree.release(&tokens_22, 16, 0);
}

/// `peek_matched_tokens` shares the walk, so it must share the polarity —
/// otherwise the scheduler's admission estimate and the actual lookup would
/// disagree about how much of a prompt is cached.
#[test]
fn peek_agrees_with_lookup_under_both_settings() {
    let tokens_31: Vec<u32> = (0..31).collect();
    let tokens_22: Vec<u32> = (0..22).collect();

    for subblock in [false, true] {
        let tree = RadixTree::with_subblock_matching(subblock);
        tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);
        let peeked = tree.peek_matched_tokens(&tokens_22, 16, 0);
        let m = tree.lookup(&tokens_22, 16, 0, 0);
        assert_eq!(peeked, m.matched_tokens, "subblock={subblock}");
        tree.release(&tokens_22, 16, 0);
    }
}

/// Exact opt-in, like every other kill switch: only "1".
#[test]
fn the_kill_switch_is_an_exact_opt_in() {
    assert!(!subblock_matching_from_value(None));
    assert!(!subblock_matching_from_value(Some("0")));
    assert!(!subblock_matching_from_value(Some("true")));
    assert!(subblock_matching_from_value(Some("1")));
}
