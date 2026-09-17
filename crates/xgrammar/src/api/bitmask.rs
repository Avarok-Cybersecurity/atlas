// SPDX-License-Identifier: AGPL-3.0-only
//
// Bitmask free functions — W7 compatibility shim.
//
// The vendored `xgrammar-rs` exposed `allocate_token_bitmask`,
// `get_bitmask_shape` and `reset_token_bitmask` as free functions in
// its `matcher` module. Atlas's `grammar/state.rs` calls all three.
// The pure-Rust port keeps the same logic in `matcher::bitmask`
// (`bitmask_size`, `TokenBitmask`); these wrappers restore the exact
// vendored signatures so Atlas compiles unchanged.

use crate::matcher::bitmask_size;

/// Return the shape of the bitmask: `(batch_size, ceil(vocab_size/32))`.
///
/// Port of the vendored `matcher::get_bitmask_shape`.
pub fn get_bitmask_shape(batch_size: usize, vocab_size: usize) -> (usize, usize) {
    (batch_size, bitmask_size(vocab_size))
}

/// Allocate a token bitmask: an `i32` buffer of
/// `batch_size * ceil(vocab_size/32)` words, initialized to all-ones
/// (`-1`, i.e. every token allowed).
///
/// Port of the vendored `matcher::allocate_token_bitmask`.
pub fn allocate_token_bitmask(batch_size: usize, vocab_size: usize) -> Box<[i32]> {
    let (_, words) = get_bitmask_shape(batch_size, vocab_size);
    vec![-1i32; batch_size * words].into_boxed_slice()
}

/// Reset a bitmask buffer to the full (all-allowed) mask.
///
/// Port of the vendored `matcher::reset_token_bitmask`.
pub fn reset_token_bitmask(bitmask: &mut [i32]) {
    bitmask.fill(-1i32);
}
