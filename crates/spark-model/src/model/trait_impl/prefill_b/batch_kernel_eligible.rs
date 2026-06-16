// SPDX-License-Identifier: AGPL-3.0-only

//! Eligibility gating for the Q12 Path B kernel-batched prefill path.
//!
//! Split out of `batch_kernel.rs` to keep both files under the 500-LoC
//! file-size cap. The pure predicate [`check_kernel_batched_eligible`] is
//! unit-tested in the sibling `batch_kernel_tests.rs`.

#![allow(clippy::too_many_arguments)]

use super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
        )
    }
}

/// Pure-data predicate extracted from [`TransformerModel::kernel_batched_eligible`]
/// so the gating rules are unit-testable without a real `TransformerModel`.
/// Caller materialises per-stream tuples `(chunk_len, chunk_start, is_last_chunk)`.
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    model_type: &str,
    head_dim: usize,
) -> bool
where
    I: IntoIterator<Item = (usize, usize, bool)>,
{
    if n < 2 {
        return false;
    }
    // No MLA layers in stack (batched attention doesn't support MLA).
    // Conservatively check via model_type — mistral is the only MLA
    // model in Atlas today.
    if model_type == "mistral" {
        return false;
    }
    // No HDIM=512 layers (Gemma-4 long-attention).
    if head_dim > 256 {
        return false;
    }
    let mut first: Option<(usize, usize, bool)> = None;
    let mut total = 0usize;
    for (chunk_len, chunk_start, is_last) in streams {
        // `chunk_len`, `chunk_start`, and `is_last_chunk` must all
        // match across streams. Different `chunk_start` produces
        // different `effective_seq_len_start` post-Marconi (which the
        // batched attention kernel cannot handle); mixing
        // `is_last_chunk` can't dispatch one finalize_last and one
        // save_checkpoint in a single batched call.
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if chunk_len != cl || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        total += chunk_len;
    }
    // Total stacked tokens fit in arena.
    total <= arena_cap
}
