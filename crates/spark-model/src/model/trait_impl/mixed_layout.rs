// SPDX-License-Identifier: AGPL-3.0-only

//! Pure layout math for the fused mixed decode+prefill step (`decode_b.rs`).
//!
//! Kept as a separate, dependency-free module so the scratch-collision
//! invariant is unit-testable without a GPU.

/// Byte offset (from the `scratch` arena base) at which the mixed-step
/// prefill metadata (positions + slots) is parked.
///
/// TWO writers park MoE routing indices+weights at `scratch[0..)` during the
/// fused layer loop, and the offset must clear BOTH:
///
///   - the prefill sub-call: `proc_count` rows
///     (`nemotron_moe` prefill / `moe/forward*.rs`), and
///   - the batched decode sub-call: `padded_n` rows
///     (`nemotron_moe/decode_multi_seq.rs`), which runs FIRST each layer —
///     `layer.prefill` then READS positions/slots from this offset.
///
/// Sizing the offset by `proc_count` alone (the pre-fix layout) let the
/// batched-decode routing write clobber the prefill positions/slots whenever
/// `padded_n > proc_count` — e.g. 5 decodes padded to 8 co-scheduled with a
/// 5-token prompt chunk, or any long prompt's short remainder chunk. The
/// result was garbage RoPE positions plus routing bytes reinterpreted as
/// slot i64s: out-of-bounds KV-pool writes / cross-request KV contamination.
///
/// `max_top_k` must be the max across layers
/// (`ModelConfig::max_num_experts_per_tok`) — per-block schedules (Puzzle)
/// can exceed the scalar `num_experts_per_tok`. The scratch arena is sized
/// for this worst case (`spark_runtime::buffers::sizes`, `moe_scratch` uses
/// the same max), so the widened offset always fits.
#[inline]
pub(super) fn mixed_prefill_meta_offset(
    proc_count: usize,
    padded_n: usize,
    max_top_k: usize,
) -> usize {
    let routing_rows = proc_count.max(padded_n);
    let moe_scratch_bytes = 2 * routing_rows * max_top_k * 4;
    (moe_scratch_bytes + 7) & !7
}

#[cfg(test)]
mod tests {
    use super::mixed_prefill_meta_offset;

    /// Fail-without-fix regression test for the mixed-step scratch clobber:
    /// the batched-decode routing region `[0 .. 2*padded_n*top_k*4)` must
    /// never overlap the prefill meta interval starting at the returned
    /// offset, for every padded batch rung × chunk length × top_k combo.
    /// (The pre-fix layout — `align8(2*proc_count*top_k*4)` — fails this for
    /// every `padded_n > proc_count`.)
    #[test]
    fn routing_region_never_overlaps_prefill_meta() {
        // top_k values: minimum, common (Qwen/Nemotron 4-8), Puzzle
        // per-block outlier 22, kernel cap 32.
        for top_k in [1usize, 4, 6, 8, 22, 32] {
            for padded_n in 1..=128usize {
                for proc_count in 1..=2048usize {
                    let meta_offset = mixed_prefill_meta_offset(proc_count, padded_n, top_k);
                    let decode_routing_end = 2 * padded_n * top_k * 4;
                    assert!(
                        decode_routing_end <= meta_offset,
                        "batched-decode routing [0..{decode_routing_end}) overlaps prefill \
                         meta at {meta_offset} (padded_n={padded_n}, proc_count={proc_count}, \
                         top_k={top_k})"
                    );
                    let prefill_routing_end = 2 * proc_count * top_k * 4;
                    assert!(
                        prefill_routing_end <= meta_offset,
                        "prefill routing [0..{prefill_routing_end}) overlaps prefill meta at \
                         {meta_offset} (padded_n={padded_n}, proc_count={proc_count}, \
                         top_k={top_k})"
                    );
                }
            }
        }
    }

    /// The offset is 8-byte aligned (the slots array parked after positions
    /// relies on the base being aligned).
    #[test]
    fn offset_is_8_byte_aligned() {
        for top_k in 1..=32usize {
            for padded_n in 1..=128usize {
                assert_eq!(mixed_prefill_meta_offset(1, padded_n, top_k) % 8, 0);
                assert_eq!(mixed_prefill_meta_offset(padded_n, 1, top_k) % 8, 0);
            }
        }
    }
}
