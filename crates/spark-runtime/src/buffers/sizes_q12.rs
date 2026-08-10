// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 kernel-batched prefill scratch sizing. Split from `sizes.rs` (500-LoC cap).

/// Streams assumed when provisioning scratch for the Q12 kernel-batched
/// prefill path. The per-stream metadata region scales with N, so the
/// scratch buffer must be sized for a realistic max concurrent batched
/// streams. Beyond this, `check_kernel_batched_eligible` falls the dispatch
/// back to the per-stream path (which respects the same arena cap), so this
/// bound only governs how often the fast path is available — never safety.
pub const Q12_SIZING_STREAMS: usize = 8;

/// Exact scratch footprint (bytes) of the Q12 kernel-batched prefill staging
/// for `n` streams of `chunk_len` tokens each. SSOT for both scratch sizing
/// (`BufferSizes::from_config`) and the pre-flight eligibility check
/// (`check_kernel_batched_eligible`), so the two can never disagree about
/// whether a batch fits. Mirrors the staging layout in `batch_kernel.rs`
/// (MoE topk area + N per-stream meta blocks) and `stage_batched.rs`
/// (stacked positions ×(3 if MRoPE) + slots + block/seq_len pointer arrays)
/// plus the per-SSM-layer `h_state_ptrs` JIT slot.
pub fn q12_batched_scratch_bytes(n: usize, chunk_len: usize, top_k: usize, mrope: bool) -> usize {
    let total = n * chunk_len;
    // MoE topk staging (indices+weights, both ×n streams), 64-byte aligned.
    let moe = ((total * top_k * 4 * 2) + 63) & !63;
    // Per-stream meta block — same formula as batch_kernel.rs.
    let per_stream_meta = ((chunk_len * 16) + 64).max(4096);
    // Stacked BatchedAttnMetadata (stage_batched.rs layout).
    let pos = (total * 4 + 7) & !7;
    let pos_streams = if mrope { 3 } else { 1 };
    let slot = (total * 8 + 7) & !7;
    let ptrs = ((n * std::mem::size_of::<u64>()) + 7) & !7;
    // VARLEN cu_seqlens [n+1] i32 prefix-sum, staged after the pointer arrays
    // (stage_batched.rs:122-126). This term scales with n, so omitting it made
    // the SSOT under-count grow with batch size: at n>=4 the h_state_ptrs JIT
    // slot overlapped a live per-stream pointer table → cross-stream KV/GDN
    // bleed in decode (n<=3 clean, absorbed by over-provisioning slack).
    let cu_seqlens = (((n + 1) * 4) + 7) & !7;
    let stage_meta = pos_streams * pos + slot + 2 * ptrs + cu_seqlens;
    // h_state_ptrs JIT slot consumed per SSM layer (N device pointers).
    let h_state_ptrs = n * std::mem::size_of::<u64>();
    moe + n * per_stream_meta + stage_meta + h_state_ptrs
}
