// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-propose width policy: how many sequences one drafter forward can
//! carry, derived (never assumed) from the resolved kernels and the arena
//! buffer capacities.
//!
//! The batched propose used to be hard-capped at 4 (`(2..=4).contains(&n)`),
//! so the scheduler ran it in groups of <= 4 — 4 drafter forwards per draft
//! position at n=16, each re-reading the whole BF16 drafter. That is the
//! second of the three eager costs the n=16 finalizer matrix named (the K=1
//! verify step measured ~1.9x a plain batch-16 decode step against a ~1.72x
//! break-even at p1~0.72).
//!
//! Nothing structural forced 4: the drafter's per-row loops are index-generic
//! and every weight-bearing GEMM is M-generic. Only three things were width-
//! bound, and all three are checked here rather than assumed:
//!
//! 1. the LM head ran on `w4a16_gemv_batch4` (MAX_M=4). `w4a16_gemv_batch8`
//!    and `w4a16_gemv_batch16` are the same template at wider MAX_M and were
//!    already compiled; [`MtpHead::lm_head_batch_kernel`] picks the narrowest
//!    resolved kernel that covers `n` (per-row accumulation order is
//!    identical across instantiations, so the output is bit-identical at
//!    matching M).
//! 2. the per-sequence drafter attention metadata lived at a FIXED offset
//!    inside the shared `scratch` buffer (`scratch + 49152 + i*2048`), which
//!    at n=16 runs 15 KB past the end of a 27B-shaped scratch allocation —
//!    a silent out-of-range H2D (the #110 failure mode: sticky CUDA-700).
//!    The drafter now owns a dedicated `propose_meta` allocation.
//! 3. the arena rows the forward writes. Most are sized `max_batch_tokens`
//!    rows of something at least as wide as what the drafter puts there, but
//!    `ssm_ba` (row = 2*num_value_heads floats) and `ssm_gates` (row =
//!    2*num_value_heads f32) are NOT — the drafter parks `[n, 2h]` and
//!    `[n, h]` BF16 there. Both are checked in bytes below.

use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::KernelHandle;

use atlas_core::config::ModelConfig;

use super::{MtpHead, MtpQuantization, ProjectionWeight};

/// Bytes of drafter attention metadata per sequence in `propose_meta`.
/// Layout per sequence i at `propose_meta + i*PROPOSE_META_STRIDE` mirrors
/// `forward_one`: [0..4) position u32 | [8..16) slot i64 | [16..20) seq_len
/// i32 | [256..) block table i32[]. 2048 bytes caps the block table at 448
/// entries (a 4096-token drafter at block size 16 needs 257).
pub(crate) const PROPOSE_META_STRIDE: usize = 2048;

/// Sequences the `propose_meta` allocation is sized for. Matches the batched
/// verify's 32-slot hidden stash (`VERIFY_WY_TABLE_SEQS`) — the widest chunk
/// the K-vs-batch ladder can hand a propose since the 32:1 rung. 32 x 2048 =
/// 64 KB.
pub(crate) const PROPOSE_META_SEQS: usize = 32;

impl MtpHead {
    /// Narrowest resolved `w4a16_gemv_batch{M}` kernel covering `n` rows, or
    /// a 0 handle when none is resolved (`try_kernel` misses are a silent 0 —
    /// gate on the handle, never assume the kernel is in this target's set).
    pub(crate) fn lm_head_batch_kernel(&self, n: usize) -> KernelHandle {
        for (max_m, k) in [
            (4usize, self.w4a16_gemv_batch4_k),
            (8, self.w4a16_gemv_batch8_k),
            (16, self.w4a16_gemv_batch16_k),
            (32, self.w4a16_gemv_batch32_k),
        ] {
            if n <= max_m && k.0 != 0 {
                return k;
            }
        }
        KernelHandle(0)
    }

    /// Whether the BF16-everything scope + non-width-dependent kernels the
    /// batched propose needs are all present. Width is [`Self::propose_batch_max`].
    fn propose_batch_scope_ok(&self) -> bool {
        let bf16_proj = |p: &ProjectionWeight| matches!(p, ProjectionWeight::Bf16(_));
        matches!(self.quant, MtpQuantization::Bf16)
            && self.kv_bf16
            && bf16_proj(&self.fc)
            && bf16_proj(&self.q_proj)
            && bf16_proj(&self.k_proj)
            && bf16_proj(&self.v_proj)
            && bf16_proj(&self.o_proj)
            && self
                .dense_ffn_generic
                .as_ref()
                .is_some_and(|(g, u, d)| bf16_proj(g) && bf16_proj(u) && bf16_proj(d))
            && self.dense_gemm_pipelined_k.0 != 0
            && self.dense_gemv_k.is_some()
            && self.deinterleave_qg_k.is_some()
            && self.moe_silu_mul_k.is_some()
            && !self.propose_meta.is_null()
    }

    /// The widest batched propose this head can run: `1` means "per-sequence
    /// only" (the caller must not batch). Every term is a measured capacity,
    /// not a constant someone hopes is big enough.
    pub(crate) fn propose_batch_max(&self, buffers: &BufferArena, config: &ModelConfig) -> usize {
        if !self.propose_batch_scope_ok() {
            return 1;
        }
        let h = config.hidden_size;
        let bf16 = 2usize;
        let sizes = buffers.sizes();
        // Rows each capacity-bound buffer can hold for THIS forward's use of
        // it. `ssm_ba` holds the [n, 2h] concat; `ssm_gates` the [n, h]
        // normed hidden. Everything else is arena-row-sized (>= h per row).
        let rows = |bytes: usize, per_row: usize| {
            if per_row == 0 { 0 } else { bytes / per_row }
        };
        let mut cap = PROPOSE_META_SEQS
            .min(rows(sizes.ssm_ba, 2 * h * bf16))
            .min(rows(sizes.ssm_gates, h * bf16))
            .min(rows(sizes.ssm_qkvz, h * bf16))
            .min(rows(sizes.ssm_deinterleaved, h * bf16))
            .min(rows(sizes.hidden_states, h * bf16))
            .min(rows(sizes.residual, h * bf16))
            .min(rows(sizes.norm_output, h * bf16))
            .min(buffers.max_batch_tokens());
        // The LM head kernels come in discrete widths; shrink to the widest
        // one that is actually resolved.
        while cap > 1 && self.lm_head_batch_kernel(cap).0 == 0 {
            cap -= 1;
        }
        cap.max(1)
    }

    /// Whether the batched cross-sequence propose can run for `n` sequences.
    /// SSOT: one call into [`Self::propose_batch_max`], no second copy of the
    /// width policy.
    pub(crate) fn can_propose_batch(
        &self,
        n: usize,
        buffers: &BufferArena,
        config: &ModelConfig,
    ) -> bool {
        n >= 2 && n <= self.propose_batch_max(buffers, config)
    }
}
