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

/// FLOOR for the per-sequence drafter-metadata stride in `propose_meta`.
/// Layout per sequence i at `propose_meta + i*propose_meta_stride` mirrors
/// `forward_one`: [0..4) position u32 | [8..16) slot i64 | [16..20) seq_len
/// i32 | [256..) block table i32[].
///
/// 2048 was the FIXED stride until PROGRESS_LOG 5.2: it caps the block table
/// at 448 entries = 7,168 tokens at block size 16 — sized in the 4K era.
/// Agentic contexts of 10-20K blew past it every step, making the batched
/// propose permanently fall back to per-sequence mode (PROGRESS_LOG 5.2/6.17).
/// The stride is now computed per head from `max_seq_len`
/// ([`propose_meta_stride_bytes`]); this constant survives only as the floor
/// so the layout can never shrink below what the 2048-era code assumed.
pub(crate) const PROPOSE_META_STRIDE_FLOOR: usize = 2048;

/// Bytes of the fixed header ahead of the block table in each meta slab.
pub(crate) const PROPOSE_META_HEADER: usize = 256;

/// Sequences the `propose_meta` allocation is sized for. Matches the batched
/// verify's 32-slot hidden stash (`VERIFY_WY_TABLE_SEQS`) — the widest chunk
/// the K-vs-batch ladder can hand a propose since the 32:1 rung.
/// 32 x stride bytes total (stride is dynamic, floor 2048).
pub(crate) const PROPOSE_META_SEQS: usize = 32;

/// Pure stride computation: header + one i32 block-table entry per KV block a
/// `max_seq_len`-token drafter sequence can reference, 8-byte aligned, never
/// below [`PROPOSE_META_STRIDE_FLOOR`] or above [`PROPOSE_META_STRIDE_CAP`].
///
/// The `+ 1` mirrors the allocator in `forward_batch_position`
/// (`blocks_needed = seq_len/bs + 1`): without it, a sequence sitting exactly
/// at `max_seq_len` needs one more entry than `align8(256 + (max/bs)*4)`
/// provides and the stride `ensure!` fires at the boundary.
pub(crate) fn propose_meta_stride_bytes(max_seq_len: usize, kv_block_size: usize) -> usize {
    let entries = (max_seq_len / kv_block_size.max(1)).saturating_add(1);
    let raw = PROPOSE_META_HEADER.saturating_add(entries.saturating_mul(4));
    let aligned = raw.saturating_add(7) & !7;
    aligned.clamp(PROPOSE_META_STRIDE_FLOOR, PROPOSE_META_STRIDE_CAP)
}

/// Ceiling for the env override. A 1M-token context needs ~262KB of block
/// table; 16 MiB is far beyond any real stride, and the cap keeps
/// `PROPOSE_META_SEQS * stride` from overflowing (release wrap would shrink
/// the alloc while `ensure!` still checks the huge stride — wild offsets).
pub(crate) const PROPOSE_META_STRIDE_CAP: usize = 1 << 24;

/// Stride actually used for a new head: [`propose_meta_stride_bytes`] unless
/// `ATLAS_PROPOSE_META_STRIDE=<bytes>` overrides it (kill switch / sizing
/// experiments). The override is value-parsed; garbage or an empty value is
/// ignored and falls through to the computed stride. Overrides are 8-byte
/// aligned up, floored at [`PROPOSE_META_STRIDE_FLOOR`] so a hostile value
/// cannot shrink the slab below the fixed header layout, and capped at
/// [`PROPOSE_META_STRIDE_CAP`] so the `16 x stride` allocation cannot
/// overflow.
pub(crate) fn propose_meta_stride_env(max_seq_len: usize, kv_block_size: usize) -> usize {
    let value = std::env::var("ATLAS_PROPOSE_META_STRIDE").ok();
    if let Some(stride) = parse_stride_override(value.as_deref()) {
        return stride;
    }
    propose_meta_stride_bytes(max_seq_len, kv_block_size)
}

fn parse_stride_override(value: Option<&str>) -> Option<usize> {
    value?
        .trim()
        .parse::<usize>()
        .ok()
        .map(clamp_stride_override)
}

/// Align-up + floor + cap for an override value; saturating so no input
/// (including `usize::MAX`) can panic under overflow checks or wrap the
/// downstream `PROPOSE_META_SEQS * stride` allocation in release.
pub(crate) fn clamp_stride_override(bytes: usize) -> usize {
    (bytes.saturating_add(7) & !7).clamp(PROPOSE_META_STRIDE_FLOOR, PROPOSE_META_STRIDE_CAP)
}

impl MtpHead {
    /// Narrowest resolved `w4a16_gemv_batch{M}` kernel covering `n` rows, or
    /// a 0 handle when none is resolved (`try_kernel` misses are a silent 0 —
    /// gate on the handle, never assume the kernel is in this target's set).
    pub(crate) fn lm_head_batch_kernel(&self, n: usize) -> KernelHandle {
        // Narrow family (4..8) first — same "narrowest resolved tier that
        // covers n" rule this loop implements, but SSOT'd in
        // `layers::w4a16_gemv_tiers` because five call sites share it.
        let narrow = self.w4a16_batchm.kernel(n as u32);
        if narrow.0 != 0 {
            return narrow;
        }
        for (max_m, k) in [
            (16usize, self.w4a16_gemv_batch16_k),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_meta_stride_floor_at_4k() {
        // 4K/16 = 256 blocks (+1) -> 256 + 257*4 = 1284 -> align8 = 1288,
        // below the 2048-era layout: the floor wins. This is exactly the
        // config the fixed stride was sized for.
        assert_eq!(
            propose_meta_stride_bytes(4096, 16),
            PROPOSE_META_STRIDE_FLOOR
        );
    }

    #[test]
    fn propose_meta_stride_16k() {
        // 16K/16 = 1024 (+1) -> 256 + 1025*4 = 4356 -> align8 = 4360. The
        // old 2048 cap (448 entries = 7,168 tokens) failed this every step.
        assert_eq!(propose_meta_stride_bytes(16 * 1024, 16), 4360);
    }

    #[test]
    fn propose_meta_stride_64k() {
        // 64K/16 = 4096 (+1) -> 256 + 4097*4 = 16644 -> align8 = 16648.
        assert_eq!(propose_meta_stride_bytes(64 * 1024, 16), 16648);
    }

    #[test]
    fn propose_meta_stride_never_below_floor() {
        // Tiny contexts (and a degenerate block size) must never shrink the
        // slab below what the fixed-2048 layout assumed.
        assert_eq!(propose_meta_stride_bytes(0, 16), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(
            propose_meta_stride_bytes(1024, 16),
            PROPOSE_META_STRIDE_FLOOR
        );
        // block_size 0 is clamped to 1 rather than dividing by zero.
        assert!(propose_meta_stride_bytes(1024, 0) >= PROPOSE_META_STRIDE_FLOOR);
    }

    #[test]
    fn propose_meta_stride_covers_boundary_and_alignment() {
        // The runtime allocates seq_len/bs + 1 block-table entries; the
        // stride must cover that at seq_len == max_seq_len, 8-byte aligned.
        for max in [
            4096usize,
            10 * 1024,
            16 * 1024,
            20 * 1024,
            64 * 1024,
            128 * 1024,
        ] {
            let s = propose_meta_stride_bytes(max, 16);
            assert_eq!(s % 8, 0, "stride {s} not 8-aligned for max={max}");
            let bt_len = (max / 16 + 1) * 4;
            assert!(
                PROPOSE_META_HEADER + bt_len <= s,
                "stride {s} cannot hold {bt_len}B block table at max={max}"
            );
        }
    }

    #[test]
    fn computed_stride_caps_extreme_contexts_without_overflow() {
        assert_eq!(
            propose_meta_stride_bytes(usize::MAX, 1),
            PROPOSE_META_STRIDE_CAP
        );
        assert!(
            PROPOSE_META_SEQS
                .checked_mul(propose_meta_stride_bytes(usize::MAX, 1))
                .is_some()
        );
    }

    #[test]
    fn stride_override_is_panic_and_overflow_safe() {
        // Hostile env values must neither panic under overflow checks
        // (usize::MAX + 7) nor produce a stride whose 16x allocation wraps
        // in release (2^60-class values shrank the slab while ensure! still
        // passed — wild device offsets).
        for hostile in [usize::MAX, 1usize << 60, (1usize << 60) + 2048] {
            let s = clamp_stride_override(hostile);
            assert_eq!(s, PROPOSE_META_STRIDE_CAP);
            assert!(PROPOSE_META_SEQS.checked_mul(s).is_some());
        }
        // Zero/small values clamp up to the floor; normal values align up.
        assert_eq!(clamp_stride_override(0), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2047), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2048), PROPOSE_META_STRIDE_FLOOR);
        assert_eq!(clamp_stride_override(2049), 2056);
        assert_eq!(clamp_stride_override(4361), 4368);
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP - 1),
            PROPOSE_META_STRIDE_CAP
        );
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP),
            PROPOSE_META_STRIDE_CAP
        );
        assert_eq!(
            clamp_stride_override(PROPOSE_META_STRIDE_CAP + 1),
            PROPOSE_META_STRIDE_CAP
        );
    }

    #[test]
    fn stride_override_parser_accepts_trimmed_bytes_and_rejects_invalid_values() {
        assert_eq!(parse_stride_override(Some(" 4361 ")), Some(4368));
        assert_eq!(
            parse_stride_override(Some("0")),
            Some(PROPOSE_META_STRIDE_FLOOR)
        );
        assert_eq!(parse_stride_override(None), None);
        assert_eq!(parse_stride_override(Some("")), None);
        assert_eq!(parse_stride_override(Some("not-bytes")), None);
        assert_eq!(parse_stride_override(Some("-1")), None);
    }
}
