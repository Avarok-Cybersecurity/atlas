// SPDX-License-Identifier: AGPL-3.0-only

//! Paged-decode attention SPLIT-K: how many splits, and which pair of kernels
//! runs them (#928).
//!
//! Split out of `run_paged_decode.rs` — which is already on the file-size
//! cap's allow list — because this is one question with one answer and three
//! call sites (the NVFP4 arm, the FP8 arm and the BF16 arm) that used to spell
//! it out twice and refuse it once.
//!
//! # What was wrong
//!
//! ```text
//! use atlas_core::device::sm121::NUM_SMS;            // 48 — the GB10 constant
//! let current_ctas = num_q_heads * split_ref_seqs(num_seqs, max_decode_seqs);
//! let num_splits = if current_ctas >= NUM_SMS { 1 } else { NUM_SMS / current_ctas };
//! ```
//!
//! On an H100 with `--max-batch-size 16` that is `24 * 16 = 384 >= 48` at
//! EVERY batch size, so `num_splits` was always 1 and `paged_decode_attn_fp8`
//! ran `grid=(24,1,1)` — 24 CTAs on 132 SMs, 231.51 us/launch for 9.93 MB of
//! KV = 42.9 GB/s = 1.28% of HBM, 22.7% of a C=1 decode step with its BF16
//! sibling (nsys, round 13 cell T1N; `ATTN-DECODE-SPLITK-ATTRIBUTION.md`).
//!
//! Both halves are fixed in [`atlas_kernels::attn_splitk`]: the SM count is
//! the compiled target's (`[hardware] sm_count`), and the `auto` policy sizes
//! for the SINGLE-STREAM shape, which is the one that starves. The policy is a
//! pure function of configuration, so the non-associative split-merge sees a
//! reduction tree that is FIXED for the life of a serve — the invariant
//! `split_ref_seqs` was added to protect
//! (`tasks/determinism_investigation.md`), preserved rather than relaxed.

use anyhow::Result;
use atlas_kernels::attn_splitk;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::Qwen3AttentionLayer;
use crate::layers::ops;

/// Everything a split-K launch needs that is NOT dtype-specific.
#[derive(Clone, Copy)]
pub(super) struct SplitkPlan {
    pub num_splits: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_seqs: u32,
    pub inv_sqrt_d: f32,
    pub q_stride: u32,
    pub sliding_window: u32,
}

/// The head dim every split-K kernel in the tree is COMPILED for.
///
/// `paged_decode_attn_fp8.cu`, `paged_decode_attn_nvfp4.cu` and the two Hopper
/// twins all take `HDIM` from a `#define` that defaults to 256 and derive each
/// lane's element count from it; none of them reads the `head_dim` argument
/// for anything but pointer arithmetic. A 128-wide head would have every lane
/// stride past its own row.
pub(super) const SPLITK_HEAD_DIM: u32 = 256;

/// The split count for this launch.
///
/// `num_seqs` is passed because the [`SplitkPolicy::Legacy`] arm — every
/// target but Hopper — is the pre-#928 rule preserved verbatim, and that rule
/// reads `split_ref_seqs(num_seqs, max_decode_seqs)`. The `Auto` and `Pinned`
/// arms do not read it at all, which is what makes them co-batch invariant;
/// `attn_splitk`'s
/// `the_auto_split_count_does_not_move_with_the_co_batched_count` is the test.
///
/// ⚠️ A non-256 head dim resolves to ONE split under any policy but `Legacy`.
/// The kernels' `HDIM` assumption (see [`SPLITK_HEAD_DIM`]) predates this
/// change and `Legacy` has been serving with it, so it keeps whatever answer
/// it has always given on that target; a policy that ASKS for more splits is
/// not the place to discover a kernel cannot address the head. Qwen3.8-27B is
/// `head_dim = 256`, so Hopper's `auto` is unaffected.
pub(super) fn num_splits(
    num_q_heads: u32,
    head_dim: u32,
    num_seqs: u32,
    max_decode_seqs: u32,
) -> u32 {
    let policy = ops::target_defaults::resolved().attn_decode_splitk.value;
    if policy != attn_splitk::SplitkPolicy::Legacy && head_dim != SPLITK_HEAD_DIM {
        return 1;
    }
    attn_splitk::num_splits(
        policy,
        atlas_kernels::TARGET_SM_COUNT,
        num_q_heads,
        super::super::split_ref_seqs(num_seqs, max_decode_seqs),
    )
}

/// `ATLAS_ATTN_DBG`: the split structure this layer resolved, printed once per
/// layer per step when a batch is co-batched.
///
/// Kept from the pre-#928 probe, and now able to answer the question it was
/// asked: under `auto` the printed `num_splits` must be identical at
/// `num_seqs=1` and `num_seqs=16`, which is the determinism claim.
pub(super) fn trace_splits(layer_idx: usize, num_seqs: u32, num_q_heads: u32, num_splits: u32) {
    if num_seqs != 1 && std::env::var("ATLAS_ATTN_DBG").is_ok() {
        tracing::debug!(
            "ATTN_DBG L{layer_idx} num_seqs={num_seqs} num_splits={num_splits} \
             (policy={} sm_count={} nq={num_q_heads})",
            ops::target_defaults::resolved()
                .attn_decode_splitk
                .value
                .label(),
            atlas_kernels::TARGET_SM_COUNT,
        );
    }
}

/// Which FP8 split-K pair to launch.
///
/// The Hopper twins (`kernels/hopper/common/paged_decode_fp8_splitk_hopper.cu`)
/// when this build carries them, the gb10 pair otherwise. Presence, not a
/// lever: the twins are the SAME algorithm with the non-split kernel's PD_BC=4
/// batched inner loop restored, so on a target that has them there is no
/// configuration under which the scalar gb10 loop is the better arm — the
/// lever that decides whether split-K runs at all is
/// `[defaults] attn_decode_splitk`, one rung up.
pub(super) struct SplitkPair {
    pub splitk: KernelHandle,
    pub reduce: KernelHandle,
}

impl Qwen3AttentionLayer {
    pub(super) fn fp8_splitk_pair(&self, head_dim: u32) -> Option<SplitkPair> {
        if head_dim == SPLITK_HEAD_DIM
            && let (Some(splitk), Some(reduce)) = (
                self.paged_decode_splitk_hopper_k,
                self.paged_decode_reduce_hopper_k,
            )
        {
            return Some(SplitkPair { splitk, reduce });
        }
        Some(SplitkPair {
            splitk: self.paged_decode_splitk_k?,
            reduce: self.paged_decode_reduce_k?,
        })
    }

    /// The BF16-KV split-K pair. Hopper-only: `run_paged_decode.rs` carried an
    /// explicit "no Split-K (not implemented for BF16 yet)" branch, so on every
    /// other target this is `None` and the caller runs the single-CTA kernel,
    /// exactly as before. `None` too for a wide head — Gemma-4's
    /// `head_dim > 256` full-attention layers keep `paged_decode_512_k`, which
    /// the twin is not compiled to replace.
    pub(super) fn bf16_splitk_pair(&self, head_dim: u32) -> Option<SplitkPair> {
        if head_dim != SPLITK_HEAD_DIM {
            return None;
        }
        Some(SplitkPair {
            splitk: self.paged_decode_splitk_bf16_hopper_k?,
            reduce: self.paged_decode_reduce_bf16_hopper_k?,
        })
    }

    /// FP8 split-K + reduce.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_splitk_fp8(
        &self,
        gpu: &dyn GpuBackend,
        pair: &SplitkPair,
        plan: SplitkPlan,
        q: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        workspace: DevicePtr,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        k_scale: f32,
        v_scale: f32,
        cache_stride: u64,
        stream: u64,
    ) -> Result<()> {
        ops::paged_decode_attn_splitk_fp8(
            gpu,
            pair.splitk,
            q,
            k_pool,
            v_pool,
            workspace,
            block_table,
            seq_lens,
            plan.max_blocks_per_seq,
            plan.num_q_heads,
            plan.num_kv_heads,
            plan.head_dim,
            plan.block_size,
            plan.inv_sqrt_d,
            plan.num_splits,
            k_scale,
            v_scale,
            plan.q_stride,
            cache_stride,
            plan.num_seqs,
            plan.sliding_window,
            stream,
        )?;
        ops::paged_decode_attn_reduce_fp8(
            gpu,
            pair.reduce,
            workspace,
            output,
            seq_lens,
            plan.num_q_heads,
            plan.head_dim,
            plan.num_splits,
            plan.num_seqs,
            stream,
        )
    }

    /// BF16 split-K + reduce.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_splitk_bf16(
        &self,
        gpu: &dyn GpuBackend,
        pair: &SplitkPair,
        plan: SplitkPlan,
        q: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        workspace: DevicePtr,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::paged_decode_attn_splitk_bf16(
            gpu,
            pair.splitk,
            q,
            k_pool,
            v_pool,
            workspace,
            block_table,
            seq_lens,
            plan.max_blocks_per_seq,
            plan.num_q_heads,
            plan.num_kv_heads,
            plan.head_dim,
            plan.block_size,
            plan.inv_sqrt_d,
            plan.num_splits,
            plan.q_stride,
            plan.num_seqs,
            plan.sliding_window,
            stream,
        )?;
        ops::paged_decode_attn_reduce_fp8(
            gpu,
            pair.reduce,
            workspace,
            output,
            seq_lens,
            plan.num_q_heads,
            plan.head_dim,
            plan.num_splits,
            plan.num_seqs,
            stream,
        )
    }
}

/// True when this launch should take the split-K path at all.
///
/// One split IS no split-K: the non-split kernel writes the BF16 output
/// directly and never touches the workspace, which is also why `sizes.rs` is
/// allowed to allocate nothing for the `Legacy`-at-1 case.
pub(super) fn splits_are_worth_it(num_splits: u32) -> bool {
    num_splits > 1
}
