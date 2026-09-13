// SPDX-License-Identifier: AGPL-3.0-only

//! The LEDGERED BF16 dequant behind the two `ATLAS_FP8_ROWWISE` GDN prefill
//! arms (`trait_prefill_proj.rs`'s `in_proj_qkvz`, `trait_prefill_helper.rs`'s
//! `out_proj`).
//!
//! # Why these arms dequantise at all
//!
//! The checkpoint ships those two projections as FP8 E4M3 with a PER-ROW
//! scale, and `cublaslt::fp8_gemm_act_weight_t_rowwise` — the only cuBLASLt
//! entry that consumes a per-row pair — returns NOT_SUPPORTED on sm_121
//! (measured 2026-08-15, reproduced through the block-scaled path with
//! `ATLAS_CUBLAS_FP8=1`, so it is the GEMM and not the weights; see
//! `ops/dispatch_proj_rowwise.rs`). The block-scaled W8A8 route the attention
//! and default GDN arms took in #927/#928 is closed to them for the same
//! reason from the other side: a per-row scale is not a `[N/128, K/128]` grid
//! and the block kernels would read it as one. So BF16 it is — lossless here,
//! since every FP8 E4M3 value is exactly representable in BF16.
//!
//! # Why the bytes live in the arena
//!
//! **#917 H100 receipt, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`.** The dequant used
//! to be `ops::cublas_bf16_proj`'s `dequant_fp8_bf16_cached`: a lazy
//! `gpu.alloc` memoised by weight pointer, `167772160` B for the fused
//! `[QKV|Z]` weight PER LAYER (`[10240,5120] + [6144,5120]`, x2 B) with no
//! `BufferSizes` entry — so `--gpu-memory-utilization` could not see it, the
//! preflight ring fitter could not price it, and a 28-token prefill consumed
//! 6120 MiB and died at layer 36 with `cuMemAlloc_v2 failed: status 2`. The
//! attention arms that shared the defect were deleted in #927 and are held
//! deleted by `qwen3_attention/prefill/alloc_tests.rs`; these two cannot be
//! deleted, because there is no other route for a per-row checkpoint.
//!
//! So the same bytes are now ONE arena allocation
//! (`BufferSizes::ssm_rowwise_w_bf16`, sized in
//! `spark_runtime::buffers::sizes_rowwise` for `num_ssm_layers` x
//! (`in_proj_qkvz` + `out_proj`) and armed only when `ATLAS_FP8_ROWWISE=1`, so
//! every other recipe's ledger is unchanged). Each layer bump-carves its two
//! slices on its FIRST prefill and remembers them here; nothing in the prefill
//! path allocates.
//!
//! The by-pointer cache is gone with it. The slot below is a per-LAYER
//! pointer, which is what the lifetime actually is — the old map was keyed on
//! a device pointer whose recycling after a model swap is exactly the hazard
//! `DerivedWeights` documents, and it outlived nothing useful.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;
use spark_runtime::gpu::DevicePtr;

impl Qwen3SsmLayer {
    /// Ledgered BF16 `in_proj_qkvz` `[qkvz_size, hidden]` for the row-wise arm.
    pub(super) fn rowwise_qkvz_bf16(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.rowwise_bf16(&self.qkvz_rowwise_bf16, ctx, fp8w, "in_proj_qkvz", stream)
    }

    /// Ledgered BF16 `out_proj` `[hidden, value_dim]` for the row-wise arm.
    pub(super) fn rowwise_out_proj_bf16(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.rowwise_bf16(&self.out_proj_rowwise_bf16, ctx, fp8w, "out_proj", stream)
    }

    /// Dequantise `fp8w` into this layer's slice of the arena slab, once.
    ///
    /// `slot` is 0 until the first prefill through this projection; after that
    /// it is the slice, and every later call is a load. `Relaxed` is enough
    /// for the same reason the arena's cursor uses it: the scheduler drives
    /// one forward at a time.
    ///
    /// ★ ALLOCATES NOTHING — that is the contract
    /// `rowwise_alloc_tests.rs` pins, and the whole point of the change.
    fn rowwise_bf16(
        &self,
        slot: &AtomicU64,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        what: &str,
        stream: u64,
    ) -> Result<DevicePtr> {
        let cached = slot.load(Ordering::Relaxed);
        if cached != 0 {
            return Ok(DevicePtr(cached));
        }
        let bytes = ops::dequant_fp8_bf16_bytes(fp8w);
        let dst = ctx
            .buffers
            .take_ssm_rowwise_w_bf16(bytes)
            .with_context(|| format!("ssm prefill: row-wise BF16 {what} weight"))?;
        ops::dequant_fp8_bf16_into(ctx.gpu, fp8w, dst, stream)
            .with_context(|| format!("ssm prefill: row-wise BF16 dequant of {what}"))?;
        slot.store(dst.0, Ordering::Relaxed);
        Ok(dst)
    }
}
