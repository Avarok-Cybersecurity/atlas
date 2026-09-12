// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 staging for the FP8 KV calibration window (Atlas #919).
//!
//! ## Why this exists
//!
//! The FP8 KV round-trip is only correct if the scale that quantized a cache
//! entry is the scale that dequantizes it: paged attention reads a sequence's
//! whole history in one pass with ONE `k_scale`/`v_scale`. That is why the
//! 2026-07-25 hardening froze the scale on the FIRST observe — but it is also
//! why #919 happened, because "the first observe" is the readiness probe and a
//! 24k serve ended up calibrated on 13 tokens.
//!
//! ## The fix, and its tradeoff
//!
//! Accumulate the amax over the requested window and, when the freeze finally
//! fires, make the already-written entries agree with the new scale by
//! REWRITING them. Of the three options considered (#919):
//!
//! * (a) hold pre-freeze tokens in BF16 — impossible without a second pool:
//!   the FP8 pools are sized for 1 byte/element and attention must be able to
//!   read those tokens back DURING the window.
//! * (b) rescale the stored FP8 codes in place — needs a new dequant/requant
//!   kernel in all five arch trees (`kernels/{hopper,b200,gb10,strix,strix-hip}`)
//!   and loses precision twice.
//! * (c) **chosen**: keep the original BF16 K/V of the window's batches, plus
//!   their slot mappings, and replay them through the EXISTING
//!   `reshape_and_cache_fp8` kernel at the frozen scale. No new kernel, and the
//!   rewritten entries are quantized once, from the original BF16, at the final
//!   scale — strictly better than a code-domain rescale.
//!
//! Replay is chronological, so "last writer wins" per slot is identical to the
//! original write order even if a block was freed and re-allocated inside the
//! window. The crossing batch is written by the caller AFTER the replay, at the
//! frozen scale, so it needs no staging.
//!
//! Cost: `window_tokens * num_kv_heads * head_dim * 2 B * 2` device bytes per
//! attention layer, freed at the freeze. 256 tokens x 4 KV heads x 128 dims is
//! 512 KiB/layer. `MAX_STAGED_TOKENS` caps a pathological `--fp8-kv-calibration-
//! tokens`.
//!
//! Known gap, documented rather than fixed: KV spilled to host
//! (`--high-speed-swap`) inside the window is not replayed. The window is a few
//! hundred tokens, long before spill pressure.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

/// Hard cap on staged calibration tokens, so an absurd
/// `--fp8-kv-calibration-tokens` cannot reserve gigabytes of device memory per
/// layer. The effective window is clamped to this in `Fp8KvCalibration::new`.
pub(super) const MAX_STAGED_TOKENS: usize = 4096;

/// Bytes per BF16 element.
const BF16: usize = 2;
/// Bytes per `slot_mapping` entry (int64, see `reshape_and_cache_fp8.cu`).
const SLOT: usize = 8;

/// Everything the replay needs to re-run `reshape_and_cache_fp8` for one layer.
#[derive(Debug, Clone, Copy)]
pub struct Fp8KvWriteTarget {
    /// `reshape_and_cache_fp8` kernel handle (the layer's `reshape_cache_k`).
    pub kernel: KernelHandle,
    /// Base pointer of this layer's K pool.
    pub k_pool: DevicePtr,
    /// Base pointer of this layer's V pool.
    pub v_pool: DevicePtr,
    /// Tokens per KV block.
    pub block_size: u32,
    /// Pool stride in ELEMENTS (`block_size * num_kv_heads * head_dim`).
    pub cache_stride: u64,
    /// Source K row stride in elements, as passed to the live write.
    pub key_stride: u32,
    /// Source V row stride in elements, as passed to the live write.
    pub value_stride: u32,
    /// This batch's `slot_mapping` (int64 per token), staged alongside the
    /// BF16 K/V so the replay lands on exactly the entries the window wrote.
    pub slot: DevicePtr,
}

#[derive(Debug, Clone, Copy)]
struct StagedBatch {
    offset_tokens: usize,
    num_tokens: u32,
}

/// Device-side copies of the calibration window's BF16 K/V and slot mappings.
#[derive(Debug)]
pub(super) struct KvStaging {
    k: DevicePtr,
    v: DevicePtr,
    slots: DevicePtr,
    capacity_tokens: usize,
    elems_per_token: usize,
    used_tokens: usize,
    batches: Vec<StagedBatch>,
}

impl Default for KvStaging {
    fn default() -> Self {
        Self {
            k: DevicePtr(0),
            v: DevicePtr(0),
            slots: DevicePtr(0),
            capacity_tokens: 0,
            elems_per_token: 0,
            used_tokens: 0,
            batches: Vec::new(),
        }
    }
}

impl KvStaging {
    /// Tokens staged so far.
    pub(super) fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    /// Copy one pre-freeze batch aside. Silently declines (returns `false`) if
    /// the window no longer fits — the caller then writes the batch at the
    /// provisional scale and the freeze simply leaves it alone, which is the
    /// pre-#919 behaviour for that batch rather than a correctness cliff.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage(
        &mut self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        num_tokens: u32,
        elems_per_token: usize,
        target: &Fp8KvWriteTarget,
        capacity_tokens: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.capacity_tokens == 0 {
            self.k = gpu.alloc(capacity_tokens * elems_per_token * BF16)?;
            self.v = gpu.alloc(capacity_tokens * elems_per_token * BF16)?;
            self.slots = gpu.alloc(capacity_tokens * SLOT)?;
            self.capacity_tokens = capacity_tokens;
            self.elems_per_token = elems_per_token;
        }
        let n = num_tokens as usize;
        if elems_per_token != self.elems_per_token || self.used_tokens + n > self.capacity_tokens {
            return Ok(false);
        }

        let row = elems_per_token * BF16;
        let dst_off = self.used_tokens * row;
        copy_rows(
            gpu,
            k,
            target.key_stride,
            self.k.offset(dst_off),
            row,
            n,
            stream,
        )?;
        copy_rows(
            gpu,
            v,
            target.value_stride,
            self.v.offset(dst_off),
            row,
            n,
            stream,
        )?;
        gpu.copy_d2d_async(
            target.slot,
            self.slots.offset(self.used_tokens * SLOT),
            n * SLOT,
            stream,
        )?;

        self.batches.push(StagedBatch {
            offset_tokens: self.used_tokens,
            num_tokens,
        });
        self.used_tokens += n;
        Ok(true)
    }

    /// Requantize every staged batch at the frozen scale, in write order, then
    /// release the staging buffers. Returns the number of tokens rewritten.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn replay_and_release(
        &mut self,
        gpu: &dyn GpuBackend,
        target: &Fp8KvWriteTarget,
        num_kv_heads: u32,
        head_dim: u32,
        k_scale: f32,
        v_scale: f32,
        stream: u64,
    ) -> Result<usize> {
        let rewritten = self.used_tokens;
        let row = self.elems_per_token * BF16;
        for batch in std::mem::take(&mut self.batches) {
            let off = batch.offset_tokens;
            ops::reshape_and_cache_fp8(
                gpu,
                target.kernel,
                self.k.offset(off * row),
                self.v.offset(off * row),
                target.k_pool,
                target.v_pool,
                self.slots.offset(off * SLOT),
                batch.num_tokens,
                num_kv_heads,
                head_dim,
                target.block_size,
                k_scale,
                v_scale,
                // The staging buffers are packed, whatever the live source
                // strides were.
                self.elems_per_token as u32,
                self.elems_per_token as u32,
                target.cache_stride,
                stream,
            )?;
        }
        self.release(gpu)?;
        Ok(rewritten)
    }

    /// Free the staging buffers. Idempotent.
    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.capacity_tokens == 0 {
            return Ok(());
        }
        gpu.free(self.k)?;
        gpu.free(self.v)?;
        gpu.free(self.slots)?;
        *self = Self::default();
        Ok(())
    }
}

/// Copy `rows` rows of `row_bytes` from a source with `src_stride` ELEMENTS
/// between rows into a packed destination.
fn copy_rows(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    src_stride: u32,
    dst: DevicePtr,
    row_bytes: usize,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let src_pitch = src_stride as usize * BF16;
    if src_pitch == row_bytes {
        return gpu.copy_d2d_async(src, dst, row_bytes * rows, stream);
    }
    gpu.copy_d2d_2d_async(src, src_pitch, dst, row_bytes, row_bytes, rows, stream)
}
