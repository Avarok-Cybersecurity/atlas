// SPDX-License-Identifier: AGPL-3.0-only

//! The FP8 arm of `write_kv_cache`: the calibration window's view of the
//! write, then the write. Split out of `write_kv_cache.rs` when #919's write
//! target pushed that file past the 500-line cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layers::fp8_calibration::Fp8KvWriteTarget;
use crate::layers::ops;

/// Largest `head_dim` the fused decode kernel's shared-memory row holds
/// (`FUSED_KFP8_MAX_HEAD_DIM` in `reshape_and_cache_fused_k_fp8.cu`). The two
/// must move together.
const FUSED_FP8_MAX_HEAD_DIM: u32 = 256;

impl Qwen3AttentionLayer {
    /// #919: `observe` accumulates the amax over the requested
    /// `--fp8-kv-calibration-tokens` window ACROSS requests and, when the
    /// window closes, requantizes the entries the window wrote — so it needs
    /// to know where this write is going. It runs BEFORE the write, so
    /// `effective_fp8_scales()` returns exactly the scale the batch is about
    /// to be written with (and every later read dequantizes with). A layer
    /// with no calibration window observes nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_fp8_kv_write(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(ref cal) = self.fp8_calibration else {
            return Ok(());
        };
        let target = Fp8KvWriteTarget {
            kernel: self.reshape_cache_k,
            k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
            block_size,
            cache_stride: kv_cache.cache_stride() as u64,
            key_stride,
            value_stride,
            slot,
        };
        cal.observe(
            gpu,
            k,
            v,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
            &target,
        )
    }
    /// Observe (outside graph capture), then write with the scales the window
    /// settled on.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_kv_cache_fp8(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
        graph_capture: bool,
    ) -> Result<()> {
        if !graph_capture {
            self.observe_fp8_kv_write(
                gpu,
                k,
                v,
                kv_cache,
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                stream,
            )?;
        }
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::reshape_and_cache_fp8(
            gpu,
            self.reshape_cache_k,
            k,
            v,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            slot,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            k_scale,
            v_scale,
            key_stride,
            value_stride,
            kv_cache.cache_stride() as u64,
            stream,
        )
    }

    /// Whether this layer's single-token decode step may take the fused
    /// k_norm+RoPE+FP8-write kernel instead of the three-launch chain.
    ///
    /// Every clause is a precondition the fused kernel does NOT check for
    /// itself, so this is the whole contract. Refusing is always safe: the
    /// caller runs the un-fused chain, which is what every other dtype and
    /// every other RoPE flavour does anyway.
    ///
    ///   * `Fp8` — the only dtype whose write is `reshape_and_cache_flash_fp8`.
    ///   * handle non-zero — `reshape_and_cache_fused_k_fp8.cu` lives in the
    ///     gb10 tree and is mirrored into the targets that inherit it
    ///     (hopper, b200); targets with their own `common/` do not carry it.
    ///   * no MLA / YaRN / proportional / MRoPE — the kernel implements
    ///     `rope_forward`'s rotate-half with a `theta`-derived frequency and
    ///     nothing else. Each of the other four flavours is a different
    ///     rotation and would need its own fused kernel.
    ///   * per-head `k_norm` present and not the `k_norm_full` variant — the
    ///     kernel reduces over `head_dim`, one CTA per (token, kv_head).
    ///   * not `norm_vanilla` — the kernel hardcodes `rms_norm`'s
    ///     offset-from-1 `(1 + w)`; `rms_norm_vanilla` applies plain `w`.
    ///   * no FP8 calibration on this layer — `observe_fp8_kv_write` measures
    ///     the amax of the POST-norm, POST-RoPE K, which on the fused path
    ///     never exists in memory. Fusing under an open window would
    ///     calibrate against raw K and mis-scale the cache.
    ///   * `head_dim` a multiple of 32 (full warps for the `__shfl_xor_sync`
    ///     butterfly, and an even pair count) and within the kernel's
    ///     shared-memory row.
    ///   * `rotary_dim` even and within `head_dim` — `rope_forward` pairs
    ///     `(d, d + rotary_dim/2)`, which is only defined for an even
    ///     `rotary_dim`. Checked HERE rather than inside the write so an
    ///     unusual value falls back instead of failing the request.
    pub(in super::super) fn fused_fp8_kv_decode_eligible(
        &self,
        head_dim: u32,
        rotary_dim: u32,
    ) -> bool {
        self.kv_dtype == KvCacheDtype::Fp8
            && self.fused_k_norm_rope_cache_write_fp8_kv_k.0 != 0
            && self.mla.is_none()
            && self.yarn_inv_freq.is_null()
            && !self.rope_proportional
            && !self.mrope_interleaved
            && self.attn.k_norm_full.is_none()
            && !self.attn.k_norm.weight.is_null()
            && !self.norm_vanilla
            && self.fp8_calibration.is_none()
            && head_dim > 0
            && head_dim.is_multiple_of(32)
            && head_dim <= FUSED_FP8_MAX_HEAD_DIM
            && rotary_dim > 0
            && rotary_dim.is_multiple_of(2)
            && rotary_dim <= head_dim
    }

    /// The fused write. Caller MUST have checked
    /// [`Self::fused_fp8_kv_decode_eligible`] and MUST have skipped both the
    /// K-side `rms_norm` and the K half of the RoPE launch — this kernel
    /// consumes the RAW projection output and redoes both internally.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn write_kv_cache_fp8_fused(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        positions: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        rms_eps: f32,
        theta: f32,
        stream: u64,
    ) -> Result<()> {
        // Defensive: the caller gates on the same predicate, so this is
        // unreachable. It is an `ensure!` rather than a `debug_assert!`
        // because the failure it guards — a fused write whose preconditions
        // do not hold — is a silently wrong KV cache, not a crash.
        anyhow::ensure!(
            self.fused_fp8_kv_decode_eligible(head_dim, rotary_dim),
            concat!(
                "write_kv_cache_fp8_fused called on an ineligible layer ",
                "(kv_dtype={:?}, head_dim={}, rotary_dim={})"
            ),
            self.kv_dtype,
            head_dim,
            rotary_dim,
        );
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::fused_k_norm_rope_cache_write_fp8_kv(
            gpu,
            self.fused_k_norm_rope_cache_write_fp8_kv_k,
            k,
            v,
            self.attn.k_norm.weight,
            positions,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            slot,
            num_tokens,
            num_kv_heads,
            head_dim,
            rotary_dim,
            block_size,
            k_scale,
            v_scale,
            key_stride,
            value_stride,
            kv_cache.cache_stride() as u64,
            rms_eps,
            theta,
            stream,
        )
    }
}
