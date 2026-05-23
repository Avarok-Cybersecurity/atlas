// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in super::super) fn write_kv_cache(
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
        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => ops::reshape_and_cache_nvfp4(
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
                key_stride,
                value_stride,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                stream,
            ),
            KvCacheDtype::Turbo4
            | KvCacheDtype::Turbo3
            | KvCacheDtype::Turbo8
            | KvCacheDtype::Turbo2 => {
                // Apply WHT to K and V before writing to turbo cache.
                // K and V are laid out as `[num_tokens, num_kv_heads, head_dim]`
                // BF16; the WHT kernel takes `[num_heads, head_dim]` and runs
                // one CTA per head. Grid must cover ALL (token × kv_head)
                // pairs — using `num_kv_heads` alone only WHTs the first
                // token's heads and leaves the rest of prefill un-WHT'd in
                // the cache, which collapses long-context decode (the cache
                // mixes WHT'd reads of Q with un-WHT'd K/V for tokens 1+).
                // WHT bookend (Turbo3/4/8 with Walsh-Hadamard decorrelation).
                // 2026-04-28: was temporarily gated behind ATLAS_TURBO_ENABLE_WHT=1
                // because FP8 per-group scales (~12% precision) compounded WHT
                // round-trip errors catastrophically. Resolved by upgrading
                // Turbo8 scales to BF16 (~0.4% precision); WHT is back on by
                // default. Turbo3/4 still use FP8 scales — they're affected
                // less because their LUTs already have lower precision targets.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(k)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                    // InnerQ post-WHT scale on K. Also accumulates K² stats during
                    // the calibration phase (head 0 only). No-op when d_innerq_active=0
                    // AND d_innerq_calibrating=0.
                    if self.innerq_apply_k_k.0 != 0 && head_dim == 128 {
                        KernelLaunch::new(gpu, self.innerq_apply_k_k)
                            .grid([total_heads, 1, 1])
                            .block([32, 1, 1])
                            .arg_ptr(k)
                            .arg_u32(head_dim)
                            .launch(stream)?;
                    }
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo8 => kv_cache.turbo8_data_bytes() as u64,
                    KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V => {
                        kv_cache.turbo3_data_bytes() as u64
                    }
                    KvCacheDtype::Turbo2 => kv_cache.turbo2_data_bytes() as u64,
                    _ => kv_cache.nvfp4_data_bytes() as u64, // turbo4 same as nvfp4
                };
                ops::reshape_and_cache_nvfp4(
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
                    key_stride,
                    value_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo3V => {
                // TurboQuant+ safer-asym: K = bf16, V = turbo3 — single
                // combined write kernel writes K as bf16 + V as 3-bit packed
                // with matched-norm scale into separate-stride pools.
                //
                // V-side WHT bookend (mirrors symmetric turbo3 path). K stays
                // in raw bf16 — no rotation needed because BF16 has enough
                // dynamic range to absorb outliers natively.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo3v(
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
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo3_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo4V => {
                // TurboQuant+ safer-asym: K = bf16, V = turbo4 — single
                // combined write kernel writes K as bf16 + V as 4-bit packed
                // with matched-norm scale into separate-stride pools.
                //
                // V-side WHT bookend (mirrors bf16k_turbo3v path). K stays
                // in raw bf16 — no rotation needed because BF16 has enough
                // dynamic range to absorb outliers natively.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo4v(
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
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo2V => {
                // TurboQuant+ safer-asym: K = bf16, V = turbo2 (6.4x V
                // compression). V-side WHT bookend; K stays raw bf16.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                ops::reshape_and_cache_bf16k_turbo2v(
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
                    key_stride,
                    value_stride,
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo2_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
            | KvCacheDtype::Turbo3KTurbo8V => {
                // TurboQuant+ both-sides asym: K and V are BOTH turbo dtypes.
                // WHT bookend applies to BOTH K and V (mirrors sym turbo3/4/8/2
                // arm) — and InnerQ apply also fires on K when active.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(k)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                    if self.innerq_apply_k_k.0 != 0 && head_dim == 128 {
                        KernelLaunch::new(gpu, self.innerq_apply_k_k)
                            .grid([total_heads, 1, 1])
                            .block([32, 1, 1])
                            .arg_ptr(k)
                            .arg_u32(head_dim)
                            .launch(stream)?;
                    }
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                // Dispatch each combo with its own (k_data, v_data) sizes.
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Turbo4KTurbo3V => ops::reshape_and_cache_turbo4k_turbo3v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Turbo4KTurbo8V => ops::reshape_and_cache_turbo4k_turbo8v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Turbo3KTurbo8V => ops::reshape_and_cache_turbo3k_turbo8v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Fp8KTurbo3V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Fp8KTurbo2V => {
                // TurboQuant+ asym for FP8 models: K = fp8 (per-tensor `k_scale`),
                // V = turbo{3,4,2}. Single combined write kernel quantizes K to
                // FP8 and V to N-bit Lloyd-Max + matched-norm scale.
                //
                // V-side WHT bookend (mirrors bf16k_turbo*v path). K side gets
                // no WHT — its FP8 dynamic range already covers attention scores
                // adequately for the per-tensor scale model is calibrated for.
                let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if !weight_pre_rotated
                    && self.wht_bf16_k.0 != 0
                    && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let (k_scale, _v_scale) = self.effective_fp8_scales();
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Fp8KTurbo3V => ops::reshape_and_cache_fp8k_turbo3v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo4V => ops::reshape_and_cache_fp8k_turbo4v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo2V => ops::reshape_and_cache_fp8k_turbo2v(
                        gpu,
                        self.reshape_cache_k,
                        k,
                        v,
                        k_pool,
                        v_pool,
                        slot,
                        num_tokens,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        key_stride,
                        value_stride,
                        k_scale,
                        k_block_stride,
                        v_block_stride,
                        kv_cache.turbo2_data_bytes() as u64,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Bf16 => ops::reshape_and_cache(
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
                key_stride,
                value_stride,
                kv_cache.cache_stride() as u64,
                stream,
            ),
            _ => {
                // FP8 KV cache
                if !graph_capture && let Some(ref cal) = self.fp8_calibration {
                    cal.observe(gpu, k, v, num_tokens, num_kv_heads, head_dim, stream)?;
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
        }
    }
}
