// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time BF16 -> block-scaled FP8 route for the GDN
//! (linear-attention) projections: whether a load takes it, which layers
//! qualify, the quantization, and what it frees.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint (`Nvfp4Variant::Bf16Raw`: Holo
//! 3.1 0.8B and 4B, Qwen3.5 dense) the loader concatenates each GDN layer's
//! `in_proj_qkv` and `in_proj_z` into the BF16 `[Q|K|V|Z]` weight and, without
//! this route, keeps it and `out_proj` in BF16. With the route on, both are
//! quantized ONCE to the 128x128 block-scaled E4M3 format a native FP8
//! checkpoint ships (`quantize_to_fp8_blockscaled`, the kernel the dense-FFN
//! route already uses), the BF16 sources are freed, and the layer is built
//! EXACTLY as the loader's native-FP8 GDN arm builds it: `in_proj_qkvz` and
//! `ssm.out_proj` NULL, no NVFP4 copy, `set_fp8_decode_weights(qkvz, out)`.
//! Every consumer of that layout is already in tree and hardened against the
//! NULL BF16 slots (`trait_decode_batched.rs`, `ssm_batched*.rs` fail fast):
//!
//! | path | kernel |
//! |---|---|
//! | M=1 decode (`ssm_forward.rs`) | `w8a16_gemv_k`: `fp8_gemv_ada` on sm_89, else `w8a16_gemv` |
//! | batched decode / verify, M 2..=4 | `w8a16_gemv_batch4` |
//! | batched, M 5..=16 | `w8a16_gemv_batch16` |
//! | batched, M > 16 | `w8a16_gemm_pipelined` |
//! | prefill QKVZ (`trait_prefill_proj.rs`) | W8A8: `per_token_group_quant_fp8` + `fp8_gemm_ada_w8a8` on sm_89 |
//! | prefill `out_proj` (`trait_prefill_helper.rs`) | the same W8A8 pair, armed by `set_w8a8_out_proj_prefill` |
//!
//! **WHY.** On a 4B Holo the GDN projections are 2.0 GB of BF16 read per
//! decode token (24 layers x 41.9 M weights x 2 B), and decode is bandwidth
//! bound, so E4M3 takes about 1.0 GB out of every step and out of VRAM. The
//! prefill GEMMs (7.9 TFLOP at a 3,921-token prompt) move from BF16 to the FP8
//! tensor cores. Expectations, to be measured.
//!
//! **THE PRICE.** E4M3 keeps 3 mantissa bits, and a GDN projection error
//! feeds a recurrent state, so this is a NUMERICS lever separate from the FFN
//! one (`[defaults] gdn_fp8_from_bf16`, false everywhere,
//! `AVAROK_GDN_FP8_FROM_BF16=0|1`). The GDN precision policy is "stay at or
//! above FP8" (`qwen35_dense.rs`), which block FP8 meets.
//!
//! **DECLINES.** Tensor parallel (the BF16 arm shards before this point; no
//! FP8 GDN shard exists), a pending LoRA adapter, a missing quantizer or M=1
//! GEMV, and per layer any extent that is not a whole 128-block (the W8A8
//! grid and the block scales need it). A declined layer keeps BF16.

use anyhow::Result;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::fp8_from_bf16::{QUANT_MODULE, RouteInputs, fp8_weight_bytes, route_selected};
use super::fp8_residency::DerivedResidency;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, Nvfp4Variant, quantize_to_fp8_blockscaled};

/// The three GDN projection extents: `[qkvz, hidden]` and `[hidden, value]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnShapes {
    pub qkvz: usize,
    pub hidden: usize,
    pub value: usize,
}

impl GdnShapes {
    pub fn of(config: &ModelConfig) -> Self {
        Self {
            qkvz: config.ssm_qkvz_size(),
            hidden: config.hidden_size,
            value: config.linear_num_value_heads * config.linear_value_head_dim,
        }
    }

    /// Every extent a whole 128-block: the weight-scale grid is
    /// `[N/128, K/128]` and the W8A8 GEMM requires `N % 128 == K % 128 == 0`.
    /// A layer that fails it keeps BF16, so it cannot be half-quantized.
    pub fn qualify(self) -> bool {
        [self.qkvz, self.hidden, self.value]
            .iter()
            .all(|&d| d > 0 && d.is_multiple_of(128))
    }

    /// Bytes one qualifying layer hands back: the BF16 `[Q|K|V|Z]` and
    /// `out_proj` minus the FP8 bytes and scale grids that replace them.
    pub fn freed_bytes(self) -> usize {
        let bf16 = (self.qkvz * self.hidden + self.hidden * self.value) * 2;
        let kept: usize = [(self.qkvz, self.hidden), (self.hidden, self.value)]
            .iter()
            .map(|&(n, k)| {
                let (w, s) = fp8_weight_bytes(n, k);
                w + s
            })
            .sum();
        bf16.saturating_sub(kept)
    }
}

/// The GDN half of `fp8_from_bf16::deferred_free_bytes`: what this route frees
/// while the model is built, or 0 when it will not run. Same clauses as
/// [`GdnFp8FromBf16::resolve`], without its log line, over every GDN layer.
pub fn deferred_free_bytes(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    bf16_checkpoint: bool,
    tp_world_size: usize,
    lora: bool,
) -> usize {
    let lever = ops::target_defaults::resolved().gdn_fp8_from_bf16.value;
    if !lever {
        return 0;
    }
    let on = route_selected(RouteInputs {
        lever,
        bf16_checkpoint,
        single_gpu: tp_world_size.max(1) == 1,
        lora,
        kernels: kernels_present(gpu),
    });
    let shapes = GdnShapes::of(config);
    if !(on && shapes.qualify()) {
        return 0;
    }
    config.num_ssm_layers() * shapes.freed_bytes()
}

/// The quantizer and the M=1 FP8 decode GEMV both resolve.
fn kernels_present(gpu: &dyn GpuBackend) -> bool {
    crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE).0 != 0
        && crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::m1_gemv(gpu).0 != 0
}

/// The route, resolved once per load: `Some` when it is on.
#[derive(Clone, Copy, Debug)]
pub struct GdnFp8FromBf16 {
    pub(super) quant_k: KernelHandle,
}

impl GdnFp8FromBf16 {
    /// Resolve from the resolved lever (passed in so a test can drive it). A
    /// target that does not declare the route issues no extra lookup at all.
    pub fn resolve(
        gpu: &dyn GpuBackend,
        lever: ops::target_defaults::Resolved<bool>,
        variant: Nvfp4Variant,
        tp_world_size: usize,
        adapter_max_rank: usize,
    ) -> Option<Self> {
        if !lever.value {
            return None;
        }
        let quant_k = crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE);
        let inputs = RouteInputs {
            lever: true,
            bf16_checkpoint: matches!(variant, Nvfp4Variant::Bf16Raw),
            single_gpu: tp_world_size.max(1) == 1,
            lora: adapter_max_rank > 0,
            kernels: quant_k.0 != 0
                && crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::m1_gemv(gpu).0 != 0,
        };
        let on = route_selected(inputs);
        if inputs.bf16_checkpoint {
            tracing::info!(
                "[avarok] GDN in_proj_qkvz + out_proj BF16 -> FP8 at load: {} \
                 (gdn_fp8_from_bf16 on{}; single_gpu={} lora={} kernels={}). 128x128 \
                 block-scaled E4M3; decode W8A16 GEMV, prefill W8A8. \
                 AVAROK_GDN_FP8_FROM_BF16=0 keeps BF16.",
                if on { "ON" } else { "declined" },
                if lever.from_env() { " (env)" } else { "" },
                inputs.single_gpu,
                inputs.lora,
                inputs.kernels,
            );
        }
        on.then_some(Self { quant_k })
    }

    /// Quantize one layer's `[Q|K|V|Z]` and `out_proj`, adopt the FP8 bytes
    /// and scale grids into the store's derived ledger, tally them as kept
    /// residency, and free both BF16 sources once their quantize is issued
    /// (same stream, so the free is ordered after the read). The `[Q|K|V|Z]`
    /// source is the loader's own concat buffer; `out_proj` is the checkpoint
    /// tensor, freed as the dense-FFN route frees its sources.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_gdn(
        &self,
        store: &WeightStore,
        qkvz: DenseWeight,
        out_proj: DenseWeight,
        shapes: GdnShapes,
        gpu: &dyn GpuBackend,
        stream: u64,
        residency: &mut DerivedResidency,
    ) -> Result<(Fp8Weight, Fp8Weight)> {
        let mut one = |src: DenseWeight, n: usize, k: usize, label: &'static str| {
            let w = quantize_to_fp8_blockscaled(&src, n, k, gpu, self.quant_k, stream)?;
            let (wb, sb) = fp8_weight_bytes(n, k);
            let d = store.derived();
            d.adopt(label, w.weight, wb);
            d.adopt("ssm fp8 from bf16 block scale", w.row_scale, sb);
            residency.keep(wb + sb);
            gpu.free(src.weight)?;
            Ok::<_, anyhow::Error>(w)
        };
        let q = one(qkvz, shapes.qkvz, shapes.hidden, "ssm qkvz fp8 from bf16")?;
        let o = one(
            out_proj,
            shapes.hidden,
            shapes.value,
            "ssm out_proj fp8 from bf16",
        )?;
        Ok((q, o))
    }
}

#[cfg(test)]
#[path = "gdn_fp8_from_bf16_tests.rs"]
mod tests;
