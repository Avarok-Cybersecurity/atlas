// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time BF16 -> block-scaled FP8 route for the
//! full-attention q/k/v/o projections: whether a load takes it, which layers
//! qualify, the quantization, and what it frees.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint (`Nvfp4Variant::Bf16Raw`) the
//! loader keeps q/k/v BF16 in `AttentionWeights` and installs o_proj as the
//! BF16 dense O-proj. With this route on, all four are quantized ONCE to the
//! 128x128 block-scaled E4M3 format a native FP8 checkpoint ships
//! (`quantize_to_fp8_blockscaled`), the BF16 sources are freed, and the layer
//! is built as the native-FP8 attention overlay builds it: q/k/v/o slots
//! NULL, no NVFP4 copy, no BF16 O-proj, and `set_fp8_weights(q, k, v, o)`.
//! No transposed FP8 twins are built: every prefill arm that reads them has a
//! W8A8 or a non-transposed W8A16 arm ahead of or behind it.
//!
//! | path | kernel |
//! |---|---|
//! | M=1 decode (`decode/attention_forward*.rs`) | `w8a16_gemv_k`: `fp8_gemv_ada` on sm_89, else `w8a16_gemv` |
//! | batched decode | the native-FP8 overlay's W8A16 GEMV and GEMM tiers |
//! | prefill chunk 0 Q/K/V (`prefill/cache_skip_qkv.rs`) | in-tree W8A8: one activation quant, `fp8_gemm_ada_w8a8` x3 (armed by `set_w8a8_qkv_prefill`) |
//! | prefill later chunks Q/K/V (`prefill/paged_qkv.rs`) | its existing W8A8 arm |
//! | prefill o_proj (`prefill/paged_oproj.rs`) | its existing W8A8 arm |
//!
//! **WHY.** On a 4B Holo the 8 attention layers are 0.59 GB of BF16 read per
//! decode token; E4M3 halves that. The prefill must not get slower: with FP8
//! weights and only the W8A16 tile GEMMs, prefill on sm_89 drops below the
//! BF16 cuBLASLt rate, which is why the chunk-0 chain gets an in-tree W8A8 arm
//! that does not need `AVAROK_CUBLAS_GEMM=attn`.
//!
//! **THE PRICE.** A NUMERICS lever (`[defaults] attn_fp8_from_bf16`, false
//! everywhere, `AVAROK_ATTN_FP8_FROM_BF16=0|1`), independent of the FFN and
//! GDN rows.
//!
//! **DECLINES.** Tensor parallel (the BF16 arm shards; no FP8 attention shard
//! exists on this path), a pending LoRA adapter (declined as the FFN and GDN
//! routes decline it, so no adapter load meets an FP8 overlay it was not
//! tested on), a missing quantizer or M=1 GEMV, and any extent that is not a
//! whole 128-block. A declined layer keeps BF16.

use anyhow::Result;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::fp8_from_bf16::{QUANT_MODULE, RouteInputs, fp8_weight_bytes, route_selected};
use super::fp8_residency::DerivedResidency;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, Nvfp4Variant, quantize_to_fp8_blockscaled};

/// The four attention projection extents, `[n, k]` each: q `[q_n, hidden]`
/// (`q_n` doubled for a gated q), k and v `[kv_n, hidden]`, o `[hidden, o_k]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShapes {
    pub q_n: usize,
    pub kv_n: usize,
    pub o_k: usize,
    pub hidden: usize,
}

impl AttnShapes {
    pub fn of(config: &ModelConfig) -> Self {
        let (nh, hd) = (config.num_attention_heads, config.head_dim);
        Self {
            q_n: nh * hd * if config.attn_gated { 2 } else { 1 },
            kv_n: config.num_key_value_heads * hd,
            o_k: nh * hd,
            hidden: config.hidden_size,
        }
    }

    /// `[n, k]` of q, k, v, o, in that order.
    pub fn projections(self) -> [(usize, usize); 4] {
        [
            (self.q_n, self.hidden),
            (self.kv_n, self.hidden),
            (self.kv_n, self.hidden),
            (self.hidden, self.o_k),
        ]
    }

    /// Every extent a whole 128-block (the scale grid and the W8A8 GEMMs).
    pub fn qualify(self) -> bool {
        [self.q_n, self.kv_n, self.o_k, self.hidden]
            .iter()
            .all(|&d| d > 0 && d.is_multiple_of(128))
    }

    /// Bytes one qualifying layer hands back: the BF16 q/k/v/o minus the FP8
    /// bytes and scale grids that replace them.
    pub fn freed_bytes(self) -> usize {
        let p = self.projections();
        let bf16: usize = p.iter().map(|&(n, k)| n * k * 2).sum();
        let kept: usize = p
            .iter()
            .map(|&(n, k)| {
                let (w, s) = fp8_weight_bytes(n, k);
                w + s
            })
            .sum();
        bf16.saturating_sub(kept)
    }
}

/// The quantizer and the M=1 FP8 decode GEMV both resolve.
fn kernels_present(gpu: &dyn GpuBackend) -> (KernelHandle, bool) {
    let quant_k = crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE);
    let gemv = crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::m1_gemv(gpu);
    (quant_k, quant_k.0 != 0 && gemv.0 != 0)
}

/// The attention part of `fp8_from_bf16::deferred_free_bytes`: what this
/// route frees while the model is built, or 0 when it will not run.
pub fn deferred_free_bytes(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    bf16_checkpoint: bool,
    tp_world_size: usize,
    lora: bool,
) -> usize {
    let lever = ops::target_defaults::resolved().attn_fp8_from_bf16.value;
    if !lever {
        return 0;
    }
    let on = route_selected(RouteInputs {
        lever,
        bf16_checkpoint,
        single_gpu: tp_world_size.max(1) == 1,
        lora,
        kernels: kernels_present(gpu).1,
    });
    let shapes = AttnShapes::of(config);
    if !(on && shapes.qualify()) {
        return 0;
    }
    config.num_attention_layers() * shapes.freed_bytes()
}

/// The route, resolved once per load: `Some` when it is on.
#[derive(Clone, Copy, Debug)]
pub struct AttnFp8FromBf16 {
    pub(super) quant_k: KernelHandle,
}

impl AttnFp8FromBf16 {
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
        let (quant_k, kernels) = kernels_present(gpu);
        let inputs = RouteInputs {
            lever: true,
            bf16_checkpoint: matches!(variant, Nvfp4Variant::Bf16Raw),
            single_gpu: tp_world_size.max(1) == 1,
            lora: adapter_max_rank > 0,
            kernels,
        };
        let on = route_selected(inputs);
        if inputs.bf16_checkpoint {
            tracing::info!(
                "[avarok] attention q/k/v/o BF16 -> FP8 at load: {} \
                 (attn_fp8_from_bf16 on{}; single_gpu={} lora={} kernels={}). 128x128 \
                 block-scaled E4M3; decode W8A16 GEMV, prefill W8A8. \
                 AVAROK_ATTN_FP8_FROM_BF16=0 keeps BF16.",
                if on { "ON" } else { "declined" },
                if lever.from_env() { " (env)" } else { "" },
                inputs.single_gpu,
                inputs.lora,
                inputs.kernels,
            );
        }
        on.then_some(Self { quant_k })
    }

    /// Quantize one layer's q, k, v and o (in that order), adopt each FP8
    /// weight and scale grid into the store's derived ledger, tally them as
    /// kept residency, and free each BF16 source once its quantize is issued
    /// (same stream). The sources are the checkpoint tensors, freed as the
    /// dense-FFN route frees its own.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_attn(
        &self,
        store: &WeightStore,
        src: [DenseWeight; 4],
        shapes: AttnShapes,
        gpu: &dyn GpuBackend,
        stream: u64,
        residency: &mut DerivedResidency,
    ) -> Result<[Fp8Weight; 4]> {
        let mut one = |src: DenseWeight, (n, k): (usize, usize)| {
            let w = quantize_to_fp8_blockscaled(&src, n, k, gpu, self.quant_k, stream)?;
            let (wb, sb) = fp8_weight_bytes(n, k);
            let d = store.derived();
            d.adopt("attn fp8 from bf16", w.weight, wb);
            d.adopt("attn fp8 from bf16 block scale", w.row_scale, sb);
            residency.keep(wb + sb);
            gpu.free(src.weight)?;
            Ok::<_, anyhow::Error>(w)
        };
        let [pq, pk, pv, po] = shapes.projections();
        let [sq, sk, sv, so] = src;
        Ok([one(sq, pq)?, one(sk, pk)?, one(sv, pv)?, one(so, po)?])
    }
}

#[cfg(test)]
#[path = "attn_fp8_from_bf16_tests.rs"]
mod tests;
