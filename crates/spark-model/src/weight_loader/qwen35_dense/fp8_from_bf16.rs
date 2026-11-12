// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time BF16 -> block-scaled FP8 route for the dense
//! FFN: whether a load takes it, which layers qualify, and the quantization.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint (`Nvfp4Variant::Bf16Raw`: Holo
//! 3.1 0.8B and 4B, Qwen3.5 dense) each layer's gate/up/down are quantized ONCE
//! at load to the 128x128 block-scaled E4M3 format a native FP8 checkpoint
//! ships (`quantize_to_fp8_blockscaled`, the kernel LongCat's experts already
//! use), and installed with `DenseFfnLayer::set_fp8_weights`. From there the
//! layer is a native-FP8 layer: decode on the W8A16 GEMVs, prefill on the W8A8
//! block-scaled arm (`dense_ffn_w8a8_prefill.rs`), batched and speculative
//! paths through `forward_prefill` exactly as for a native FP8 checkpoint. No
//! new dispatch, no new weight format, no download.
//!
//! **WHY.** A single-stream decode step is bandwidth-bound: it streams every
//! weight once per token. On a 4B Holo the FFN is 4.53 GB of the 8.68 GB of
//! BF16 weights, so E4M3 removes 2.26 GB from every decode step and from
//! VRAM. That is the difference between a 16 GB card fitting the model with a
//! usable KV cache and not (see the 4B attempts recorded for the Ada target).
//! Prefill is compute-bound, and the W8A8 arm feeds FP8 tensor cores at twice
//! the BF16 rate. Both are expectations to be measured, not measurements.
//!
//! **THE PRICE.** E4M3 keeps 3 mantissa bits against BF16's 7. That is a
//! numerics change, so the route is a TARGET DECLARATION
//! (`[defaults] dense_fp8_from_bf16`, false on every target until a card shows
//! it holds accuracy) with `AVAROK_DENSE_FP8_FROM_BF16=0|1` as the A/B. The
//! format is the one the official block-FP8 releases use, so the expected
//! accuracy is theirs.
//!
//! **SCOPE.** The dense FFN only. Attention q/k/v/o and the GDN projections
//! keep their BF16 paths; each has a native-FP8 overlay of its own and is the
//! next step, not this one.
//!
//! **LIFETIME.** The BF16 source is freed right after its quantize, as
//! `quantized_any`'s `Bf16Raw` arm frees it after the NVFP4 quantize: without
//! that, the load peak would be the BF16 checkpoint PLUS every FP8 copy.

use anyhow::Result;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::fp8_residency::DerivedResidency;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, Nvfp4Variant, quantize_to_fp8_blockscaled};

/// Module and entry of the block-scaled quantizer (`kernels/<hw>/common/`).
pub const QUANT_MODULE: &str = "quantize_bf16_to_fp8_blockscaled";

/// Everything the per-load decision reads, as data, so the CPU table test can
/// pin each clause without a GPU or the process environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteInputs {
    /// `[defaults] dense_fp8_from_bf16` after `AVAROK_DENSE_FP8_FROM_BF16`.
    pub lever: bool,
    /// The checkpoint is plain BF16 (`Nvfp4Variant::Bf16Raw`). A native FP8
    /// checkpoint has its own route; an NVFP4 one has nothing BF16 to quantize.
    pub bf16_checkpoint: bool,
    /// `tp_world_size == 1`: FP8 FFN sharding is not implemented, the same
    /// clause the native-FP8 route has.
    pub single_gpu: bool,
    /// A LoRA adapter is pending (`adapter_max_rank > 0`).
    /// `DenseFfnLayer::set_lora_weights` refuses an FP8 overlay, so the route
    /// declines rather than turn an adapter load into a refused one.
    pub lora: bool,
    /// The quantizer and the FP8 M=1 decode GEMV both resolve. Without the
    /// GEMV the FP8 arm would launch a null handle on the first token.
    pub kernels: bool,
}

/// The decision table. Pure.
pub fn route_selected(i: RouteInputs) -> bool {
    i.lever && i.bf16_checkpoint && i.single_gpu && !i.lora && i.kernels
}

/// `[n, k]` BF16 on the store under `{name}`.
fn store_bf16(store: &WeightStore, name: &str, n: usize, k: usize) -> bool {
    store
        .get(name)
        .is_ok_and(|w| w.dtype == WeightDtype::BF16 && w.shape == [n, k])
}

/// Per-layer qualification: gate and up are `[inter, hidden]` and down is
/// `[hidden, inter]`, all BF16, and both extents are whole 16-element GEMV
/// chunks. A layer that fails it keeps its BF16 path, so a checkpoint with an
/// odd layer cannot be half-quantized into a crash.
pub fn layer_shapes_ok(store: &WeightStore, lp: &str, hidden: usize, inter: usize) -> bool {
    hidden > 0
        && inter > 0
        && hidden.is_multiple_of(16)
        && inter.is_multiple_of(16)
        && store_bf16(store, &format!("{lp}.mlp.gate_proj.weight"), inter, hidden)
        && store_bf16(store, &format!("{lp}.mlp.up_proj.weight"), inter, hidden)
        && store_bf16(store, &format!("{lp}.mlp.down_proj.weight"), hidden, inter)
}

/// FP8 bytes plus the FP32 scale grid one quantized `[n, k]` weight costs.
pub fn fp8_weight_bytes(n: usize, k: usize) -> (usize, usize) {
    (n * k, n.div_ceil(128) * k.div_ceil(128) * 4)
}

/// Bytes one qualifying layer hands back: its BF16 gate, up and down minus
/// the FP8 bytes and scale grids that replace them.
pub fn layer_freed_bytes(hidden: usize, inter: usize) -> usize {
    let bf16 = 3 * hidden * inter * 2;
    let kept: usize = [(inter, hidden), (inter, hidden), (hidden, inter)]
        .iter()
        .map(|&(n, k)| {
            let (w, s) = fp8_weight_bytes(n, k);
            w + s
        })
        .sum();
    bf16.saturating_sub(kept)
}

/// What the route will free while the model is built, or 0 when it will not
/// run on this load: the same clauses as [`Fp8FromBf16::resolve`], without
/// its log line, over the layers [`layer_shapes_ok`] admits.
///
/// WHY. The server's post-load memory audit measures free memory between the
/// checkpoint upload and `load_layers`, while the BF16 FFN this route replaces
/// still occupies it, and the inference buffers it guards are allocated after
/// the layers are built. Without this credit the audit refuses serves that
/// fit: on a 16 GB card the 4B with `--speculative` and the FP8 FFN measured
/// 5.98 GB free against 6.68 GB needed, while the route frees 2.2 GB first.
pub fn deferred_free_bytes(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    tp_world_size: usize,
    lora: bool,
) -> usize {
    // The factory's loader choice for this route (`loader_for_config`).
    let family = config.model_type.to_lowercase().replace(['-', '.'], "_");
    let dense_loader = matches!(
        family.as_str(),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35"
    ) && config.is_qwen35_dense();
    let lever = ops::target_defaults::resolved().dense_fp8_from_bf16.value;
    if !(dense_loader && lever) {
        return 0;
    }
    let quant_k = crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE);
    let gemv = crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::resolve(gpu).gemv;
    let on = route_selected(RouteInputs {
        lever,
        bf16_checkpoint: matches!(
            crate::weight_map::detect_nvfp4_variant(store, config),
            Nvfp4Variant::Bf16Raw
        ),
        single_gpu: tp_world_size.max(1) == 1,
        lora,
        kernels: quant_k.0 != 0 && gemv.0 != 0,
    });
    if !on {
        return 0;
    }
    let (h, i) = (config.hidden_size, config.intermediate_size);
    let layers = (0..config.num_hidden_layers)
        .filter(|&l| layer_shapes_ok(store, &config.layer_prefix(l), h, i))
        .count();
    layers * layer_freed_bytes(h, i)
}

/// The route, resolved once per load: `Some` when it is on.
#[derive(Clone, Copy, Debug)]
pub struct Fp8FromBf16 {
    quant_k: KernelHandle,
}

impl Fp8FromBf16 {
    /// Resolve the route from the resolved lever
    /// (`ops::target_defaults::resolved().dense_fp8_from_bf16`, passed in so a
    /// test can drive it) and, only when the lever is on, the kernels. A
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
        let gemv = crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::resolve(gpu).gemv;
        let inputs = RouteInputs {
            lever: true,
            bf16_checkpoint: matches!(variant, Nvfp4Variant::Bf16Raw),
            single_gpu: tp_world_size.max(1) == 1,
            lora: adapter_max_rank > 0,
            kernels: quant_k.0 != 0 && gemv.0 != 0,
        };
        let on = route_selected(inputs);
        if inputs.bf16_checkpoint {
            tracing::info!(
                "[avarok] dense FFN BF16 -> FP8 at load: {} (dense_fp8_from_bf16 on{}; \
                 single_gpu={} lora={} kernels={}). 128x128 block-scaled E4M3, \
                 AVAROK_DENSE_FP8_FROM_BF16=0 keeps BF16.",
                if on { "ON" } else { "declined" },
                if lever.from_env() { " (env)" } else { "" },
                inputs.single_gpu,
                inputs.lora,
                inputs.kernels,
            );
        }
        on.then_some(Self { quant_k })
    }

    /// Quantize one layer's gate, up and down. Each projection's FP8 bytes and
    /// scale grid are adopted into the store's derived ledger (#736: every
    /// load-time allocation has an owner) and tallied as kept residency; each
    /// BF16 source is freed once its quantize has been issued (see LIFETIME).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_ffn(
        &self,
        store: &WeightStore,
        lp: &str,
        hidden: usize,
        inter: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        residency: &mut DerivedResidency,
    ) -> Result<[Fp8Weight; 3]> {
        let one = |name: &str, n: usize, k: usize, residency: &mut DerivedResidency| {
            let src = store.get(&format!("{lp}.mlp.{name}.weight"))?;
            let bf16 = DenseWeight { weight: src.ptr };
            let w = quantize_to_fp8_blockscaled(&bf16, n, k, gpu, self.quant_k, stream)?;
            let (wb, sb) = fp8_weight_bytes(n, k);
            let d = store.derived();
            d.adopt("ffn fp8 from bf16", w.weight, wb);
            d.adopt("ffn fp8 from bf16 block scale", w.row_scale, sb);
            residency.keep(wb + sb);
            // The checkpoint tensor itself, not a derived copy, so it is not a
            // `residency.free`: the summary's `weights` figure is where it
            // shows. The store's map keeps the (now dead) entry, exactly as
            // after `quantized_any`'s Bf16Raw free; nothing reads it again.
            gpu.free(src.ptr)?;
            Ok::<_, anyhow::Error>(w)
        };
        Ok([
            one("gate_proj", inter, hidden, residency)?,
            one("up_proj", inter, hidden, residency)?,
            one("down_proj", hidden, inter, residency)?,
        ])
    }
}

#[cfg(test)]
#[path = "fp8_from_bf16_tests.rs"]
mod tests;
