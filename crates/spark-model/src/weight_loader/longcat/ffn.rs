// SPDX-License-Identifier: AGPL-3.0-only

//! The two per-sublayer FFN builders: the dense SwiGLU MLP every sublayer
//! has, and the shortcut MoE that sublayer 0 computes and sublayer 1 adds.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_f32_as_bf16,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn build_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
    let inter = config.intermediate_size;
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let q = |name: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let w = dense(store, name)?;
        quantize_to_nvfp4(&w, n, k, gpu, absmax_k, quantize_k, stream)
    };
    let weights = DenseFfnWeights {
        gate_proj: q(&format!("{prefix}.gate_proj.weight"), inter, h)?,
        up_proj: q(&format!("{prefix}.up_proj.weight"), inter, h)?,
        down_proj: q(&format!("{prefix}.down_proj.weight"), h, inter)?,
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    let mut layer = DenseFfnLayer::new(weights, gpu)?;

    // Precision lever, mirroring `ATLAS_NVFP4_MLA=0` on the attention side.
    // LongCat ships plain BF16 with no calibration metadata, so these three
    // projections are runtime-quantized above and lose whatever 4 bits cost.
    // The per-sublayer dense FFN is 2.94 GB of the checkpoint across all 28
    // sublayers, so holding it in BF16 costs ~2.2 GB over NVFP4 — cheap next
    // to the 63 GB of routed experts, which is why it is a separate switch.
    //
    // Both copies stay resident: `set_bf16_weights` only redirects dispatch,
    // and the NVFP4 copy is the fallback for any arm that has no BF16 kernel.
    if bf16_dense_ffn() {
        layer.set_bf16_weights(
            dense(store, &format!("{prefix}.gate_proj.weight"))?,
            dense(store, &format!("{prefix}.up_proj.weight"))?,
            dense(store, &format!("{prefix}.down_proj.weight"))?,
        );
    }
    Ok(FfnComponent::Dense(layer))
}

/// `ATLAS_LONGCAT_BF16_FFN=1` keeps the per-sublayer dense FFN in BF16.
pub(super) fn bf16_dense_ffn() -> bool {
    std::env::var("ATLAS_LONGCAT_BF16_FFN").as_deref() == Ok("1")
}

/// The block's shortcut MoE: `mlp.router.*` + `mlp.experts.{e}.*`. LongCat has
/// NO shared expert (the zero/identity experts play that role), so the shared
/// slot is the zero-filled dummy the fused kernels still read.
pub(super) fn build_shortcut_moe(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    let p = format!("{layer_prefix}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    // Router classifier + bias ship F32 (`_keep_in_fp32_modules`); the gate
    // GEMV wants BF16 weights, the bias stays F32 for the router kernel.
    let gate_name = format!("{p}.router.classifier.weight");
    let gate = dense_f32_as_bf16(store, &gate_name, gpu)
        .or_else(|_| dense(store, &gate_name))
        .context("longcat: router classifier")?;
    let correction_bias = dense(store, &format!("{p}.router.e_score_correction_bias"))
        .context("longcat: router e_score_correction_bias")?;

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };
    let group = 16usize;
    let mk_zero = |packed: usize, scale: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed)?,
            weight_scale: alloc_zero(scale)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        up_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        down_proj: mk_zero(h * inter / 2, h * (inter / group))?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // LongCat ships PLAIN BF16 experts (torch_dtype bfloat16, no NVFP4/FP8
    // metadata), so they are runtime-quantized at load — the Bf16Raw variant.
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let qctx = crate::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    // The experts are 63.0 of the 70.2 GB resident, and after the MLA and
    // dense-FFN levers they are the ONLY component still quantized — so they
    // are the whole remaining gap against the reference logits. FP8 is the
    // upgrade that fits: BF16 would add ~47 GB against a 97.3 GB budget,
    // FP8 adds ~15.75 GB. Because routing reads only top-12 of 256 per token
    // it is also the CHEAPEST lever at decode (+0.74 GB/token, less than
    // either dense BF16 arm).
    let fp8 = fp8_experts();
    let mut experts = Vec::with_capacity(config.num_experts);
    let mut fp8_experts_vec = Vec::with_capacity(if fp8 { config.num_experts } else { 0 });
    let fp8_quant_k = if fp8 {
        gpu.kernel(
            "quantize_bf16_to_fp8_blockscaled",
            "quantize_bf16_to_fp8_blockscaled",
        )?
    } else {
        spark_runtime::gpu::KernelHandle(0)
    };
    for e in 0..config.num_experts {
        let ep = format!("{p}.experts.{e}");
        if fp8 {
            // NVFP4 slot stays NULL so both copies are never resident at
            // once; dispatch takes the FP8 tables, which are installed below
            // and gate every arm that would otherwise read these pointers.
            experts.push(ExpertWeight::null());
            fp8_experts_vec.push(crate::weight_map::Fp8ExpertWeight {
                gate_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
                up_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
                down_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.down_proj"),
                    h,
                    inter,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
            });
            continue;
        }
        experts.push(ExpertWeight {
            gate_proj: quantized_any(
                store,
                &format!("{ep}.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            up_proj: quantized_any(
                store,
                &format!("{ep}.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            down_proj: quantized_any(
                store,
                &format!("{ep}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?,
        });
    }

    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    let mut moe = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    if fp8 {
        // Fails LOUD. A silent fall-through here would leave the NULL NVFP4
        // experts installed above as the only expert weights, and every token
        // would route into zeroed matrices — fluent, confident, wrong output
        // with nothing in the log.
        // Zero-filled shared slot, NOT nulls: LongCat has no shared expert
        // (the identity experts play that role), but the fused kernels read
        // the slot unconditionally — same reason `mk_zero` exists above for
        // the NVFP4 twin. A null here is a device-side deref, not a no-op.
        let mk_zero_fp8 = |n: usize, k: usize| -> Result<crate::weight_map::Fp8Weight> {
            Ok(crate::weight_map::Fp8Weight {
                weight: alloc_zero(n * k)?,
                row_scale: alloc_zero(n.div_ceil(128) * k.div_ceil(128) * 4)?,
                n: n as u32,
                k: k as u32,
                scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
            })
        };
        moe.set_fp8_experts(
            &fp8_experts_vec,
            crate::weight_map::Fp8ExpertWeight {
                gate_proj: mk_zero_fp8(inter, h)?,
                up_proj: mk_zero_fp8(inter, h)?,
                down_proj: mk_zero_fp8(h, inter)?,
            },
            gpu,
        )
        .context("longcat: installing FP8 expert pointer tables")?;
    }
    Ok(FfnComponent::Moe(moe))
}

/// `ATLAS_LONGCAT_FP8_EXPERTS=1` runtime-quantizes the routed experts to
/// block-scaled FP8 instead of NVFP4.
pub(super) fn fp8_experts() -> bool {
    std::env::var("ATLAS_LONGCAT_FP8_EXPERTS").as_deref() == Ok("1")
}

/// One expert projection: BF16 from the store → block-scaled FP8, then free
/// the BF16 source. Mirrors the `Bf16Raw` NVFP4 arm's free — without it the
/// 63 GB of BF16 experts stay resident alongside their 31.5 GB FP8 copies.
fn quant_expert_fp8(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<crate::weight_map::Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let bf16 = DenseWeight { weight: w.ptr };
    let q = crate::weight_map::quantize_to_fp8_blockscaled(&bf16, n, k, gpu, quantize_k, stream)?;
    // The kernel reads the BF16 source on `stream`; the free must not race it.
    gpu.synchronize(stream)?;
    gpu.free(w.ptr)?;
    Ok(q)
}
