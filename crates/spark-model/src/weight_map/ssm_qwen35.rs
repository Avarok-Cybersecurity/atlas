// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

pub use super::*;

#[path = "ssm_qwen35/dequant_fp8.rs"]
mod dequant_fp8;
use dequant_fp8::dequant_fp8_block_slice_bf16;

/// Qwen3.5 SSM weights with separate projections.
pub struct SsmWeightsQwen35 {
    /// QKV projection: [qkv_size, hidden_size] BF16 (Q+K+V, no Z).
    pub in_proj_qkv: DenseWeight,
    /// Z gate projection: [z_size, hidden_size] BF16.
    pub in_proj_z: DenseWeight,
    /// Alpha projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_a: DenseWeight,
    /// Beta projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_b: DenseWeight,
    /// Conv1d weight: [d_inner, 1, d_conv] BF16.
    pub conv1d: DenseWeight,
    /// A_log parameter: `[num_v_heads]` FP32.
    pub a_log: DenseWeight,
    /// dt_bias parameter: `[num_v_heads]` FP32.
    pub dt_bias: DenseWeight,
    /// Gate norm weight: `[value_dim]` BF16.
    pub norm: DenseWeight,
    /// Output projection: [value_dim, hidden_size] BF16 (NOT NVFP4 — quantizer skipped these).
    pub out_proj: DenseWeight,
}

/// Load SSM weights for Qwen3.5 (separate projections, BF16 out_proj).
pub(crate) fn load_ssm_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<SsmWeightsQwen35> {
    load_ssm_qwen35_parts(store, layer_prefix, gpu, variant, true, true)
}

/// [`load_ssm_qwen35`] with the linear projections optional: `load_in_proj`
/// covers `in_proj_qkv` / `in_proj_z`, `load_out_proj` covers `out_proj`.
/// `false` leaves the projection as a NULL [`DenseWeight`] — the native-EXL3
/// GDN arm (`ATLAS_EXL3_NATIVE_DENSE=1`) serves it from the packed trellis
/// and must not read a `.weight` that was never materialized; the BA/conv/
/// gate tensors still load exactly as before.
pub(crate) fn load_ssm_qwen35_parts(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    // Kept for loader-dispatch signature parity; `dense_auto` routes by the
    // projection's actual on-disk dtype rather than the model-wide variant.
    _variant: Nvfp4Variant,
    load_in_proj: bool,
    load_out_proj: bool,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.linear_attn");

    // Per-projection load by on-disk dtype. FP8 (Holo, block-scaled) and BF16
    // (AEON, in modules_to_not_convert) ship plain `.weight` → `dense_auto`.
    // The sakamakismile AgentWorld re-quant instead quantizes the GDN/SSM
    // projections to NVFP4 (`weight_packed`); dequant those to BF16, with dims
    // inferred from the packed shape (rows = packed[0], cols = packed[1] * 2,
    // since E2M1 packs 2 values/byte). Mirrors the dense loader's `load_ssm_proj`
    // (qwen35_dense.rs) — without this the NVFP4 SSM projections fail with
    // "linear_attn.in_proj_qkv.weight not found in store".
    let load_proj = |prefix: &str| -> Result<DenseWeight> {
        if store.contains(&format!("{prefix}.weight_packed")) {
            let shape = store.get(&format!("{prefix}.weight_packed"))?.shape.clone();
            dequant_nvfp4_to_bf16(store, prefix, shape[0], shape[1] * 2, gpu)
        } else {
            dense_auto(store, &format!("{prefix}.weight"), gpu)
        }
    };
    // The natively-served linears (skipped): NULL, the FP8-native precedent.
    let linear = |prefix: &str, load: bool| -> Result<DenseWeight> {
        if load {
            load_proj(prefix)
        } else {
            Ok(DenseWeight {
                weight: DevicePtr::NULL,
            })
        }
    };

    Ok(SsmWeightsQwen35 {
        in_proj_qkv: linear(&format!("{p}.in_proj_qkv"), load_in_proj)?,
        in_proj_z: linear(&format!("{p}.in_proj_z"), load_in_proj)?,
        in_proj_a: load_proj(&format!("{p}.in_proj_a"))?,
        in_proj_b: load_proj(&format!("{p}.in_proj_b"))?,
        conv1d: dense_auto(store, &format!("{p}.conv1d.weight"), gpu)?,
        // A_log and dt_bias MUST be FP32 — BF16 precision causes exponential
        // error amplification in the GDR decay gate at 8k+ tokens.
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        // norm.weight is safe as BF16 (no recurrent amplification)
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: linear(&format!("{p}.out_proj"), load_out_proj)?,
    })
}

/// Load MoE weights for Qwen3.5, auto-selecting NVFP4 naming convention.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
/// `force_all_experts`: when true, EVERY routed expert is loaded regardless of
/// `is_local_expert` — the draft (MTP) module's own MoE, which is REPLICATED on
/// every EP rank rather than sharded. Its `mtp.*` tensors are not sharded by the
/// upload, and the draft forward has no all-reduce, so a rank>0 draft would
/// otherwise route into NULL experts; that mismatch is why MTP was refused under
/// `ep_world_size > 1`. Replication costs ~1.3 GB/rank (the draft MoE) against
/// the ~20 GB/rank EP=2 saves, and keeps the latency-critical draft path free of
/// a collective. Ignored when `skip_routed_experts` is set (the native-EXL3 arm
/// serves those from packed trellis instead — see `load_moe_qwen4exp_exl3`).
///
/// `skip_routed_experts`: when true, routed experts get NULL weights (saves memory
/// when native FP8 MoE dispatch handles them). Shared expert is always loaded.
pub(crate) fn load_moe_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    skip_routed_experts: bool,
    force_all_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense_auto(store, &format!("{p}.gate.weight"), gpu)?;
    let shared_expert_gate = dense_auto(store, &format!("{p}.shared_expert_gate.weight"), gpu)?;

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let qctx = QuantizeCtx {
        absmax_k,
        quantize_k,
        stream,
    };

    // Qwen3.6-35B-A3B BF16 release ships a FUSED MoE layout: one
    // `experts.gate_up_proj: [num_experts, 2*inter, hidden]` and one
    // `experts.down_proj: [num_experts, hidden, inter]` per layer. Slice
    // each expert at load time and runtime-quantize to NVFP4.
    let fused_gate_up_key = format!("{p}.experts.gate_up_proj");
    let fused_down_key = format!("{p}.experts.down_proj");
    // FUSED expert layout: one `experts.gate_up_proj [E, 2*inter, h]` + one
    // `experts.down_proj [E, h, inter]` per layer, sliced per expert at load and
    // runtime-quantized to NVFP4. Two on-disk dtypes occur in the wild:
    //   - BF16 (Qwen3.6-35B-A3B BF16 release) → slice and quantize directly.
    //   - FP8E4M3 block-scaled (lovedheart AgentWorld-35B FP8: routed experts
    //     fused-FP8 with `*_scale_inv`, while attention/SSM/shared are BF16) →
    //     dequant each slice FP8→BF16 (reusing dequant_fp8_blockscaled_bf16)
    //     then quantize to NVFP4. Equivalent to the proven NVFP4 expert decode
    //     path (cf. ATLAS_FORCE_NVFP4_MOE), so no native-FP8 fused-shared kernel
    //     contract is involved. Detection is dtype-based, not variant-based, so
    //     it also covers a fused-BF16 layer inside a globally-FP8 checkpoint.
    let is_fused = store.contains(&fused_gate_up_key) && store.contains(&fused_down_key);
    let fused_is_fp8 = is_fused
        && store
            .get(&fused_gate_up_key)
            .map(|w| w.dtype == WeightDtype::FP8E4M3)
            .unwrap_or(false);

    let load_expert_fused = |expert_idx: usize| -> Result<ExpertWeight> {
        let fused_gu = store.get(&fused_gate_up_key)?;
        let fused_d = store.get(&fused_down_key)?;
        if fused_is_fp8 {
            // gate_up: [E, 2*inter, h] FP8 + gate_up_proj_scale_inv [E, sn, sk]
            // down:    [E, h, inter] FP8   + down_proj_scale_inv    [E, sn, sk]
            let gu_s = store.get(&format!("{fused_gate_up_key}_scale_inv"))?;
            let d_s = store.get(&format!("{fused_down_key}_scale_inv"))?;
            let (gu_sn, gu_sk) = (gu_s.shape[1], gu_s.shape[2]);
            let (d_sn, d_sk) = (d_s.shape[1], d_s.shape[2]);
            let gu_s_f32 = gu_s.dtype == WeightDtype::FP32;
            let d_s_f32 = d_s.dtype == WeightDtype::FP32;
            let gu_w_stride = 2 * inter * h; // FP8 = 1 byte/element
            let d_w_stride = h * inter;
            let gu_s_elem = if gu_s_f32 { 4 } else { 2 };
            let d_s_elem = if d_s_f32 { 4 } else { 2 };
            let gu_s_stride = gu_sn * gu_sk * gu_s_elem;
            let d_s_stride = d_sn * d_sk * d_s_elem;
            // Dequant the whole gate_up[e] [2*inter, h] FP8 → BF16, then slice
            // gate (rows 0..inter) and up (rows inter..2*inter).
            let gu_bf16 = dequant_fp8_block_slice_bf16(
                gpu,
                fused_gu.ptr.offset(expert_idx * gu_w_stride),
                gu_s.ptr.offset(expert_idx * gu_s_stride),
                2 * inter,
                h,
                gu_sn,
                gu_sk,
                gu_s_f32,
            )?;
            let down_bf16 = dequant_fp8_block_slice_bf16(
                gpu,
                fused_d.ptr.offset(expert_idx * d_w_stride),
                d_s.ptr.offset(expert_idx * d_s_stride),
                h,
                inter,
                d_sn,
                d_sk,
                d_s_f32,
            )?;
            let gate_dw = DenseWeight { weight: gu_bf16 };
            let up_dw = DenseWeight {
                weight: gu_bf16.offset(inter * h * 2), // BF16 = 2 bytes
            };
            let down_dw = DenseWeight { weight: down_bf16 };
            let out = ExpertWeight {
                gate_proj: quantize_to_nvfp4(
                    &gate_dw, inter, h, gpu, absmax_k, quantize_k, stream,
                )?,
                up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
                down_proj: quantize_to_nvfp4(
                    &down_dw, h, inter, gpu, absmax_k, quantize_k, stream,
                )?,
            };
            gpu.free(gu_bf16)?;
            gpu.free(down_bf16)?;
            Ok(out)
        } else {
            // BF16 fused: slice and quantize directly.
            let bf16 = 2usize;
            let gu_per_expert_bytes = 2 * inter * h * bf16;
            let d_per_expert_bytes = h * inter * bf16;
            let gate_off = expert_idx * gu_per_expert_bytes;
            let up_off = gate_off + inter * h * bf16;
            let down_off = expert_idx * d_per_expert_bytes;
            let gate_dw = DenseWeight {
                weight: fused_gu.ptr.offset(gate_off),
            };
            let up_dw = DenseWeight {
                weight: fused_gu.ptr.offset(up_off),
            };
            let down_dw = DenseWeight {
                weight: fused_d.ptr.offset(down_off),
            };
            Ok(ExpertWeight {
                gate_proj: quantize_to_nvfp4(
                    &gate_dw, inter, h, gpu, absmax_k, quantize_k, stream,
                )?,
                up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
                down_proj: quantize_to_nvfp4(
                    &down_dw, h, inter, gpu, absmax_k, quantize_k, stream,
                )?,
            })
        }
    };

    // Route every projection through `quantized_any` so the per-tensor BF16
    // fallback applies uniformly to shared and routed experts. Hybrid MoE
    // checkpoints (AgentWorld-35B, Qwen3.5-397B) ship the shared expert — and
    // occasionally individual routed experts — as unquantized BF16 even when
    // the model is globally FP8/NVFP4. Dispatching on the global `variant`
    // alone sent those tensors down the FP8/NVFP4 arm and failed with
    // "weight_scale_inv not found" before the fallback could catch them.
    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        Ok(ExpertWeight {
            gate_proj: quantized_any(
                store,
                &format!("{prefix}.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            up_proj: quantized_any(
                store,
                &format!("{prefix}.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            down_proj: quantized_any(
                store,
                &format!("{prefix}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?,
        })
    };

    let shared_expert = load_expert(&format!("{p}.shared_expert"))?;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !(force_all_experts || config.is_local_expert(e)) {
            experts.push(ExpertWeight::null());
        } else if is_fused {
            experts.push(load_expert_fused(e)?);
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
        }
    }

    // Fused layout (BF16 or FP8 source) shares ONE `experts.gate_up_proj` +
    // `experts.down_proj` tensor across all experts (sliced by offset), so it
    // can't be freed per-expert like the separate path. Free the shared source
    // here now that every expert has been quantized to NVFP4 — #200 only frees
    // the per-slice dequant intermediates, not these originals, so this is
    // additive (no double-free). Drops the redundant ~60GB so only the NVFP4
    // copies remain resident.
    if is_fused {
        if let Ok(w) = store.get(&fused_gate_up_key) {
            let _ = store.release_ptr(gpu, w.ptr);
        }
        if let Ok(w) = store.get(&fused_down_key) {
            let _ = store.release_ptr(gpu, w.ptr);
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

#[path = "ssm_qwen35_moe_variants.rs"]
mod ssm_qwen35_moe_variants;
pub use ssm_qwen35_moe_variants::*;
