// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

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
    let p = format!("{layer_prefix}.linear_attn");

    // For FP8 models: in_proj_qkv, in_proj_z, out_proj are FP8 block-scaled.
    // conv1d, in_proj_a, in_proj_b are BF16 (in modules_to_not_convert).
    let load_proj = |name: &str| -> Result<DenseWeight> {
        match variant {
            Nvfp4Variant::Fp8Dequanted => dense_auto(store, name, gpu),
            _ => dense(store, name),
        }
    };

    Ok(SsmWeightsQwen35 {
        in_proj_qkv: load_proj(&format!("{p}.in_proj_qkv.weight"))?,
        in_proj_z: load_proj(&format!("{p}.in_proj_z.weight"))?,
        in_proj_a: dense(store, &format!("{p}.in_proj_a.weight"))?,
        in_proj_b: dense(store, &format!("{p}.in_proj_b.weight"))?,
        conv1d: dense(store, &format!("{p}.conv1d.weight"))?,
        // A_log and dt_bias MUST be FP32 — BF16 precision causes exponential
        // error amplification in the GDR decay gate at 8k+ tokens.
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        // norm.weight is safe as BF16 (no recurrent amplification)
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: load_proj(&format!("{p}.out_proj.weight"))?,
    })
}

/// Load MoE weights for Qwen3.5, auto-selecting NVFP4 naming convention.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
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
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let shared_expert_gate = dense(store, &format!("{p}.shared_expert_gate.weight"))?;

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let load_bf16_then_nvfp4 = |full_prefix: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let bf16 = dense(store, &format!("{full_prefix}.weight"))?;
        quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
    };

    // Runtime-quant context for the `quantized_any` BF16 fallback used when a
    // shared/routed expert ships as raw BF16 (no weight_scale) per the
    // checkpoint's ModelOpt ignore list (issue #97).
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
    let is_fused_bf16 = variant == Nvfp4Variant::Bf16Raw
        && store.contains(&fused_gate_up_key)
        && store.contains(&fused_down_key);

    let load_expert_fused = |expert_idx: usize| -> Result<ExpertWeight> {
        // gate_up: [num_experts, 2*inter, hidden] BF16
        let fused_gu = store.get(&fused_gate_up_key)?;
        // down: [num_experts, hidden, inter] BF16
        let fused_d = store.get(&fused_down_key)?;
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
            gate_proj: quantize_to_nvfp4(&gate_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
            up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
            down_proj: quantize_to_nvfp4(&down_dw, h, inter, gpu, absmax_k, quantize_k, stream)?,
        })
    };

    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        match variant {
            Nvfp4Variant::Bf16Raw => Ok(ExpertWeight {
                gate_proj: load_bf16_then_nvfp4(&format!("{prefix}.gate_proj"), inter, h)?,
                up_proj: load_bf16_then_nvfp4(&format!("{prefix}.up_proj"), inter, h)?,
                down_proj: load_bf16_then_nvfp4(&format!("{prefix}.down_proj"), h, inter)?,
            }),
            Nvfp4Variant::Fp8Dequanted => Ok(ExpertWeight {
                gate_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                up_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.up_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                down_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.down_proj"),
                    h,
                    inter,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
            }),
            // `quantized_any` (not `quantized_auto`): some NVFP4 checkpoints
            // keep the shared expert in BF16 via the ModelOpt quant_config
            // `ignore` list (e.g. Qwen3.5-397B-A17B-NVFP4, whose 184-entry
            // ignore list covers `*.mlp.shared_expert.*`), so no weight_scale
            // ships for it. `quantized_any`'s per-key `has_only_dense` fallback
            // runtime-quantizes the raw BF16 — the same path the routed experts
            // already take — instead of hard-failing on the absent scale
            // (issue #97). For checkpoints whose shared expert IS quantized
            // (35B/122B), the scale is present so this is identical to before.
            _ => Ok(ExpertWeight {
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
            }),
        }
    };

    let shared_expert = load_expert(&format!("{p}.shared_expert"))?;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else if is_fused_bf16 {
            experts.push(load_expert_fused(e)?);
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
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

/// Load MoE experts as native FP8 weights (no NVFP4 conversion).
///
/// Returns the standard MoeWeights (with NVFP4 gate/shared for compatibility)
/// PLUS a Vec of Fp8ExpertWeight for native FP8 dispatch.
pub(crate) fn load_moe_qwen35_fp8_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<Vec<Fp8ExpertWeight>> {
    let p = format!("{layer_prefix}.mlp");
    let mut fp8_experts = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.gate_proj"),
                    gpu,
                )?,
                up_proj: load_fp8_block_scaled_as_fp8weight(store, &format!("{ep}.up_proj"), gpu)?,
                down_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.down_proj"),
                    gpu,
                )?,
            });
        } else {
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                up_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                down_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
            });
        }
    }

    // Also load shared expert as FP8
    let shared_prefix = format!("{p}.shared_expert");
    let _shared_fp8 = Fp8ExpertWeight {
        gate_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.gate_proj"),
            gpu,
        )?,
        up_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.up_proj"),
            gpu,
        )?,
        down_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.down_proj"),
            gpu,
        )?,
    };

    Ok(fp8_experts)
}

/// Load MoE weights for models without shared experts (e.g. Qwen3-VL).
///
/// Creates zero-filled dummy shared expert weights so the fused MoE kernels
/// (which always launch top_k+1 blocks) produce zero contribution from the
/// shared expert slot. `weight_scale_2 = 0.0` ensures dequant → 0.
pub(crate) fn load_moe_no_shared(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;

    // Allocate correctly-sized zero-filled GPU buffers for dummy shared expert.
    // The fused kernel always runs a shared expert block (blockIdx.y == top_k),
    // which reads full expert-sized weight matrices. Buffers must match real
    // expert dimensions or the kernel will read out of bounds (CUDA error 900).
    // weight_scale_2 = 0.0 ensures dequant → 0 regardless of packed contents.
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize; // NVFP4 quantization group size (matches kernel GROUP_SIZE)

    // gate_proj/up_proj: [inter, h] → packed = inter * h / 2, scale = inter * (h / group_size)
    let gu_packed_bytes = inter * h / 2;
    let gu_scale_bytes = inter * (h / group_size);
    // down_proj: [h, inter] → packed = h * inter / 2, scale = h * (inter / group_size)
    let d_packed_bytes = h * inter / 2;
    let d_scale_bytes = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };

    let mk_zero_quant = |packed_sz: usize, scale_sz: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed_sz)?,
            weight_scale: alloc_zero(scale_sz)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        })
    };

    let shared_expert = ExpertWeight {
        gate_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        up_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        down_proj: mk_zero_quant(d_packed_bytes, d_scale_bytes)?,
    };
    // Gate weight for shared expert: zero BF16 [hidden_size] → sigmoid(0)=0.5.
    // Doesn't matter since shared_out is all zeros (0.5 * 0 = 0).
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    gpu,
                    variant,
                )?,
                up_proj: quantized_auto(store, &format!("{p}.experts.{e}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    gpu,
                    variant,
                )?,
            });
        } else {
            experts.push(ExpertWeight::null());
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
