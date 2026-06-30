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
    // Kept for loader-dispatch signature parity; `dense_auto` routes by the
    // projection's actual on-disk dtype rather than the model-wide variant.
    _variant: Nvfp4Variant,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.linear_attn");

    // For FP8 models: in_proj_qkv, in_proj_z, out_proj are FP8 block-scaled.
    // conv1d, in_proj_a, in_proj_b are BF16 (in modules_to_not_convert).
    let load_proj = |name: &str| -> Result<DenseWeight> { dense_auto(store, name, gpu) };

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

/// Dequant a block-scaled FP8 weight to a fresh BF16 device buffer, given
/// device pointers (not store keys). Mirrors `dequant_fp8_blockscaled_to_bf16`
/// but operates on an aliased slice of a *fused* expert tensor
/// (`experts.gate_up_proj` / `experts.down_proj`), where per-expert weights
/// cannot be addressed by name. Caller owns and frees the returned buffer.
#[allow(clippy::too_many_arguments)]
fn dequant_fp8_block_slice_bf16(
    gpu: &dyn GpuBackend,
    weight_ptr: DevicePtr,
    scale_ptr: DevicePtr,
    n: usize,
    k: usize,
    sn: usize,
    sk: usize,
    scale_is_f32: bool,
) -> Result<DevicePtr> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let out = gpu.alloc(n * k * 2)?; // BF16 = 2 bytes/element
    let block_n = (n / sn) as u32;
    let block_k = (k / sk) as u32;
    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(weight_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk as u32)
        .arg_u32(scale_is_f32 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
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
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else if is_fused {
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
            // Remote-expert placeholder: NULL pointers never dereferenced.
            // `Fp8BlockScaled` chosen as the format tag because that's the
            // dominant disk format for Qwen FP8 checkpoints — keeps the
            // tag consistent with what the routed expert would carry if
            // it weren't remote.
            let null_block = Fp8Weight {
                weight: DevicePtr::NULL,
                row_scale: DevicePtr::NULL,
                n: 0,
                k: 0,
                scale_format: WeightQuantFormat::Fp8BlockScaled,
            };
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: null_block,
                up_proj: null_block,
                down_proj: null_block,
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
