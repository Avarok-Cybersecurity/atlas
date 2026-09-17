// SPDX-License-Identifier: AGPL-3.0-only

//! The FP8-expert and no-shared-expert MoE loaders, split out of
//! `ssm_qwen35.rs` to keep it under the 500-line cap.

use super::*;

/// Load MoE experts as native FP8 weights (no NVFP4 conversion).
///
/// Returns the standard MoeWeights (with NVFP4 gate/shared for compatibility)
/// PLUS a Vec of Fp8ExpertWeight for native FP8 dispatch.
pub(crate) fn load_moe_qwen35_fp8_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &avarok_core::config::ModelConfig,
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
    config: &avarok_core::config::ModelConfig,
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
            weight_scale_2_vec: DevicePtr::NULL,
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
