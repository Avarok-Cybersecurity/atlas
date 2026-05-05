// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Nemotron-H Mamba-2 SSM weights.
///
/// in_proj produces [z(d_inner), x(d_inner), B(n_groups*state), C(n_groups*state), dt(num_heads)].
pub struct NemotronSsmWeights {
    /// in_proj: [in_proj_size, hidden_size] NVFP4.
    pub in_proj: QuantizedWeight,
    /// out_proj: `[hidden_size, d_inner]` NVFP4.
    pub out_proj: QuantizedWeight,
    /// conv1d weight: `[d_xBC, 1, conv_kernel]` BF16.
    pub conv1d_weight: DenseWeight,
    /// conv1d bias: `[d_xBC]` BF16.
    pub conv1d_bias: DenseWeight,
    /// A_log: `[mamba_num_heads]` BF16 (cast to FP32 at runtime).
    pub a_log: DenseWeight,
    /// D skip-connection: `[mamba_num_heads]` BF16.
    pub d_param: DenseWeight,
    /// dt_bias: `[mamba_num_heads]` BF16.
    pub dt_bias: DenseWeight,
    /// SSM internal norm: `[d_inner]` BF16 (applied to y before gating with z).
    pub ssm_norm: DenseWeight,
}

/// Nemotron-H 2-projection expert (up_proj + relu² + down_proj, no gate_proj).
#[derive(Debug, Clone, Copy)]
pub struct NemotronExpertWeight {
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

impl NemotronExpertWeight {
    pub fn null() -> Self {
        Self {
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
        }
    }
}

/// Nemotron-H MoE layer weights.
pub struct NemotronMoeWeights {
    /// Router gate: [num_experts, hidden_size] F32→BF16.
    pub gate: DenseWeight,
    /// Expert score correction bias: `[num_experts]` F32.
    pub e_score_correction_bias: DenseWeight,
    /// Per-expert weights (routed): NVFP4.
    pub experts: Vec<NemotronExpertWeight>,
    /// Shared expert up_proj: [shared_inter, hidden_size] NVFP4.
    pub shared_up: QuantizedWeight,
    /// Shared expert down_proj: [hidden_size, shared_inter] NVFP4.
    pub shared_down: QuantizedWeight,
    /// LatentMoE: fc1 [moe_latent_size, hidden_size] BF16 (dequant from FP8 at load).
    /// Present only for Super 120B (moe_latent_size > 0).
    pub fc1_latent_proj: Option<DenseWeight>,
    /// LatentMoE: fc2 [hidden_size, moe_latent_size] BF16.
    /// Present only for Super 120B (moe_latent_size > 0).
    pub fc2_latent_proj: Option<DenseWeight>,
}

/// SSM weight quantization format detected at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronSsmQuant {
    /// NVFP4: has weight_scale + weight_scale_2 (Nano NVFP4 model).
    Nvfp4,
    /// FP8 E4M3: has weight_scale but no weight_scale_2 (Super 120B).
    Fp8,
    /// BF16: no weight_scale at all (BF16 layers adjacent to attention).
    Bf16,
}

/// Load Nemotron-H Mamba-2 SSM weights.
///
/// Mixed quantization: NVFP4, FP8, or BF16 depending on layer and model.
/// Non-NVFP4 projections use QuantizedWeight::null() — caller runtime-quantizes.
pub(crate) fn load_nemotron_ssm(
    store: &WeightStore,
    _layer: usize,
    gpu: &dyn GpuBackend,
    layer_prefix: &str,
) -> Result<(NemotronSsmWeights, NemotronSsmQuant)> {
    let p = format!("{layer_prefix}.mixer");
    let has_scale = store.contains(&format!("{p}.in_proj.weight_scale"));
    let has_scale2 = store.contains(&format!("{p}.in_proj.weight_scale_2"));
    let quant = if has_scale && has_scale2 {
        NemotronSsmQuant::Nvfp4
    } else if has_scale {
        NemotronSsmQuant::Fp8
    } else {
        NemotronSsmQuant::Bf16
    };
    let in_proj = if quant == NemotronSsmQuant::Nvfp4 {
        quantized(store, &format!("{p}.in_proj"), gpu)?
    } else {
        QuantizedWeight::null()
    };
    let out_proj = if quant == NemotronSsmQuant::Nvfp4 {
        quantized(store, &format!("{p}.out_proj"), gpu)?
    } else {
        QuantizedWeight::null()
    };
    // A_log, D, dt_bias, conv1d.bias are BF16 in safetensors but consumed as FP32 by kernels.
    Ok((
        NemotronSsmWeights {
            in_proj,
            out_proj,
            conv1d_weight: dense(store, &format!("{p}.conv1d.weight"))?,
            conv1d_bias: dense_bf16_as_f32(store, &format!("{p}.conv1d.bias"), gpu)?,
            a_log: dense_bf16_as_f32(store, &format!("{p}.A_log"), gpu)?,
            d_param: dense_bf16_as_f32(store, &format!("{p}.D"), gpu)?,
            dt_bias: dense_bf16_as_f32(store, &format!("{p}.dt_bias"), gpu)?,
            ssm_norm: dense(store, &format!("{p}.norm.weight"))?,
        },
        quant,
    ))
}

/// Load Nemotron-H attention weights.
///
/// Standard GQA: 32 Q heads, 2 KV heads, head_dim=128.
/// Mixed quantization: some attention layers are NVFP4, some BF16.
/// BF16 layers return dense Q/K/V in AttentionWeights + null QuantizedWeights.
pub(crate) fn load_nemotron_attention(
    store: &WeightStore,
    layer: usize,
    gpu: &dyn GpuBackend,
    layer_prefix: &str,
) -> Result<(
    AttentionWeights,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
    DenseWeight,
    bool,
)> {
    let p = format!("{layer_prefix}.mixer");
    // Distinguish NVFP4 (has weight_scale_2) from FP8 (has weight_scale only) from BF16 (neither).
    let is_nvfp4 = store.contains(&format!("{p}.q_proj.weight_scale_2"));
    let is_fp8 = !is_nvfp4 && store.contains(&format!("{p}.q_proj.weight_scale"));
    let dummy = DenseWeight {
        weight: DevicePtr::NULL,
    };

    let (q_dense, k_dense, v_dense, o_dense, o_proj, q_nvfp4, k_nvfp4, v_nvfp4) = if is_nvfp4 {
        let q = quantized(store, &format!("{p}.q_proj"), gpu)?;
        let k = quantized(store, &format!("{p}.k_proj"), gpu)?;
        let v = quantized(store, &format!("{p}.v_proj"), gpu)?;
        let o = quantized(store, &format!("{p}.o_proj"), gpu)?;
        (dummy, dummy, dummy, dummy, o, Some(q), Some(k), Some(v))
    } else {
        // FP8 or BF16 — dequant to BF16 dense, caller will quantize to NVFP4.
        let load_proj = |name: &str| -> Result<DenseWeight> {
            let prefix = format!("{p}.{name}");
            if store.contains(&format!("{prefix}.weight_scale")) {
                dequant_fp8_to_bf16(store, &prefix, gpu)
            } else {
                dense(store, &format!("{prefix}.weight"))
            }
        };
        let q = load_proj("q_proj")?;
        let k = load_proj("k_proj")?;
        let v = load_proj("v_proj")?;
        let o = load_proj("o_proj")?;
        if is_fp8 && layer < 2 {
            tracing::info!("L{layer} Attention: FP8 → BF16 (runtime quantization to NVFP4)");
        }
        (q, k, v, o, QuantizedWeight::null(), None, None, None)
    };

    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj: q_dense,
        k_proj: k_dense,
        v_proj: v_dense,
        o_proj,
        q_norm: dummy,
        k_norm: dummy,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    Ok((attn, q_nvfp4, k_nvfp4, v_nvfp4, o_dense, is_nvfp4))
}

/// Load Nemotron-H MoE weights.
///
/// Handles both Nano (all NVFP4) and Super 120B (mixed FP8/NVFP4/BF16 + LatentMoE).
/// For FP8 shared_up and fc1_latent_proj, dequants to BF16 then runtime-quantizes to NVFP4.
pub(crate) fn load_nemotron_moe(
    store: &WeightStore,
    layer: usize,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    absmax_k: Option<spark_runtime::gpu::KernelHandle>,
    quantize_k: Option<spark_runtime::gpu::KernelHandle>,
    stream: u64,
    scratch: Option<DevicePtr>,
    layer_prefix: &str,
) -> Result<NemotronMoeWeights> {
    let p = format!("{layer_prefix}.mixer");
    // Gate weight: F32 in Nano 30B, BF16 in Super 120B. Convert to BF16 if needed.
    // The gate GEMV kernel expects BF16 input.
    let gate_name = format!("{p}.gate.weight");
    let gate_w = store.get(&gate_name)?;
    let gate = if gate_w.dtype == WeightDtype::FP32 {
        dense_f32_as_bf16(store, &gate_name, gpu)?
    } else {
        DenseWeight { weight: gate_w.ptr }
    };
    // e_score_correction_bias stays F32 — bias_add_bf16_f32 kernel consumes F32 bias.
    let e_score_correction_bias = dense(store, &format!("{p}.gate.e_score_correction_bias"))?;

    // Shared expert: detect FP8 vs NVFP4 by presence of weight_scale_2.
    let shared_up_prefix = format!("{p}.shared_experts.up_proj");
    let shared_up_has_s2 = store.contains(&format!("{shared_up_prefix}.weight_scale_2"));
    let shared_up_has_s = store.contains(&format!("{shared_up_prefix}.weight_scale"));
    let shared_up = if shared_up_has_s2 {
        quantized(store, &shared_up_prefix, gpu)?
    } else {
        // FP8 or BF16 — dequant to scratch, then quantize to NVFP4
        let bf16 = if shared_up_has_s {
            if let Some(s) = scratch {
                dequant_fp8_to_bf16_into(store, &shared_up_prefix, gpu, s)?
            } else {
                dequant_fp8_to_bf16(store, &shared_up_prefix, gpu)?
            }
        } else {
            dense(store, &format!("{shared_up_prefix}.weight"))?
        };
        quantize_to_nvfp4(
            &bf16,
            config.shared_expert_intermediate_size,
            config.hidden_size,
            gpu,
            absmax_k.unwrap(),
            quantize_k.unwrap(),
            stream,
        )?
    };

    let shared_down_prefix = format!("{p}.shared_experts.down_proj");
    let shared_down_has_s2 = store.contains(&format!("{shared_down_prefix}.weight_scale_2"));
    let shared_down_has_s = store.contains(&format!("{shared_down_prefix}.weight_scale"));
    let shared_down = if shared_down_has_s2 {
        quantized(store, &shared_down_prefix, gpu)?
    } else {
        let bf16 = if shared_down_has_s {
            if let Some(s) = scratch {
                dequant_fp8_to_bf16_into(store, &shared_down_prefix, gpu, s)?
            } else {
                dequant_fp8_to_bf16(store, &shared_down_prefix, gpu)?
            }
        } else {
            dense(store, &format!("{shared_down_prefix}.weight"))?
        };
        quantize_to_nvfp4(
            &bf16,
            config.hidden_size,
            config.shared_expert_intermediate_size,
            gpu,
            absmax_k.unwrap(),
            quantize_k.unwrap(),
            stream,
        )?
    };

    // LatentMoE projections (Super 120B only).
    // fc1 persists as BF16 — must allocate (not scratch).
    let (fc1_latent_proj, fc2_latent_proj) = if config.moe_latent_size > 0 {
        let fc1_prefix = format!("{p}.fc1_latent_proj");
        let fc1 = if store.contains(&format!("{fc1_prefix}.weight_scale")) {
            dequant_fp8_to_bf16(store, &fc1_prefix, gpu)?
        } else {
            dense(store, &format!("{fc1_prefix}.weight"))?
        };
        let fc2_prefix = format!("{p}.fc2_latent_proj");
        let fc2 = if store.contains(&format!("{fc2_prefix}.weight_scale")) {
            dequant_fp8_to_bf16(store, &fc2_prefix, gpu)?
        } else {
            dense(store, &format!("{fc2_prefix}.weight"))?
        };
        (Some(fc1), Some(fc2))
    } else {
        (None, None)
    };

    // Routed experts: detect NVFP4 vs FP8 vs BF16 from first local expert.
    let moe_input = config.moe_input_size();
    let moe_inter = config.moe_intermediate_size;
    let first_local = (0..num_experts).find(|e| config.is_local_expert(*e));
    let experts_are_nvfp4 = first_local
        .is_none_or(|e| store.contains(&format!("{p}.experts.{e}.up_proj.weight_scale_2")));
    let experts_are_fp8 = !experts_are_nvfp4
        && first_local
            .is_some_and(|e| store.contains(&format!("{p}.experts.{e}.up_proj.weight_scale")));
    if !experts_are_nvfp4 && layer < 2 {
        tracing::info!(
            "L{layer} MoE experts: {} → NVFP4 (runtime quantization, {} experts)",
            if experts_are_fp8 { "FP8" } else { "BF16" },
            num_experts,
        );
    }

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let up_prefix = format!("{p}.experts.{e}.up_proj");
            let down_prefix = format!("{p}.experts.{e}.down_proj");
            let (up_proj, down_proj) = if experts_are_nvfp4 {
                (
                    quantized(store, &up_prefix, gpu)?,
                    quantized(store, &down_prefix, gpu)?,
                )
            } else {
                let up_bf16 = if experts_are_fp8 {
                    if let Some(s) = scratch {
                        dequant_fp8_to_bf16_into(store, &up_prefix, gpu, s)?
                    } else {
                        dequant_fp8_to_bf16(store, &up_prefix, gpu)?
                    }
                } else {
                    dense(store, &format!("{up_prefix}.weight"))?
                };
                let up = quantize_to_nvfp4(
                    &up_bf16,
                    moe_inter,
                    moe_input,
                    gpu,
                    absmax_k.unwrap(),
                    quantize_k.unwrap(),
                    stream,
                )?;
                let down_bf16 = if experts_are_fp8 {
                    if let Some(s) = scratch {
                        dequant_fp8_to_bf16_into(store, &down_prefix, gpu, s)?
                    } else {
                        dequant_fp8_to_bf16(store, &down_prefix, gpu)?
                    }
                } else {
                    dense(store, &format!("{down_prefix}.weight"))?
                };
                let down = quantize_to_nvfp4(
                    &down_bf16,
                    moe_input,
                    moe_inter,
                    gpu,
                    absmax_k.unwrap(),
                    quantize_k.unwrap(),
                    stream,
                )?;
                (up, down)
            };
            experts.push(NemotronExpertWeight { up_proj, down_proj });
        } else {
            experts.push(NemotronExpertWeight::null());
        }
    }

    Ok(NemotronMoeWeights {
        gate,
        e_score_correction_bias,
        experts,
        shared_up,
        shared_down,
        fc1_latent_proj,
        fc2_latent_proj,
    })
}
