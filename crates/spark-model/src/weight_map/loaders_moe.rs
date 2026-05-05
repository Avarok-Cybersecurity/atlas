// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

pub(super) fn load_moe_inner(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");
    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let shared_expert_gate = dense(store, &format!("{p}.shared_expert_gate.weight"))?;

    // Always load shared expert as NVFP4 — it's applied separately in MoE dispatch
    // and the FP8 fused batch kernel only handles routed experts.
    let shared_expert = ExpertWeight {
        gate_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.gate_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        up_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.up_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        down_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.down_proj"),
            h,
            inter,
            gpu,
            variant,
            qctx,
        )?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else {
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                up_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                down_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    h,
                    inter,
                    gpu,
                    variant,
                    qctx,
                )?,
            });
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

/// Load MoE weights for Mistral models (w1/w2/w3 naming convention).
/// w1 = gate_proj, w2 = down_proj, w3 = up_proj.
pub(crate) fn load_moe_mistral(
    store: &WeightStore,
    layer: usize,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<MoeWeights> {
    // Mistral weight names: layers.{i}.gate.weight, layers.{i}.experts.{e}.w1/w2/w3
    // No "model." prefix, no "feed_forward" or "block_sparse_moe" nesting.
    let p = format!("layers.{layer}");

    let gate = dense(store, &format!("{p}.gate.weight"))
        .or_else(|_| dense(store, &format!("model.layers.{layer}.gate.weight")))
        .context("Mistral: MoE gate weight not found")?;

    // Shared expert: layers.{i}.shared_experts.w1/w2/w3 (NVFP4 packed)
    // No shared_expert_gate in Mistral (the blend uses a fixed sigmoid)
    let se_prefix = format!("{p}.shared_experts");
    let shared_expert = if store.contains(&format!("{se_prefix}.w1.weight_packed")) {
        ExpertWeight {
            gate_proj: quantized_v2(store, &format!("{se_prefix}.w1"), gpu)
                .context("Mistral: shared expert w1")?,
            up_proj: quantized_v2(store, &format!("{se_prefix}.w3"), gpu)
                .context("Mistral: shared expert w3")?,
            down_proj: quantized_v2(store, &format!("{se_prefix}.w2"), gpu)
                .context("Mistral: shared expert w2")?,
        }
    } else {
        tracing::warn!("L{layer}: no shared expert weights, using NULL");
        ExpertWeight::null()
    };
    let shared_expert_gate = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            let gate_proj = quantized_v2(store, &format!("{ep}.w1"), gpu)
                .with_context(|| format!("Mistral expert {e}: w1 not found"))?;
            let up_proj = quantized_v2(store, &format!("{ep}.w3"), gpu)
                .with_context(|| format!("Mistral expert {e}: w3 not found"))?;
            let down_proj = quantized_v2(store, &format!("{ep}.w2"), gpu)
                .with_context(|| format!("Mistral expert {e}: w2 not found"))?;
            experts.push(ExpertWeight {
                gate_proj,
                up_proj,
                down_proj,
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

/// Load MTP head weights from the weight store.
///
/// MTP weights use `mtp.*` prefix. For NVFP4/BF16 models all are BF16 in safetensors.
/// For FP8 models, projections/experts are FP8 block-scaled → dequanted to BF16 here.
/// Quantization to NVFP4 happens in the weight_loader, not here.
pub(crate) fn load_mtp(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<MtpWeights> {
    let p = "mtp.layers.0.self_attn";
    let mlp = "mtp.layers.0.mlp";

    // For FP8: expert/projection weights are FP8, norms/gates are BF16.
    let load = |name: &str| -> Result<DenseWeight> {
        match variant {
            Nvfp4Variant::Fp8Dequanted => dense_auto(store, name, gpu),
            // Bf16Raw fine-tunes already store .weight as BF16; no dequant needed.
            Nvfp4Variant::Bf16Raw => dense(store, name),
            _ => dense(store, name),
        }
    };

    let shared_expert = DenseExpertWeight {
        gate_proj: load(&format!("{mlp}.shared_expert.gate_proj.weight"))?,
        up_proj: load(&format!("{mlp}.shared_expert.up_proj.weight"))?,
        down_proj: load(&format!("{mlp}.shared_expert.down_proj.weight"))?,
    };

    // Two on-disk MoE expert layouts exist for MTP heads:
    //
    // (A) Per-expert split (Sehyo / nvidia / standard compressed-tensors,
    //     and the Qwen FP8 main-model checkpoints):
    //         mtp.layers.0.mlp.experts.{e}.gate_proj.weight   [I, H]
    //         mtp.layers.0.mlp.experts.{e}.up_proj.weight     [I, H]
    //         mtp.layers.0.mlp.experts.{e}.down_proj.weight   [H, I]
    //
    // (B) Stacked + fused (RedHatAI llm-compressor — e.g.
    //     RedHatAI/Qwen3.6-35B-A3B-NVFP4):
    //         mtp.layers.0.mlp.experts.gate_up_proj   [E, 2*I, H] BF16
    //         mtp.layers.0.mlp.experts.down_proj      [E,   H, I] BF16
    //     The 2*I dim packs gate then up along axis 1 (gate = first half).
    //
    // For (B) we hand back per-expert `DenseWeight`s as DevicePtr offsets
    // into the shared stacked tensor — zero-copy, no extra GPU allocation.
    // The downstream MoE kernel doesn't care that the underlying memory is
    // contiguous across experts; it only reads each expert's weight as a
    // `[I, H]` (gate/up) or `[H, I]` (down) BF16 matrix.
    let stacked_gate_up = format!("{mlp}.experts.gate_up_proj");
    let stacked_down = format!("{mlp}.experts.down_proj");
    let experts = if store.contains(&stacked_gate_up) && store.contains(&stacked_down) {
        load_mtp_experts_stacked(store, mlp, num_experts)?
    } else {
        let mut v = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            v.push(DenseExpertWeight {
                gate_proj: load(&format!("{mlp}.experts.{e}.gate_proj.weight"))?,
                up_proj: load(&format!("{mlp}.experts.{e}.up_proj.weight"))?,
                down_proj: load(&format!("{mlp}.experts.{e}.down_proj.weight"))?,
            });
        }
        v
    };

    Ok(MtpWeights {
        pre_fc_norm_embedding: dense(store, "mtp.pre_fc_norm_embedding.weight")?,
        pre_fc_norm_hidden: dense(store, "mtp.pre_fc_norm_hidden.weight")?,
        fc: load("mtp.fc.weight")?,
        input_layernorm: dense(store, "mtp.layers.0.input_layernorm.weight")?,
        q_proj: load(&format!("{p}.q_proj.weight"))?,
        k_proj: load(&format!("{p}.k_proj.weight"))?,
        v_proj: load(&format!("{p}.v_proj.weight"))?,
        o_proj: load(&format!("{p}.o_proj.weight"))?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        post_attn_layernorm: dense(store, "mtp.layers.0.post_attention_layernorm.weight")?,
        moe_gate: dense(store, &format!("{mlp}.gate.weight"))?,
        shared_expert,
        shared_expert_gate: dense(store, &format!("{mlp}.shared_expert_gate.weight"))?,
        experts,
        norm: dense(store, "mtp.norm.weight")?,
    })
}
