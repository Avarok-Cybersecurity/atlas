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
    config: &avarok_core::config::ModelConfig,
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
    config: &avarok_core::config::ModelConfig,
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
///
/// Two FFN flavors are auto-detected:
/// - **MoE** (Qwen3.5-MoE / Qwen3.6-A3B): router `mtp.layers.0.mlp.gate.weight`
///   plus per-expert or stacked expert tensors and a shared expert.
/// - **Dense** (Qwen3.6-27B-FP8): single `mtp.layers.0.mlp.{gate,up,down}_proj.weight`
///   triple, no router or experts. Distinguished by absence of `.gate.weight`
///   and presence of `.gate_proj.weight` directly under `mlp`.
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
            // Standard & friends: usually BF16 on disk (dense_auto's BF16 arm
            // returns the raw pointer, identical to `dense`), but checkpoints
            // that quantize the MTP head too (e.g. centml modelopt W4A4:
            // mtp.fc/*_proj as packed NVFP4 + weight_scale/_2) route through
            // dense_auto's UInt8 arm for a one-time NVFP4->BF16 dequant.
            // Feeding the packed U8 pointer to the BF16 GEMV read 4x past the
            // allocation (cuMemsetD8Async status 700 on first decode).
            _ => dense_auto(store, name, gpu),
        }
    };

    // Dense FFN MTP head: triple of {gate,up,down}_proj directly under
    // `mtp.layers.0.mlp`, with no `.gate.weight` router. Short-circuit before
    // any MoE-shaped loads so a dense checkpoint doesn't trip on missing
    // `shared_expert.*` tensors.
    let dense_gate_proj = format!("{mlp}.gate_proj.weight");
    let moe_router = format!("{mlp}.gate.weight");
    if store.contains(&dense_gate_proj) && !store.contains(&moe_router) {
        let dense_ffn = DenseExpertWeight {
            gate_proj: load(&dense_gate_proj)?,
            up_proj: load(&format!("{mlp}.up_proj.weight"))?,
            down_proj: load(&format!("{mlp}.down_proj.weight"))?,
        };
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        return Ok(MtpWeights {
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
            moe_gate: null,
            shared_expert: DenseExpertWeight {
                gate_proj: null,
                up_proj: null,
                down_proj: null,
            },
            shared_expert_gate: null,
            experts: Vec::new(),
            dense_ffn: Some(dense_ffn),
            norm: dense(store, "mtp.norm.weight")?,
            fp8_experts: None,
        });
    }

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
    let stacked = store.contains(&stacked_gate_up) && store.contains(&stacked_down);
    let experts = if stacked {
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
    // Native FP8 tables ride along with the dequants (the head decides which
    // to keep — see `MtpWeights::fp8_experts`). Gated on the DEQUANTING
    // variants only: under `Bf16Raw` the `load` closure hands out store
    // pointers, and `release_bf16_expert_dequants` must never free those.
    let fp8_experts =
        if !stacked && variant != Nvfp4Variant::Bf16Raw && mtp_experts_native_fp8(store, mlp) {
            Some(load_mtp_fp8_experts(store, mlp, num_experts, gpu)?)
        } else {
            None
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
        dense_ffn: None,
        norm: dense(store, "mtp.norm.weight")?,
        fp8_experts,
    })
}

/// True when the MTP routed experts are stored per expert as FP8 E4M3 with a
/// `weight_scale_inv` block-scale sibling — the layout
/// `load_fp8_block_scaled_as_fp8weight` consumes and the native-FP8 main
/// layers already serve. Probes expert 0's `gate_proj`; a checkpoint that
/// mixes formats across experts fails loudly inside the per-expert load.
pub(crate) fn mtp_experts_native_fp8(store: &WeightStore, mlp: &str) -> bool {
    let weight = format!("{mlp}.experts.0.gate_proj.weight");
    store
        .get(&weight)
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false)
        && store.contains(&format!("{mlp}.experts.0.gate_proj.weight_scale_inv"))
}

/// The MTP routed + shared experts as FP8 block-scaled tables (FP8 bytes are
/// the store's; the FP32 block scales are widened into owned allocations).
fn load_mtp_fp8_experts(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<MtpFp8Experts> {
    let proj = |prefix: &str, name: &str| -> Result<Fp8Weight> {
        load_fp8_block_scaled_as_fp8weight(store, &format!("{prefix}.{name}"), gpu)
    };
    let load = |prefix: &str| -> Result<Fp8ExpertWeight> {
        Ok(Fp8ExpertWeight {
            gate_proj: proj(prefix, "gate_proj")?,
            up_proj: proj(prefix, "up_proj")?,
            down_proj: proj(prefix, "down_proj")?,
        })
    };
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        experts.push(
            load(&format!("{mlp}.experts.{e}"))
                .with_context(|| format!("MTP expert {e}: native FP8 tables"))?,
        );
    }
    let shared_expert =
        load(&format!("{mlp}.shared_expert")).context("MTP shared expert: native FP8 tables")?;
    Ok(MtpFp8Experts {
        experts,
        shared_expert,
    })
}

#[cfg(test)]
mod mtp_fp8_detect_tests {
    use super::*;
    use spark_runtime::weights::WeightTensor;
    use std::collections::HashMap;

    const MLP: &str = "mtp.layers.0.mlp";

    fn store(entries: Vec<(String, WeightDtype)>) -> WeightStore {
        let mut m = HashMap::new();
        for (name, dtype) in entries {
            m.insert(
                name,
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: vec![512, 2048],
                    dtype,
                },
            );
        }
        WeightStore::from_map(m)
    }

    fn gate0(suffix: &str) -> String {
        format!("{MLP}.experts.0.gate_proj.{suffix}")
    }

    #[test]
    fn fp8_weight_with_block_scale_is_native_fp8() {
        let s = store(vec![
            (gate0("weight"), WeightDtype::FP8E4M3),
            (gate0("weight_scale_inv"), WeightDtype::BF16),
        ]);
        assert!(mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn bf16_experts_are_not_native_fp8() {
        // A BF16 MTP head (Sehyo / RedHatAI NVFP4 checkpoints): the dequant
        // path is the only copy, so the tables must NOT be attached (and the
        // release must never run).
        let s = store(vec![(gate0("weight"), WeightDtype::BF16)]);
        assert!(!mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn fp8_without_a_block_scale_is_refused() {
        // FP8 bytes with no `weight_scale_inv` is not the block-scaled layout
        // the grouped kernels index (per-tensor FP8 has a different dequant).
        let s = store(vec![(gate0("weight"), WeightDtype::FP8E4M3)]);
        assert!(!mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn absent_experts_are_not_native_fp8() {
        assert!(!mtp_experts_native_fp8(&store(vec![]), MLP));
    }
}
