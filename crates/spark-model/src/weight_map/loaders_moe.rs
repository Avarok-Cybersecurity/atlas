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
///
/// Two FFN flavors are auto-detected:
/// - **MoE** (Qwen3.5-MoE / Qwen3.6-A3B): router `mtp.layers.0.mlp.gate.weight`
///   plus per-expert or stacked expert tensors and a shared expert.
/// - **Dense** (Qwen3.6-27B-FP8): single `mtp.layers.0.mlp.{gate,up,down}_proj.weight`
///   triple, no router or experts. Distinguished by absence of `.gate.weight`
///   and presence of `.gate_proj.weight` directly under `mlp`.
// Audit every `mtp.*` tensor at load time. Env-gated behind
// `ATLAS_MTP_WEIGHT_AUDIT=1`. Catches the most-cited MTP weight-loader
// failure pattern (silent BF16 placeholder when an FP8 scale or weight
// is missing) by logging each tensor's name/shape/dtype and a sampled
// L2 norm. A zero/NaN sample L2 indicates the loader silently kept a
// placeholder.
fn audit_mtp_tensors(store: &WeightStore, gpu: &dyn GpuBackend) {
    if std::env::var("ATLAS_MTP_WEIGHT_AUDIT").ok().as_deref() != Some("1") {
        return;
    }
    let mut names: Vec<&str> = store.names().filter(|n| n.starts_with("mtp.")).collect();
    names.sort_unstable();
    tracing::info!("MTP_WEIGHT_AUDIT: scanning {} tensors", names.len());
    for name in names {
        let Ok(w) = store.get(name) else {
            tracing::warn!("MTP_WEIGHT_AUDIT name='{name}' status=missing");
            continue;
        };
        // Sample up to first 4096 elements — enough to catch zero/NaN/identity.
        let elem_bytes = w.dtype.byte_size();
        let total_elems = w.num_elements();
        let sample_elems = total_elems.min(4096);
        let sample_bytes = sample_elems * elem_bytes;
        let mut buf = vec![0u8; sample_bytes];
        if let Err(e) = gpu.copy_d2h(w.ptr, &mut buf) {
            tracing::warn!(
                "MTP_WEIGHT_AUDIT name='{name}' dtype={:?} shape={:?} read_err={e}",
                w.dtype, w.shape
            );
            continue;
        }
        let (l2_sq, n_nan, n_zero) = match w.dtype {
            spark_runtime::weights::WeightDtype::BF16 => {
                let mut s = 0.0f64;
                let mut nan = 0usize;
                let mut zero = 0usize;
                for chunk in buf.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f = f32::from_bits((bits as u32) << 16);
                    if f.is_nan() {
                        nan += 1;
                    } else {
                        if f == 0.0 { zero += 1; }
                        s += (f as f64) * (f as f64);
                    }
                }
                (s.sqrt(), nan, zero)
            }
            spark_runtime::weights::WeightDtype::FP32 => {
                let mut s = 0.0f64;
                let mut nan = 0usize;
                let mut zero = 0usize;
                for chunk in buf.chunks_exact(4) {
                    let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if f.is_nan() { nan += 1; }
                    else {
                        if f == 0.0 { zero += 1; }
                        s += (f as f64) * (f as f64);
                    }
                }
                (s.sqrt(), nan, zero)
            }
            spark_runtime::weights::WeightDtype::FP8E4M3 => {
                // E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7.
                // Decode each byte to f32 for L2.
                let mut s = 0.0f64;
                let mut nan = 0usize;
                let mut zero = 0usize;
                for &b in buf.iter() {
                    if b == 0 || b == 0x80 { zero += 1; continue; }
                    let sign = (b >> 7) & 1;
                    let exp = ((b >> 3) & 0x0F) as i32;
                    let mant = (b & 0x07) as i32;
                    if exp == 0x0F && mant == 0x07 {
                        nan += 1;
                        continue;
                    }
                    let val = if exp == 0 {
                        // subnormal: 2^(1-7) * (mant/8)
                        (mant as f32) / 8.0 * f32::powi(2.0, -6)
                    } else {
                        let m = 1.0 + (mant as f32) / 8.0;
                        m * f32::powi(2.0, exp - 7)
                    };
                    let v = if sign == 1 { -val } else { val };
                    s += (v as f64) * (v as f64);
                }
                (s.sqrt(), nan, zero)
            }
            spark_runtime::weights::WeightDtype::UInt8 => {
                let mut s = 0u64;
                let mut zero = 0usize;
                for &b in buf.iter() {
                    if b == 0 { zero += 1; }
                    s += (b as u64) * (b as u64);
                }
                ((s as f64).sqrt(), 0, zero)
            }
        };
        let zero_frac = (n_zero as f64) / (sample_elems.max(1) as f64);
        let status = if n_nan > 0 {
            "NAN"
        } else if l2_sq == 0.0 {
            "ZERO"
        } else if zero_frac > 0.99 {
            "PLACEHOLDER?"
        } else {
            "ok"
        };
        tracing::info!(
            "MTP_WEIGHT_AUDIT name='{name}' dtype={:?} shape={:?} sample_l2={:.4} sample_n={} n_nan={} n_zero={} status={}",
            w.dtype, w.shape, l2_sq, sample_elems, n_nan, n_zero, status,
        );
    }
    // Pairing check: every FP8 .weight should have a matching .weight_scale_inv.
    let mut unpaired: Vec<String> = Vec::new();
    for name in store.names() {
        if !name.starts_with("mtp.") || !name.ends_with(".weight") {
            continue;
        }
        let Ok(w) = store.get(name) else { continue };
        if !matches!(w.dtype, spark_runtime::weights::WeightDtype::FP8E4M3) {
            continue;
        }
        let scale_name = format!("{}_scale_inv", name);
        if !store.contains(&scale_name) {
            unpaired.push(name.to_string());
        }
    }
    if unpaired.is_empty() {
        tracing::info!("MTP_WEIGHT_AUDIT pairing_check ok (all FP8 tensors have scale_inv)");
    } else {
        for n in &unpaired {
            tracing::warn!("MTP_WEIGHT_AUDIT unpaired_fp8 name='{n}' missing='{n}_scale_inv'");
        }
    }
}

pub(crate) fn load_mtp(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<MtpWeights> {
    audit_mtp_tensors(store, gpu);
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
        // Detect native FP8 block-scaled projections on disk and load
        // them at native precision (no BF16 round-trip). Marker:
        // `{self_attn}.q_proj.weight_scale_inv` exists ⇒ FP8
        // block-scaled. `mtp.fc.weight` is BF16 in the upstream
        // Qwen3.6-FP8 MTP checkpoints (it's the concat projection,
        // not part of the transformer block), so the marker uses
        // q_proj instead.
        let q_proj_scale_key = format!("{p}.q_proj.weight_scale_inv");
        let native_fp8 = if store.contains(&q_proj_scale_key) {
            tracing::info!(
                "MTP head: detected native FP8 block-scaled projections; loading zero-copy at FP8 precision"
            );
            Some(MtpNativeFp8Projections {
                q_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{p}.q_proj"),
                    gpu,
                )?,
                k_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{p}.k_proj"),
                    gpu,
                )?,
                v_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{p}.v_proj"),
                    gpu,
                )?,
                o_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{p}.o_proj"),
                    gpu,
                )?,
                dense_ffn: Some(MtpNativeFp8DenseFfn {
                    gate_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{mlp}.gate_proj"),
                        gpu,
                    )?,
                    up_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{mlp}.up_proj"),
                        gpu,
                    )?,
                    down_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{mlp}.down_proj"),
                        gpu,
                    )?,
                }),
            })
        } else {
            None
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
            native_fp8,
            norm: dense(store, "mtp.norm.weight")?,
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
        dense_ffn: None,
        // MoE MTP heads don't yet have a native-FP8 fast path (the
        // FP8 block-scaled MoE expert plumbing for the MTP head is
        // a future cleanup — none of the current FP8 MTP checkpoints
        // ship as MoE). Falls back to the existing runtime-quantize
        // path via `MtpHead::quantize_proj`.
        native_fp8: None,
        norm: dense(store, "mtp.norm.weight")?,
    })
}
