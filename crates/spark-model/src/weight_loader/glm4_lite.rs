// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-4.7-Flash weight loader (DeepSeek-V3-style MLA + 64-expert MoE).
//!
//! Architecture (`config.json` → `Glm4MoeLiteForCausalLM`):
//!   * 47 hidden layers; `first_k_dense_replace = 1` → layer 0 is a dense
//!     FFN (gate/up/down, `intermediate_size = 10240`), layers 1–46 are MoE.
//!   * MLA attention: q_lora=768, kv_lora=512, qk_nope=192, qk_rope=64,
//!     v_head_dim=256, 20 Q/KV heads. Attention weights stay BF16.
//!   * MoE: 64 routed experts + 1 shared expert (`shared_experts` plural),
//!     top-k 4, `noaux_tc` sigmoid routing + `e_score_correction_bias` (F32).
//!   * MTP (Phase 4, deferred): layer 47 has `eh_proj` NVFP4 + embed_tokens +
//!     enorm/hnorm + a full MoE block.
//!   * NVFP4 layout: compressed-tensors (`weight_packed`, `weight_scale`,
//!     `weight_global_scale`, `input_global_scale`).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::moe::MoeLayer;
use crate::layers::qwen3_attention::MlaWeights;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense, detect_nvfp4_variant,
    load_dense_ffn, quantize_to_nvfp4, quantized_auto,
};
use crate::weight_map::{MoeWeights, ExpertWeight};

pub struct Glm4LiteWeightLoader;

impl ModelWeightLoader for Glm4LiteWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_embedding(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight").context("GLM embed_tokens")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight").context("GLM final norm")
    }

    fn load_lm_head(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "lm_head.weight").context("GLM lm_head")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Phase 4 deferred: GLM MTP uses NVFP4 `eh_proj` at layer 47.
        // Return None to run without speculative decoding initially.
        Ok(None)
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let n = config.num_hidden_layers;
        let h = config.hidden_size;
        let q_lora = config.q_lora_rank;
        let kv_lora = config.kv_lora_rank;
        let nope = config.qk_nope_head_dim;
        let rope = config.qk_rope_head_dim;
        let v_dim = config.v_head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let hd = config.head_dim; // v_head_dim = 256

        // first_k_dense_replace: layer 0 uses dense FFN, rest use MoE.
        // ModelConfig doesn't expose this directly; use intermediate_size > 0
        // as the proxy (layer 0 uses intermediate_size, layers 1+ use moe_intermediate_size).
        let first_k_dense = if config.intermediate_size > 0 { 1usize } else { 0 };

        let variant = detect_nvfp4_variant(store, config);
        tracing::info!("GLM-4.7-Flash: detected NVFP4 variant {variant:?}");

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        // Standard RoPE inv_freq for GLM (no YaRN; rope_theta=1e6).
        // Shape: [rope/2] F32. Shared across all layers.
        let inv_freq = compute_standard_inv_freq(config.rope_theta, rope, gpu)?;

        let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n);

        for i in 0..n {
            let lp = format!("model.layers.{i}");
            let ap = format!("{lp}.self_attn");

            let input_norm =
                dense(store, &format!("{lp}.input_layernorm.weight"))
                    .with_context(|| format!("GLM L{i} input_layernorm"))?;
            let post_norm =
                dense(store, &format!("{lp}.post_attention_layernorm.weight"))
                    .with_context(|| format!("GLM L{i} post_attention_layernorm"))?;

            // ── MLA loading (phases A–E, inlined for GLM tensor names) ──
            let mla = load_glm_mla(
                store, &ap, config, gpu, absmax_k, quantize_k, stream,
                i, h, q_lora, kv_lora, nope, rope, v_dim, n_heads, n_kv, hd, inv_freq,
            )
            .with_context(|| format!("GLM L{i} MLA"))?;

            // ── FFN ──
            let ffn = if i < first_k_dense {
                // Layer 0: dense SwiGLU FFN (intermediate_size = 10240).
                let dw = load_dense_ffn(
                    store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
                )
                .with_context(|| format!("GLM L{i} dense FFN"))?;
                let layer = DenseFfnLayer::new(dw, gpu)
                    .with_context(|| format!("GLM L{i} DenseFfnLayer::new"))?;
                FfnComponent::Dense(layer)
            } else {
                // Layers 1–46: noaux_tc MoE (64 routed + 1 shared expert).
                load_glm_moe(store, &lp, config, gpu, variant, absmax_k, quantize_k, stream, i)
                    .with_context(|| format!("GLM L{i} MoE"))?
            };

            let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Bf16);
            let attn = dummy_attn();
            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm, attn, post_norm, ffn, i, None, None, None, gpu, kv_dtype, 0, config,
            )
            .with_context(|| format!("GLM L{i} Qwen3AttentionLayer::new_ungated"))?;
            layer.set_mla_weights(mla);
            layers.push(Box::new(layer));

            if (i + 1) % 8 == 0 || i == n - 1 {
                let free = gpu.free_memory().unwrap_or(0);
                tracing::info!("GLM L{}/{n} done — {:.1} GB free", i + 1, free as f64 / 1e9);
            }
        }

        Ok(layers)
    }
}

// ── Helper: standard RoPE inv_freq (no YaRN) ────────────────────────────────

fn compute_standard_inv_freq(
    rope_theta: f64,
    rope_dim: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let n_pairs = rope_dim / 2;
    let theta = rope_theta as f32;
    let dim_f = rope_dim as f32;
    let inv_freq: Vec<f32> = (0..n_pairs)
        .map(|j| 1.0_f32 / theta.powf((2 * j) as f32 / dim_f))
        .collect();
    let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    tracing::info!(
        "GLM RoPE inv_freq: {n_pairs} pairs, theta={rope_theta:.0e}, \
         [0]={:.4e} [last]={:.4e}",
        inv_freq.first().copied().unwrap_or(0.0),
        inv_freq.last().copied().unwrap_or(0.0),
    );
    Ok(ptr)
}

// ── Helper: MLA weight loading ───────────────────────────────────────────────

/// Load all MLA weights for one GLM layer, computing the pre-absorbed matrices
/// W_UK_T, W_UV, W_QK_absorbed, and block-diagonals used by the fused MLA kernels.
#[allow(clippy::too_many_arguments)]
fn load_glm_mla(
    store: &WeightStore,
    ap: &str, // e.g. "model.layers.0.self_attn"
    _config: &ModelConfig,
    gpu: &dyn GpuBackend,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    layer_idx: usize,
    h: usize,
    q_lora: usize,
    kv_lora: usize,
    nope: usize,
    rope: usize,
    v_dim: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    inv_freq: DevicePtr,
) -> Result<MlaWeights> {
    let bf16 = 2usize;

    // ── Phase A: LoRA QKV tensors (BF16, then NVFP4) ────────────────────────
    // GLM names: q_a_proj / q_b_proj / q_a_layernorm
    //            kv_a_proj_with_mqa / kv_b_proj / kv_a_layernorm / o_proj
    let wq_a_dense = dense(store, &format!("{ap}.q_a_proj.weight"))?;
    let wq_a_nvfp4 = quantize_to_nvfp4(&wq_a_dense, q_lora, h, gpu, absmax_k, quantize_k, stream)
        .ok();

    let wq_b = dense(store, &format!("{ap}.q_b_proj.weight"))?;
    let wq_b_nvfp4 =
        quantize_to_nvfp4(&wq_b, n_heads * hd, q_lora, gpu, absmax_k, quantize_k, stream).ok();

    let q_a_norm = dense(store, &format!("{ap}.q_a_layernorm.weight"))?;

    // wkv_a: [kv_lora + rope, h] — first kv_lora rows are the latent,
    // last rope rows are K_rope used for positional encoding.
    let wkv_a_dense = dense(store, &format!("{ap}.kv_a_proj_with_mqa.weight"))?;
    let wkv_a_nvfp4 = quantize_to_nvfp4(
        &wkv_a_dense,
        kv_lora + rope,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )
    .ok();
    let wkv_a_rope_dense = DenseWeight {
        weight: wkv_a_dense.weight.offset(kv_lora * h * bf16),
    };

    let wkv_b = dense(store, &format!("{ap}.kv_b_proj.weight"))?;
    let kv_a_norm = dense(store, &format!("{ap}.kv_a_layernorm.weight"))?;

    // ── Phase E: output projection ───────────────────────────────────────────
    let o_dense_bf16 = dense(store, &format!("{ap}.o_proj.weight"))?;
    let o_nvfp4 =
        quantize_to_nvfp4(&o_dense_bf16, h, n_heads * hd, gpu, absmax_k, quantize_k, stream).ok();

    // ── Phase B: per-head transpose of W_UK, W_UV; extract wq_b_rope ────────
    let stride = nope + v_dim;
    let wkv_b_total_rows = n_kv * stride;
    let wkv_b_bytes = wkv_b_total_rows * kv_lora * bf16;
    let mut wkv_b_host = vec![0u8; wkv_b_bytes];
    gpu.copy_d2h(wkv_b.weight, &mut wkv_b_host)?;

    // Transpose K portion: [nope, kv_lora] → [kv_lora, nope] per head.
    let w_uk_per_head = kv_lora * nope * bf16;
    let mut w_uk_host = vec![0u8; n_kv * w_uk_per_head];
    for head in 0..n_kv {
        for p in 0..nope {
            for lkv in 0..kv_lora {
                let src_off = ((head * stride + p) * kv_lora + lkv) * bf16;
                let dst_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                w_uk_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&wkv_b_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uk_t_ptr = gpu.alloc(n_kv * w_uk_per_head)?;
    gpu.copy_h2d(&w_uk_host, w_uk_t_ptr)?;

    // W_UV: [n_kv, v_dim, kv_lora] — transposed-convention GEMV.
    let w_uv_ptr = gpu.alloc(n_kv * kv_lora * v_dim * bf16)?;
    for head in 0..n_kv {
        for v in 0..v_dim {
            let src_row = head * stride + nope + v;
            let src = wkv_b.weight.offset(src_row * kv_lora * bf16);
            let dst = w_uv_ptr.offset((head * v_dim * kv_lora + v * kv_lora) * bf16);
            gpu.copy_d2d(src, dst, kv_lora * bf16)?;
        }
    }

    // wq_b_rope: rope rows of wq_b per head.
    let wqbr_ptr = gpu.alloc(n_kv * rope * q_lora * bf16)?;
    {
        let mut wqb_host = vec![0u8; n_heads * hd * q_lora * bf16];
        gpu.copy_d2h(wq_b.weight, &mut wqb_host)?;
        for head in 0..n_kv {
            for r in 0..rope {
                let src_row = head * hd + nope + r;
                let src_off = src_row * q_lora * bf16;
                let dst_off = (head * rope + r) * q_lora * bf16;
                let dst = wqbr_ptr.offset(dst_off);
                let chunk = &wqb_host[src_off..src_off + q_lora * bf16];
                let mut tmp = chunk.to_vec();
                gpu.copy_h2d(&tmp, dst)?;
                tmp.clear();
            }
        }
    }

    // ── Phase C: precompute W_QK_absorbed [n_kv*kv_lora, q_lora] ────────────
    let wqk_ptr = gpu.alloc(n_kv * kv_lora * q_lora * bf16)?;
    {
        let wqb_bytes = n_heads * hd * q_lora * bf16;
        let mut wqb_buf = vec![0u8; wqb_bytes];
        gpu.copy_d2h(wq_b.weight, &mut wqb_buf)?;
        let wuk_bytes = n_kv * kv_lora * nope * bf16;
        let mut wuk_buf = vec![0u8; wuk_bytes];
        gpu.copy_d2h(w_uk_t_ptr, &mut wuk_buf)?;

        let to_f32 = |buf: &[u8], idx: usize| -> f32 {
            let bits = u16::from_le_bytes([buf[idx * 2], buf[idx * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        };
        let mut wqk_f32 = vec![0.0f32; n_kv * kv_lora * q_lora];
        for n in 0..n_kv {
            for lkv in 0..kv_lora {
                for l in 0..q_lora {
                    let mut sum = 0.0f32;
                    for p in 0..nope {
                        let wqb_val = to_f32(&wqb_buf, (n * hd + p) * q_lora + l);
                        let wuk_val = to_f32(&wuk_buf, n * kv_lora * nope + lkv * nope + p);
                        sum += wqb_val * wuk_val;
                    }
                    wqk_f32[(n * kv_lora + lkv) * q_lora + l] = sum;
                }
            }
        }
        let wqk_bf16: Vec<u8> = wqk_f32
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        gpu.copy_h2d(&wqk_bf16, wqk_ptr)?;
        if layer_idx == 0 {
            tracing::info!(
                "GLM W_QK_absorbed: [{}, {}] ({:.1} MB/layer)",
                n_kv * kv_lora,
                q_lora,
                (n_kv * kv_lora * q_lora * bf16) as f64 / 1e6,
            );
        }
    }

    // ── Phase D: block-diagonal W_UK_BD and W_UV_BD ──────────────────────────
    let bd_rows = n_kv * kv_lora;
    let bd_cols = n_kv * nope;
    let bd_size = bd_rows * bd_cols * bf16;
    let mut w_uk_bd_host = vec![0u8; bd_size];
    for head in 0..n_kv {
        for lkv in 0..kv_lora {
            for p in 0..nope {
                let src_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                let dst_row = head * kv_lora + lkv;
                let dst_col = head * nope + p;
                let dst_off = (dst_row * bd_cols + dst_col) * bf16;
                w_uk_bd_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&w_uk_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uk_bd_ptr = gpu.alloc(bd_size)?;
    gpu.copy_h2d(&w_uk_bd_host, w_uk_bd_ptr)?;

    let uv_bd_rows = n_kv * v_dim;
    let uv_bd_cols = n_kv * kv_lora;
    let uv_bd_size = uv_bd_rows * uv_bd_cols * bf16;
    let mut w_uv_host = vec![0u8; n_kv * v_dim * kv_lora * bf16];
    gpu.copy_d2h(w_uv_ptr, &mut w_uv_host)?;
    let mut w_uv_bd_host = vec![0u8; uv_bd_size];
    for head in 0..n_kv {
        for v in 0..v_dim {
            for l in 0..kv_lora {
                let src_off = (head * v_dim * kv_lora + v * kv_lora + l) * bf16;
                let dst_row = head * v_dim + v;
                let dst_col = head * kv_lora + l;
                let dst_off = (dst_row * uv_bd_cols + dst_col) * bf16;
                w_uv_bd_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&w_uv_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uv_bd_ptr = gpu.alloc(uv_bd_size)?;
    gpu.copy_h2d(&w_uv_bd_host, w_uv_bd_ptr)?;

    if layer_idx == 0 {
        tracing::info!(
            "GLM MLA block-diag: W_UK [{bd_rows},{bd_cols}] ({:.1}MB), \
             W_UV [{uv_bd_rows},{uv_bd_cols}] ({:.1}MB)",
            bd_size as f64 / 1e6,
            uv_bd_size as f64 / 1e6,
        );
    }

    Ok(MlaWeights {
        wq_a: wq_a_dense,
        wq_a_nvfp4,
        wq_b,
        wq_b_nvfp4,
        q_a_norm,
        wkv_a: wkv_a_dense,
        wkv_a_nvfp4,
        wkv_b,
        kv_a_norm,
        wkv_a_rope: wkv_a_rope_dense,
        wkv_a_merged: DenseWeight { weight: wkv_a_dense.weight },
        wo: o_dense_bf16,
        wo_nvfp4: o_nvfp4,
        wq_b_rope: DenseWeight { weight: wqbr_ptr },
        w_uk_t: DenseWeight { weight: w_uk_t_ptr },
        w_uv: DenseWeight { weight: w_uv_ptr },
        w_qk_absorbed: DenseWeight { weight: wqk_ptr },
        w_uk_block_diag: DenseWeight { weight: w_uk_bd_ptr },
        w_uv_block_diag: DenseWeight { weight: w_uv_bd_ptr },
        yarn_inv_freq: inv_freq,
        q_lora_rank: q_lora,
        kv_lora_rank: kv_lora,
        nope,
        rope,
        v_dim,
    })
}

// ── Helper: GLM MoE loading (noaux_tc: sigmoid + correction bias) ────────────

#[allow(clippy::too_many_arguments)]
fn load_glm_moe(
    store: &WeightStore,
    lp: &str, // e.g. "model.layers.1"
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn GpuBackend,
    variant: crate::weight_map::Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    _layer_idx: usize,
) -> Result<FfnComponent> {
    let mlp = format!("{lp}.mlp");
    let n_experts = config.num_experts;
    let _inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    // Routing gate: BF16 [n_experts, h].
    let gate = dense(store, &format!("{mlp}.gate.weight"))?;

    // Correction bias: F32 [n_experts] — GLM ships it as F32 already.
    let correction_bias =
        dense(store, &format!("{mlp}.gate.e_score_correction_bias")).ok();

    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        Ok(ExpertWeight {
            gate_proj: quantized_auto(store, &format!("{prefix}.gate_proj"), gpu, variant)?,
            up_proj: quantized_auto(store, &format!("{prefix}.up_proj"), gpu, variant)?,
            down_proj: quantized_auto(store, &format!("{prefix}.down_proj"), gpu, variant)?,
        })
    };

    // 64 routed experts.
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let expert =
            load_expert(&format!("{mlp}.experts.{e}")).unwrap_or_else(|err| {
                tracing::warn!("GLM expert {e} load failed: {err:#}; using null");
                ExpertWeight::null()
            });
        experts.push(expert);
    }

    // 1 shared expert — GLM uses `shared_experts` (plural).
    // No shared_expert_gate; the shared expert is always applied unconditionally.
    // A null gate DenseWeight signals to the MoE kernel to skip sigmoid-gating.
    let shared_expert = load_expert(&format!("{mlp}.shared_experts"))
        .unwrap_or_else(|err| {
            tracing::warn!("GLM shared_experts load failed: {err:#}; using null");
            ExpertWeight::null()
        });
    let shared_expert_gate = DenseWeight {
        weight: DevicePtr::NULL,
    };

    // Quantize gate weight to NVFP4 for the fused gate GEMM path.
    let gate_nvfp4 =
        quantize_to_nvfp4(&gate, n_experts, h, gpu, absmax_k, quantize_k, stream).ok();

    let moe_weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias,
    };

    let mut moe = MoeLayer::new(moe_weights, n_experts, gate_nvfp4, gpu, config)?;
    // Skip MoE transpose for single-GPU to save ~33 GB peak.
    // Prefill falls back to the untransposed path (slightly slower).
    if config.ep_world_size > 1 {
        if let Err(e) = moe.transpose_for_prefill(gpu, config) {
            tracing::warn!("GLM MoE transpose failed: {e:#}; using untransposed");
        }
    }
    moe.predequant_for_prefill(gpu, config, stream)?;

    Ok(FfnComponent::Moe(moe))
}

// ── Helper: dummy AttentionWeights stub (MLA path never reads these) ─────────

fn dummy_attn() -> AttentionWeights {
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let null_qw = QuantizedWeight {
        weight: DevicePtr::NULL,
        weight_scale: DevicePtr::NULL,
        weight_scale_2: 0.0,
        input_scale: DevicePtr::NULL,
    };
    AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: null_qw,
        q_norm: null,
        k_norm: null,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    }
}
