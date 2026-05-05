// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 weight loader: per-layer construction (`load_layers`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::loader_b::{build_bf16_mlp, build_moe_ffn};
use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnActivation, FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use crate::weight_map::{
    AttentionWeights, QuantizeCtx, dense, dense_auto, detect_nvfp4_variant, load_kv_scales,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn load_layers_impl(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    let variant = detect_nvfp4_variant(store, config);
    tracing::info!("Gemma-4 NVFP4 variant: {:?}", variant);

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let qctx = QuantizeCtx {
        absmax_k,
        quantize_k,
        stream,
    };
    let h = config.hidden_size;

    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);

        // ── Layer norms ──
        // Gemma-4 has 4 norms: input, post_attn, pre_ffn, post_ffn.
        // The existing Qwen3AttentionLayer uses:
        //   input_norm  = pre-attention norm
        //   post_attn_norm = pre-FFN norm (applied before FFN input)
        // Map: input_layernorm → input_norm, pre_feedforward_layernorm → post_attn_norm.
        // Gemma-4's model-specific rms_norm kernel uses the absolute
        // convention `out = x * rms * weight` (see
        // `kernels/gb10/gemma-4-*/nvfp4/rms_norm.cu`), so norm weights
        // are loaded as-is — no convention shift needed.
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let pre_ffn_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?;

        // Post-sub-layer norms: applied to attn/FFN output before residual add.
        let post_attn_out_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let post_ffn_norm_w = dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?;

        // Read layer_scalar (BF16 [1] tensor).
        // Applied at the END of the layer forward pass to the ENTIRE hidden_states
        // (including residual), NOT fused into post-norm weights.
        // Reference: hidden_states = (residual + post_attn_norm(attn) + post_ffn_norm(ffn)) * layer_scalar
        let layer_scalar_key = format!("{lp}.layer_scalar");
        let layer_scalar_val = if store.contains(&layer_scalar_key) {
            let mut scalar_buf = [0u8; 2];
            let wt = store.get(&layer_scalar_key)?;
            gpu.copy_d2h(wt.ptr, &mut scalar_buf)?;
            let scalar_bf16 = u16::from_le_bytes(scalar_buf);
            let scalar_f32 = f32::from_bits((scalar_bf16 as u32) << 16);
            tracing::info!("L{i}: layer_scalar = {scalar_f32:.6}");
            Some(scalar_f32)
        } else {
            None
        };

        // ── Detect layer type from weight shapes ──
        let p = format!("{lp}.self_attn");
        let sliding_head_dim: usize = 256;
        let q_out_dim = store
            .get(&format!("{p}.q_proj.weight"))
            .map(|w| w.shape[0])
            .unwrap_or(config.num_attention_heads * sliding_head_dim);
        let kv_out_dim = store
            .get(&format!("{p}.k_proj.weight"))
            .map(|w| w.shape[0])
            .unwrap_or(config.num_key_value_heads * sliding_head_dim);
        let is_full_attn = q_out_dim != config.num_attention_heads * sliding_head_dim;

        // ── Attention weights ──
        // 31B: BF16 on disk (dense_auto). 26B MoE: NVFP4 on disk (dequant to BF16).
        let attn_is_nvfp4 = store.contains(&format!("{p}.q_proj.weight_scale"));
        let q_dense = if attn_is_nvfp4 {
            use crate::weight_map::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.q_proj"), q_out_dim, h, gpu)?
        } else {
            dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?
        };
        let k_dense = if attn_is_nvfp4 {
            use crate::weight_map::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.k_proj"), kv_out_dim, h, gpu)?
        } else {
            dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?
        };
        let v_key = format!("{p}.v_proj.weight");
        let v_dense =
            if store.contains(&v_key) || store.contains(&format!("{p}.v_proj.weight_scale")) {
                if attn_is_nvfp4 {
                    use crate::weight_map::dequant_nvfp4_to_bf16;
                    dequant_nvfp4_to_bf16(store, &format!("{p}.v_proj"), kv_out_dim, h, gpu)?
                } else {
                    dense_auto(store, &v_key, gpu)?
                }
            } else {
                k_dense // K=V: alias K as V
            };
        let o_dense = if attn_is_nvfp4 {
            use crate::weight_map::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.o_proj"), h, q_out_dim, gpu)?
        } else {
            dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?
        };
        if is_full_attn {
            tracing::info!("L{i}: full attention (Q_dim={q_out_dim}, K_dim={kv_out_dim}, K=V)");
        } else {
            tracing::debug!("L{i}: sliding attention (Q_dim={q_out_dim}, K_dim={kv_out_dim})");
        }

        // Attention quantization choice for Gemma-4. Atlas's runtime
        // BF16→NVFP4 path uses a single per-tensor absmax for scale2,
        // which loses precision in low-magnitude rows when the tensor
        // has a few outlier rows (Gemma-4-31B's calibration boost
        // channels: input_layernorm.weight max=444 at ch 3970,
        // final_norm max=510 at ch 4501). The compounding noise across
        // 60 dense layers flips a 0.125-logit-gap argmax tiebreak at
        // decode step 1 on creative prompts, after which the wrong
        // KV cache state self-reinforces a stopword loop ("Crystals a
        // a a a a..."). Bisected via ATLAS_DIAG_GEMMA4=1 logits dump.
        //
        // Default for Gemma-4 dense (31B): use BF16 attention via
        // dense_gemv fallback (qwen3_attention/decode.rs:622-699). This
        // keeps q/k/v_proj at BF16, costs ~10 GB extra GPU memory, and
        // produces coherent creative output ("Crest of the sea, / Tide
        // of the ocean's heart, / Tide of the sea." for the ocean
        // haiku). Gemma-4 MoE (26B) keeps NVFP4 — its dual-FFN per-
        // token activation is naturally lower-precision-tolerant and
        // it works correctly on creative prompts already.
        //
        // Override via ATLAS_GEMMA4_BF16_ATTN=0 to force NVFP4 for A/B
        // testing; ATLAS_GEMMA4_BF16_ATTN=1 to force BF16 even on MoE.
        // Default to BF16 attention for ALL Gemma-4 variants (dense AND MoE).
        // The 31B-dense creative-collapse fix from 2026-05-01 lands here
        // because attention NVFP4 quantization noise compounds across 60
        // layers and flips the decode-step-1 argmax. The 26B MoE was
        // initially exempted ("dual-FFN per-token activation is naturally
        // lower-precision-tolerant") but the dual-DGX sweep showed 26B
        // fails the 16K long-context test with repetition collapse —
        // same drift class, just visible after 14k decode steps instead
        // of the first one. 26B has ~28 GB total in BF16-attn mode,
        // well under the 119 GB single-GPU budget.
        let bf16_attn_default = true;
        let bf16_attn = match std::env::var("ATLAS_GEMMA4_BF16_ATTN").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => bf16_attn_default,
        };
        // BF16 MLP (gate/up/down) — DEFAULT OFF after empirical
        // verification. The 2026-05-02 evening test on Gemma-4-31B
        // confirmed BF16 MLP does NOT fix the residual drift:
        //
        //   * Greedy fib output is BIT-IDENTICAL to NVFP4 MLP
        //     (same broken `if n == 0:\n    return 0` indentation).
        //   * Creative haiku at temp=0.3 actively REGRESSED — model
        //     collapsed into emoji repetition
        //     (`Blue🌊wavescrashashore! No.` × N).
        //
        // So the drift is not in the MLP NVFP4 quantization. Combined
        // with the prior FP32 lm_head and FP32 QK^T verifications
        // (also no-ops), this exhausts the sampler-side and
        // weight-precision levers. The fib failure is intrinsic to
        // the Gemma-4-31B-NVFP4 checkpoint at greedy temp=0 — same
        // tokenizer + sampling that vLLM/HF would also hit.
        //
        // Memory cost when on: 60 layers × 3 weights × hidden(5376) ×
        // intermediate(21504) × 2 bytes = ~41 GB extra. Does fit in
        // 119 GB but increases swap-out risk. Leaving infrastructure
        // wired for future bisection (`ATLAS_GEMMA4_BF16_MLP=1`
        // re-enables; `=0` is the default).
        let bf16_mlp_default = false;
        let bf16_mlp = match std::env::var("ATLAS_GEMMA4_BF16_MLP").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => bf16_mlp_default,
        };
        // ── TP shard q/k/v/o BF16 BEFORE quantize ──
        // Per-layer dim overrides: q_out_dim and kv_out_dim are read from
        // the on-disk shape, varying per layer (sliding=256 head_dim,
        // full=different). Under TP, slice each dim by tp_size; the
        // quantize calls below pass `local_*` dims so per-rank NVFP4
        // matches its local shard.
        //
        // K=V aliasing: when v_dense aliases k_dense (sliding layers),
        // sharding k_dense in place makes the alias stale. We re-alias
        // after the slice so both still point at the shared sharded
        // weight.
        let tp_rank = config.tp_rank;
        let tp_size = config.tp_world_size.max(1);
        let v_aliases_k = v_dense.weight == k_dense.weight;
        let (mut q_dense, mut k_dense, mut v_dense, mut o_dense) =
            (q_dense, k_dense, v_dense, o_dense);
        let local_q_out = q_out_dim / tp_size;
        let local_kv_out = kv_out_dim / tp_size;
        if tp_size > 1 {
            let (qp, _, _) = shard_dense_bf16(
                q_dense.weight,
                q_out_dim,
                h,
                TpShardKind::ColumnParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if qp != q_dense.weight {
                gpu.free(q_dense.weight)?;
            }
            q_dense.weight = qp;
            let (kp, _, _) = shard_dense_bf16(
                k_dense.weight,
                kv_out_dim,
                h,
                TpShardKind::ColumnParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if kp != k_dense.weight {
                gpu.free(k_dense.weight)?;
            }
            k_dense.weight = kp;
            if v_aliases_k {
                // Re-alias V to the freshly sharded K.
                v_dense.weight = k_dense.weight;
            } else {
                let (vp, _, _) = shard_dense_bf16(
                    v_dense.weight,
                    kv_out_dim,
                    h,
                    TpShardKind::ColumnParallel,
                    tp_rank,
                    tp_size,
                    gpu,
                )?;
                if vp != v_dense.weight {
                    gpu.free(v_dense.weight)?;
                }
                v_dense.weight = vp;
            }
            let (op, _, _) = shard_dense_bf16(
                o_dense.weight,
                h,
                q_out_dim,
                TpShardKind::RowParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if op != o_dense.weight {
                gpu.free(o_dense.weight)?;
            }
            o_dense.weight = op;
        }
        let q_out_dim = local_q_out;
        let kv_out_dim = local_kv_out;
        let (q_nvfp4_opt, k_nvfp4_opt, v_nvfp4_opt) = if bf16_attn {
            tracing::info!(
                "L{i}: BF16 attention (dense_gemv path) — skip NVFP4 q/k/v quant for Gemma-4 precision"
            );
            (None, None, None)
        } else {
            let q = quantize_to_nvfp4(&q_dense, q_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let k = quantize_to_nvfp4(&k_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let v = quantize_to_nvfp4(&v_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            (Some(q), Some(k), Some(v))
        };
        // Honor Nvidia ModelOpt's official ignore list for Gemma-4:
        // ALL self_attn projections (q/k/v/o) stay BF16. Atlas
        // previously quantized o_proj unconditionally, losing ~7 bits
        // per layer to per-tensor absmax across 60 layers — a major
        // contributor to the creative-collapse drift. When bf16_attn
        // is on, we still emit a placeholder NVFP4 o_proj (the
        // AttentionWeights struct currently requires it) but the
        // layer dispatch below installs `o_dense_bf16` which the
        // decode/prefill paths prefer over the NVFP4 path.
        let o_nvfp4 = quantize_to_nvfp4(&o_dense, h, q_out_dim, gpu, absmax_k, quantize_k, stream)?;

        let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

        let attn = AttentionWeights {
            q_proj: q_dense,
            k_proj: k_dense,
            v_proj: v_dense,
            o_proj: o_nvfp4,
            q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
            k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
            q_norm_full: None,
            k_norm_full: None,
            k_scale,
            v_scale,
        };

        // ── MLP weights (NVFP4 on disk, Standard triple-scale format) ──
        // For Gemma-4 dense with bf16_mlp=true we ALSO dequantize the
        // NVFP4 weights to BF16 dense buffers and install them on the
        // FfnComponent::Dense via `set_bf16_weights`. The decode/prefill
        // dispatch in DenseFfnLayer prefers BF16 when set, falls back
        // to NVFP4 otherwise (matches the o_proj BF16 pattern). NVFP4
        // weights are kept loaded as a fallback / for non-MLP code that
        // still uses them.
        let gate_proj = quantized_any(
            store,
            &format!("{lp}.mlp.gate_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let up_proj = quantized_any(
            store,
            &format!("{lp}.mlp.up_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let down_proj = quantized_any(
            store,
            &format!("{lp}.mlp.down_proj"),
            h,
            config.intermediate_size,
            gpu,
            variant,
            qctx,
        )?;
        let ffn_weights = DenseFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
        };
        let bf16_mlp_weights = build_bf16_mlp(store, &lp, bf16_mlp, config, gpu, h)?;
        gpu.synchronize(stream)?;
        tracing::info!(
            "L{i}: FFN weights loaded (bf16_mlp={bf16_mlp}), building DenseFfnLayer (GELU)..."
        );
        let mut ffn_layer =
            DenseFfnLayer::new_with_activation(ffn_weights, FfnActivation::GeLU, gpu)?;
        if let Some((g, u, d)) = bf16_mlp_weights {
            ffn_layer.set_bf16_weights(g, u, d);
            tracing::info!("L{i}: BF16 MLP weights installed");
        }
        let ffn = FfnComponent::Dense(ffn_layer);
        gpu.synchronize(stream)?;
        tracing::info!("L{i}: DenseFfnLayer built");

        // ── MoE experts (Gemma-4 26B) — extracted to loader_b ──
        let moe_ffn = build_moe_ffn(
            store, &lp, i, config, gpu, variant, qctx, h, absmax_k, quantize_k, stream,
        )?;

        tracing::info!("L{i}: building attention layer...");

        // ── Construct attention layer (ungated Q, GQA with Q/K norms) ──
        // Full-attention layers have different head counts: 64 Q, 4 KV (vs 32/16 for sliding).
        // Create a per-layer config override so the attention kernels use correct dimensions.
        let layer_kv_dtype = layer_kv_dtypes[i];
        // Construct with global config (for buffer sizing / kernel selection).
        // Then set per-layer overrides for heterogeneous layers.
        let mut layer = Qwen3AttentionLayer::new_ungated(
            input_norm,
            attn,
            pre_ffn_norm,
            ffn,
            i,
            q_nvfp4_opt,
            k_nvfp4_opt,
            v_nvfp4_opt,
            gpu,
            layer_kv_dtype,
            config.fp8_kv_calibration_tokens,
            config,
        )?;
        // ALL layers get dimension + RoPE overrides because config uses max values
        // for buffer sizing but layers have different actual dimensions.
        //
        // Gemma-4 heterogeneous attention:
        //   Sliding: 32 Q × 256 hd, 16 KV × 256 hd  (q_out=8192, kv_out=4096)
        //   Full:    32 Q × 512 hd,  4 KV × 512 hd  (q_out=16384, kv_out=2048)
        // Derive head_dim from Q proj shape and known num_q_heads=32 (constant).
        let actual_head_dim = q_out_dim / config.num_attention_heads; // 256 or 512
        let actual_kv_heads = kv_out_dim / actual_head_dim;
        layer.set_dimension_overrides(actual_head_dim, config.num_attention_heads, actual_kv_heads);
        // Gemma-4: QK-norm handles scaling, so attention scale = 1.0
        layer.set_attn_scale_override(1.0);
        // Install BF16 dense fallback for o_proj — matches Nvidia
        // ModelOpt's official ignore list (`*.self_attn*` covers q/k/v
        // AND o; o was previously runtime-quantized to NVFP4 which
        // accumulated ~7-bit precision loss per layer over 60 layers).
        // The decode/prefill dispatch checks `o_dense_bf16` first and
        // falls through to the NVFP4 path when None.
        if bf16_attn {
            layer.set_o_dense_bf16(o_dense);
        }
        // Set post-sublayer norms (pre-scaled by layer_scalar at load time)
        layer.set_post_sublayer_norms(post_attn_out_norm, post_ffn_norm_w);
        // Layer scalar: applied to ENTIRE hidden_states at end of layer
        if let Some(scalar) = layer_scalar_val {
            layer.set_layer_scalar(scalar);
        }
        // Gemma-4 v_norm — applied at EVERY layer per HF reference
        // (modeling_gemma4.py:1170 declares
        // `Gemma4RMSNorm(head_dim, with_scale=False)` for v_norm AND
        // line 1220 applies `value_states = self.v_norm(value_states)`
        // unconditionally on every layer that owns its own KV state).
        // Atlas previously only allocated this for K=V (full-attention)
        // layers, leaving the 50/60 sliding layers without v_norm.
        // Missing v_norm leaves V un-rescaled — over 50 layers the
        // attention output drifts enough to flip greedy argmax tiebreaks
        // on creative prompts ("haiku" → "Blue a a a..." collapse).
        //
        // K=V detection (full-attention layers where v_proj weight is
        // missing) is preserved separately — it controls weight
        // ALIASING (V buffer points at K projection output), not the
        // application of v_norm. Sliding layers get v_norm without
        // K=V aliasing.
        //
        // Gemma-4's model-specific rms_norm kernel uses the absolute
        // convention `out = x * rms * weight` (see
        // `kernels/gb10/gemma-4-*/nvfp4/rms_norm.cu:100`). To get
        // pure-RMS behavior we use a ONES-filled weight buffer (NOT
        // zeros — zeros would multiply V by zero and wipe the
        // attention contribution).
        let is_k_eq_v = !store.contains(&v_key);
        let v_norm_w = super::loader_b::make_v_norm_ones_bf16(gpu, actual_head_dim)?;
        if is_k_eq_v {
            layer.set_k_eq_v(v_norm_w);
        } else {
            layer.set_v_norm(v_norm_w);
        }
        if is_full_attn {
            // Full attention: head_dim=512, theta=1M, rope_type="proportional"
            // (HF's `_compute_default_rope_parameters` for partial_rotary_factor=0.25):
            //   rope_angles = int(0.25 * head_dim / 2) = 64
            //   pairs are (i, i + head_dim/2), freq denom = head_dim
            // The proportional kernel reads `rotary_dim_override` as rope_angles.
            let rope_angles = ((actual_head_dim as f32 * 0.25) / 2.0) as u32;
            layer.set_rope_overrides(1_000_000.0, rope_angles);
            layer.set_rope_proportional(true);
            // Full layers attend to the entire KV cache — window=None → 0 in kernel.
            layer.set_sliding_window(None);
            tracing::info!(
                "L{i}: full attn: hd={actual_head_dim}, nq={}, nkv={actual_kv_heads}, rope=1M/proportional/angles={rope_angles}, K=V={is_k_eq_v}",
                config.num_attention_heads
            );
        } else {
            // Sliding attention: hd=256, theta=10K, full rotation (rotary_dim = head_dim = 256)
            layer.set_rope_overrides(10_000.0, actual_head_dim as u32);
            // Sliding layers only attend to the last `window_size` tokens.
            if config.sliding_window > 0 {
                layer.set_sliding_window(Some(config.sliding_window));
            }
            if i < 3 {
                tracing::info!(
                    "L{i}: sliding attn: hd={actual_head_dim}, nkv={actual_kv_heads}, window={:?}",
                    config.sliding_window
                );
            }
        }
        // Set dual FFN (MoE alongside dense) if present
        if let Some((moe_comp, pre_norm, post_norm, dense_norm)) = moe_ffn {
            layer.set_moe_ffn(moe_comp, pre_norm, post_norm, dense_norm);
        }

        gpu.synchronize(stream)?;
        tracing::info!("L{i}: attention layer built OK");

        // Transposed weights for prefill GEMM (column-major layout)
        // Skip prefill weight transpose for now — diagnose CUDA 700 first.
        // These are optional optimizations; decode works without them.
        // TODO: re-enable once basic inference works
        // let qt = q_nvfp4.transpose_for_gemm(gpu, num_heads * head_dim, h)?;
        // let kt = k_nvfp4.transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
        // let vt = v_nvfp4.transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
        // let ot = o_nvfp4.transpose_for_gemm(gpu, h, num_heads * head_dim)?;
        // layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
        // layer.predequant_for_prefill(gpu, config, stream)?;

        layers.push(Box::new(layer));

        if (i + 1) % 10 == 0 {
            tracing::info!("Loaded layers 0..{}", i + 1);
        }
    }

    tracing::info!(
        "Gemma-4 weight loader: {} layers (all attention, ungated, dense FFN)",
        layers.len(),
    );

    Ok(layers)
}
