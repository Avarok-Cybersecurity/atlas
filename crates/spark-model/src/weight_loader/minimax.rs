// SPDX-License-Identifier: AGPL-3.0-only

//! MiniMax M2 weight loader.
//!
//! Architecturally MiniMax M2 (both M2.1 and M2.7 — same `minimax_m2`
//! model_type, weights differ only) is a cousin of Qwen3.5-122B-A10B:
//!   * 62 full-attention layers (no SSM/Mamba)
//!   * GQA 48 Q heads / 8 KV heads, head_dim 128
//!   * Partial RoPE (rotary_dim=64 on head_dim=128)
//!   * Full-hidden qk_norm (RMSNorm over the concatenated Q/K projections
//!     before RoPE — wired via `AttentionWeights::q_norm_full` /
//!     `k_norm_full`)
//!   * 256 experts top-8 with **sigmoid** routing + correction bias
//!   * 3 MTP draft modules (vs 1 in Qwen3.5) — M5 follow-up
//!   * Native FP8 E4M3 with `weight_block_size=[128,128]` — M4 follow-up
//!
//! Layer construction here produces a `Vec<Qwen3AttentionLayer>` with:
//!   * Attention: runtime-quantize Q/K/V/O from BF16 to NVFP4 (same
//!     code path as qwen35_dense `Standard` variant); `AttentionWeights`
//!     carries both the full-hidden qk_norm weights (MiniMax-active) and
//!     dummy per-head `q_norm/k_norm` slots (unused).
//!   * MoE: 256 experts, no shared expert, via the new
//!     `load_moe_minimax` helper. The `correction_bias` tensor is
//!     populated on `MoeWeights`; the MoE layer itself still dispatches
//!     through `moe_topk_softmax` in this commit — the sigmoid+bias
//!     dispatch lands in a follow-up when `MoeLayer::new_sigmoid` is
//!     introduced. Until that follows, runtime output is structurally
//!     correct but quantitatively wrong (softmax routing on a
//!     sigmoid-trained model) and should only be used to smoke-test
//!     the full load path on tiny-random weights.
//!
//! See `docs/MINIMAX-M2-IMPL-PLAN.md` for the full wire-up map.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::tp_shard::{
    TpShardKind, load_qk_norms_tp, load_qkvo_tp, shard_dense_1d_bf16, shard_dense_bf16,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense, dense_auto,
    detect_nvfp4_variant, load_kv_scales, load_moe_minimax, quantize_to_nvfp4,
};

pub struct MinimaxM2WeightLoader;

impl ModelWeightLoader for MinimaxM2WeightLoader {
    fn supports_tp(&self) -> bool {
        // MiniMax M2 was the reference implementation: Q/K/V col-parallel,
        // O row-parallel, q_norm/k_norm 1D-sharded, attention head counts
        // pre-divided by tp_size at construction. See `load_layers` below.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "minimax_m2: loading {} layers, variant={:?}, hidden_size={h}",
            config.num_hidden_layers,
            variant,
        );

        // Note: MoE prefill-transpose is deferred to a post-load pass in
        // `factory::build` after LM-head NVFP4 quantization frees ~22 GB of
        // BF16 headroom — doing it here at layer 0 would see only 46 GB
        // free vs a 58.9 GB transpose cost and skip it entirely. See
        // `crates/spark-model/src/factory.rs` step between LM-head quant
        // and buffer-arena allocation for the actual transpose call site.

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        // Dummy weight for the unused per-head q_norm/k_norm slots. MiniMax
        // normalizes over the full projected hidden (q_norm_full), not per
        // head — so the existing Qwen3-convention slot is intentionally
        // left NULL. The attention forward checks q_norm_full first; if
        // `Some`, it takes precedence and the NULL slot is ignored.
        let dummy_norm = DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        };

        for i in 0..config.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            tracing::debug!("minimax_m2: layer {i}");
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // ── MoE ────────────────────────────────────────────────────
            // 256 experts, no shared expert, sigmoid-routable bias loaded
            // into MoeWeights.correction_bias for M3 dispatch.
            let moe_weights = load_moe_minimax(
                store,
                &lp,
                config.num_experts,
                gpu,
                config,
                variant,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let gate_nvfp4 = quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let mut moe_layer = MoeLayer::new(
                moe_weights,
                config.num_experts,
                Some(gate_nvfp4),
                gpu,
                config,
            )?;
            // Wire up MoE prefill acceleration. `predequant_for_prefill` is
            // cheap (~50 MB total: gate only, no shared expert) and always
            // runs here. `transpose_for_prefill` is deferred to the post-load
            // pass in `factory::build` so it sees the ~65 GB free memory
            // window after LM-head NVFP4 quantization (vs the ~46 GB window
            // available at layer 0 here).
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
            let ffn = FfnComponent::Moe(moe_layer);

            // ── Attention (ungated Q, full-hidden qk_norm) ─────────────
            let p = format!("{lp}.self_attn");
            // dense_auto dequants FP8→BF16 (new GPU alloc) for each
            // projection; we runtime-quantize immediately to NVFP4 then
            // free BOTH the transient BF16 dequant buffer AND the
            // original FP8 source on GPU. Without freeing the FP8 source,
            // MiniMax M2's 230 GB checkpoint (109 GB/rank under EP=2)
            // stays resident and the runtime NVFP4 allocations push the
            // rank OOM. Attention forward uses NVFP4 via q/k/v_nvfp4 —
            // the DenseWeight slots in AttentionWeights are only
            // consulted by the BF16 fallback that MiniMax doesn't take.
            let tp_rank = config.tp_rank;
            let tp_size = config.tp_world_size;
            let load_and_quant = |name: &str,
                                  full_n: usize,
                                  full_k: usize,
                                  kind: TpShardKind|
             -> Result<(DenseWeight, QuantizedWeight)> {
                let wkey = format!("{p}.{name}.weight");
                let scale_key = format!("{p}.{name}.weight_scale_inv");
                let (src_ptr, src_is_fp8) = {
                    let t = store.get(&wkey)?;
                    (
                        t.ptr,
                        t.dtype == spark_runtime::weights::WeightDtype::FP8E4M3,
                    )
                };
                let scale_ptr = if src_is_fp8 && store.contains(&scale_key) {
                    Some(store.get(&scale_key)?.ptr)
                } else {
                    None
                };
                let dense_w = dense_auto(store, &wkey, gpu)?;
                // TP shard the BF16 weight before NVFP4 quantization. When
                // tp_size == 1 the helper returns the input pointer untouched
                // so the existing single-rank path is unchanged.
                let (sharded_ptr, local_n, local_k) =
                    shard_dense_bf16(dense_w.weight, full_n, full_k, kind, tp_rank, tp_size, gpu)?;
                let sharded = DenseWeight {
                    weight: sharded_ptr,
                };
                let q = quantize_to_nvfp4(
                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                )?;
                if sharded_ptr != dense_w.weight {
                    gpu.free(sharded_ptr)?;
                }
                if src_is_fp8 {
                    // M2 path: dense_auto allocated a fresh BF16 dequant
                    // buffer separate from the FP8 source on GPU. Free
                    // both: BF16 dequant is no longer needed once NVFP4
                    // is built, and the FP8 source can be released
                    // because the attention forward only reads the
                    // NVFP4 path. The WeightStore retains stale
                    // pointers; nothing reads them again.
                    gpu.free(dense_w.weight)?;
                    gpu.free(src_ptr)?;
                    if let Some(sp) = scale_ptr {
                        gpu.free(sp)?;
                    }
                } else {
                    // M2.7-NVFP4 path: source is BF16 and dense_auto
                    // returned the raw mmap'd store ptr (no dequant
                    // alloc). After NVFP4 quantization the BF16 source
                    // is no longer needed. Free it to recover ~22 GB
                    // per rank (4 projections × 62 layers × ~88 MB),
                    // which is the headroom needed for the MoE prefill
                    // transpose to fit on a 122 GB GB10. WeightStore
                    // retains a stale ptr — safe because nothing reads
                    // attention weights by name after load_layers
                    // returns.
                    gpu.free(src_ptr)?;
                }
                Ok((
                    DenseWeight {
                        weight: spark_runtime::gpu::DevicePtr::NULL,
                    },
                    q,
                ))
            };
            // Q/K/V column-parallel + O row-parallel via the standard TP
            // helper. `load_and_quant` is the loader-specific format closure;
            // it reads a BF16 weight from the store, TP-shards before
            // quantization, and converts to NVFP4. Helper handles the
            // dimension math for all four projections.
            let [
                (q_dense, q_nvfp4),
                (k_dense, k_nvfp4),
                (v_dense, v_nvfp4),
                (_o_dense, o_nvfp4),
            ] = load_qkvo_tp(config, |name, full_n, full_k, kind| {
                load_and_quant(name, full_n, full_k, kind)
            })?;

            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

            // MiniMax's q_norm is RMSNorm over the full projected Q:
            // weight shape `[num_heads * head_dim]`. k_norm is likewise
            // `[num_kv_heads * head_dim]`. Both applied BEFORE RoPE.
            //
            // Under TP, both vectors shard column-parallel matching the
            // local Q/K projection output. shard_dense_1d_bf16 returns the
            // input ptr untouched when tp_size == 1.
            let (q_norm_full, k_norm_full) = load_qk_norms_tp(config, |name, full_dim| {
                let src = dense(store, &format!("{p}.{name}.weight"))?;
                let (ptr, _) = shard_dense_1d_bf16(src.weight, full_dim, tp_rank, tp_size, gpu)?;
                Ok::<_, anyhow::Error>(DenseWeight { weight: ptr })
            })?;

            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                // Per-head slots intentionally NULL — the full-hidden
                // norm below takes precedence in the attention forward.
                q_norm: dummy_norm,
                k_norm: dummy_norm,
                q_norm_full: Some(q_norm_full),
                k_norm_full: Some(k_norm_full),
                k_scale,
                v_scale,
            };

            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                post_attn_norm,
                ffn,
                i, // attn_layer_idx — every layer is an attention layer for MiniMax
                Some(q_nvfp4),
                Some(k_nvfp4),
                Some(v_nvfp4),
                gpu,
                layer_kv_dtypes[i],
                config.fp8_kv_calibration_tokens,
                config,
            )?;

            // Transpose attention NVFP4 weights for prefill coalesced reads.
            // Without this, prefill.rs falls through to the tiny-tile
            // `w4a16_gemm` kernel (64×64×16, sync loads) instead of the
            // efficient `w4a16_gemm_n128_m128` path. Per-layer cost ≈
            // 24.6 MB (Q+K+V+O); 62 layers × 24.6 MB ≈ 1.5 GB per rank —
            // fits comfortably in post-MoE-transpose headroom.
            let q_proj_n = config.num_attention_heads * config.head_dim;
            let kv_proj_n = config.num_key_value_heads * config.head_dim;
            let qt = q_nvfp4.transpose_for_gemm(gpu, q_proj_n, h)?;
            let kt = k_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let vt = v_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let ot = layer.attn.o_proj.transpose_for_gemm(gpu, h, q_proj_n)?;
            layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));

            layers.push(Box::new(layer));
        }

        tracing::info!("minimax_m2: built {} layers", layers.len());
        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // MiniMax uses multi-module MTP (3 heads). The single-module
        // trait method returns None so any loader that calls the
        // old entry point gets a clean "no MTP" signal; the real
        // work happens in `load_mtp_weights_multi` below.
        Ok(None)
    }

    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        // MiniMax M2 MTP module layout in the checkpoint (verified against
        //   modeling_minimax_m2.py, MiniMaxM2ForCausalLM):
        //   model.layers.{N + i}.{...}   for i in 0..num_mtp_modules
        // where N = config.num_hidden_layers.
        //
        // Each module is a full transformer layer (attention + MoE) plus
        // the usual MTP concat block (embed-norm, hidden-norm, fc,
        // output-norm). Weight shapes match the main layers.
        //
        // The tiny-random test variant ships none of these tensors (see
        // docs/MINIMAX-M5-DESIGN.md §"Open questions"). Detect that
        // absence by probing for the first expected key — if the
        // checkpoint has no MTP modules, return an empty Vec so the
        // engine disables speculative decoding cleanly instead of
        // erroring out mid-load.
        let first_mtp_idx = config.num_hidden_layers;
        let probe = format!("model.layers.{first_mtp_idx}.input_layernorm.weight");
        if !store.contains(&probe) {
            tracing::info!(
                "minimax_m2: no MTP module weights found in checkpoint \
                 (expected starting at layer {first_mtp_idx}); MTP disabled"
            );
            return Ok(Vec::new());
        }

        // Real 229B checkpoint staging is a separate effort; once those
        // weights are present, populate one `MtpWeights` per module
        // following the same shape as `load_layers` (attention + MoE
        // sharing the existing helpers). Until then fail fast with a
        // clear message so nobody thinks MTP is silently working.
        anyhow::bail!(
            "minimax_m2: MTP module weights detected at layer {first_mtp_idx} \
             but the MiniMax loader hasn't implemented per-module extraction \
             yet. Run with --speculative 0 for non-MTP decode, or await \
             MiniMax M5 phase-3 (populate load_mtp_weights_multi with the \
             concrete Mixtral-convention weight keys)."
        )
    }
}
