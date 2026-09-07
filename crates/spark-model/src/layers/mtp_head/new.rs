// SPDX-License-Identifier: AGPL-3.0-only

//! MTP head constructor.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::{MtpHead, MtpQuantization, ProjectionWeight};
use crate::layers::{MoeLayer, MoeSite};
use crate::weight_map::{DenseWeight, MoeWeights, MtpWeights, QuantizedWeight, quantize_to_nvfp4};

impl MtpHead {
    pub fn new(
        weights: MtpWeights,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        // Padded transposed twin of the SHARED main head (see the field docs):
        // `Some` only when the drafter head IS the main head, so routing the
        // batched-propose lm_head through it is the exact weight at tile-GEMM
        // bandwidth. Caller passes `None` for dedicated draft heads.
        lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        quant: MtpQuantization,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        main_kv_blocks: usize,
        // This model's levers — the drafter-prefill policy decides whether the
        // dedicated prefill scratch is allocated at all.
        levers: &crate::layers::ops::ModelLevers,
    ) -> Result<Self> {
        let stream = gpu.default_stream();
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let nvfp4_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;

        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        // Dense MTP heads use the dense `intermediate_size`; MoE MTP heads
        // use `moe_intermediate_size`. The bundled `mtp.safetensors` always
        // matches the main model's FFN width (Qwen3.6-27B FP8: 17408 dense;
        // Qwen3.6-A3B-NVFP4: 1024 per expert).
        let inter = if config.moe_intermediate_size > 0 {
            config.moe_intermediate_size
        } else {
            config.intermediate_size
        };

        let q = |bf16: &DenseWeight, n: usize, k: usize| -> Result<ProjectionWeight> {
            Self::quantize_proj(bf16, n, k, quant, gpu, absmax_k, nvfp4_k, fp8_k, stream)
        };

        // Quantize projections
        let fc = q(&weights.fc, h, h * 2)?;
        let q_proj = q(&weights.q_proj, nq * hd * 2, h)?;
        let k_proj = q(&weights.k_proj, nkv * hd, h)?;
        let v_proj = q(&weights.v_proj, nkv * hd, h)?;
        let o_proj = q(&weights.o_proj, h, nq * hd)?;

        // Dense FFN MTP heads (Qwen3.6-27B-FP8) bypass the MoE setup entirely.
        // We quantize the dense gate/up/down triple and stash it; the MoE
        // fields stay None and the forward path takes the dense shortcut.
        let dense_ffn_generic = if let Some(dense_ffn) = weights.dense_ffn.as_ref() {
            if matches!(quant, MtpQuantization::Nvfp4) {
                anyhow::bail!(
                    "MTP NVFP4 mode is not supported for dense FFN MTP heads yet \
                     (Qwen3.6-27B-FP8 ships an FP8 MTP head — use \
                     `--mtp-quantization fp8` or `bf16`)"
                );
            }
            Some((
                q(&dense_ffn.gate_proj, inter, h)?,
                q(&dense_ffn.up_proj, inter, h)?,
                q(&dense_ffn.down_proj, h, inter)?,
            ))
        } else {
            None
        };

        // MoE: NVFP4 uses fused MoeLayer; FP8/BF16 stores per-expert weights
        let (moe_nvfp4, moe_experts_generic, moe_shared_generic) = if dense_ffn_generic.is_some() {
            (None, None, None)
        } else {
            match quant {
                MtpQuantization::Nvfp4 => {
                    let gate_nvfp4 = quantize_to_nvfp4(
                        &weights.moe_gate,
                        config.num_experts,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let mut experts = Vec::with_capacity(weights.experts.len());
                    for (i, de) in weights.experts.iter().enumerate() {
                        let gate_proj = quantize_to_nvfp4(
                            &de.gate_proj,
                            inter,
                            h,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        let up_proj = quantize_to_nvfp4(
                            &de.up_proj,
                            inter,
                            h,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        let down_proj = quantize_to_nvfp4(
                            &de.down_proj,
                            h,
                            inter,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        experts.push(crate::weight_map::ExpertWeight {
                            gate_proj,
                            up_proj,
                            down_proj,
                        });
                        if (i + 1) % 128 == 0 {
                            tracing::info!(
                                "  MTP experts quantized: {}/{}",
                                i + 1,
                                weights.experts.len()
                            );
                        }
                    }
                    let shared_gate = quantize_to_nvfp4(
                        &weights.shared_expert.gate_proj,
                        inter,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let shared_up = quantize_to_nvfp4(
                        &weights.shared_expert.up_proj,
                        inter,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let shared_down = quantize_to_nvfp4(
                        &weights.shared_expert.down_proj,
                        h,
                        inter,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let moe_weights = MoeWeights {
                        gate: weights.moe_gate,
                        shared_expert: crate::weight_map::ExpertWeight {
                            gate_proj: shared_gate,
                            up_proj: shared_up,
                            down_proj: shared_down,
                        },
                        shared_expert_gate: weights.shared_expert_gate,
                        experts,
                        router_pre_norm: None,
                        correction_bias: None,
                    };
                    let moe = MoeLayer::new(
                        MoeSite::MtpHead,
                        moe_weights,
                        config.num_experts,
                        Some(gate_nvfp4),
                        gpu,
                        config,
                    )?;
                    (Some(moe), None, None)
                }
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    let mut experts_g = Vec::with_capacity(weights.experts.len());
                    for (i, de) in weights.experts.iter().enumerate() {
                        let gate_proj = q(&de.gate_proj, inter, h)?;
                        let up_proj = q(&de.up_proj, inter, h)?;
                        let down_proj = q(&de.down_proj, h, inter)?;
                        experts_g.push((gate_proj, up_proj, down_proj));
                        if (i + 1) % 128 == 0 {
                            tracing::info!(
                                "  MTP experts quantized: {}/{}",
                                i + 1,
                                weights.experts.len()
                            );
                        }
                    }
                    let shared = (
                        q(&weights.shared_expert.gate_proj, inter, h)?,
                        q(&weights.shared_expert.up_proj, inter, h)?,
                        q(&weights.shared_expert.down_proj, h, inter)?,
                    );
                    (None, Some(experts_g), Some(shared))
                }
            }
        };

        // MTP KV cache: 1 attention layer. The FP8 KV path hard-codes
        // k_scale=v_scale=1.0, which on Qwen3.6-A3B (large deep-layer K/V
        // magnitudes) collapsed the single MTP attention layer's output to a
        // constant → constant draft token 0 → ~0% acceptance, making
        // --mtp-quantization fp8 a net slowdown. Use BF16 KV for both bf16
        // and fp8 MTP heads — the MTP KV is one tiny layer, so BF16 cost is
        // negligible. NVFP4 MTP keeps FP8 KV (measured-good acceptance;
        // FP8-path changes must stay additive for NVFP4).
        let kv_bf16 = matches!(quant, MtpQuantization::Bf16 | MtpQuantization::Fp8);
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: nkv,
            head_dim: hd,
            num_layers: 1,
            dtype: if kv_bf16 {
                KvCacheDtype::Bf16
            } else {
                KvCacheDtype::Fp8
            },
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        // The drafter's KV pool must admit EVERY concurrently-drafting
        // sequence, not one: `max_seq_len/bs + 1` was sized before MTP propose
        // went batched, and at C=16 x ~2K-token contexts it is ~2x short — the
        // "KV cache exhausted" ERROR spam from run_mtp_propose_batched is this
        // pool (the 15K-block MAIN pool never fills on that workload), and each
        // hit degrades the batched propose to the per-step fallback. Scale by
        // the MTP concurrency cap, bounded by the main pool's block count: the
        // drafter cannot need more live tokens than the main KV can hold, so
        // the cap keeps a 128K `--max-seq-len` config from blindly allocating
        // seqs x 8K blocks. Cost at the 16K/16-seq bench config: 15,203 blocks
        // x 64 KB = ~0.97 GB — pre-charged against the KV budget by
        // `factory::build`'s MTP propose-pool reserve, which mirrors THIS
        // arithmetic; change one and change both.
        let per_seq_blocks = max_seq_len / kv_config.block_size + 1;
        let mtp_num_blocks = per_seq_blocks
            .saturating_mul(crate::speculative::mtp_max_seqs())
            .min(main_kv_blocks.max(per_seq_blocks));
        // Per-sequence `propose_meta` stride, from the SAME block size the
        // drafter pool uses (so the two cannot drift). The old fixed 2048
        // capped the block table at 448 entries = 7,168 tokens — sized in
        // the 4K era; agentic contexts of 10-20K tripped the stride ensure!
        // every step and made the batched propose permanently fall back to
        // per-sequence mode (PROGRESS_LOG 5.2/6.17). Floor 2048; override
        // ATLAS_PROPOSE_META_STRIDE=<bytes>.
        let propose_meta_stride =
            super::batch_caps::propose_meta_stride_env(max_seq_len, kv_config.block_size);
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        // Extra kernel handles for BF16/FP8 paths
        let (
            dense_gemv_k,
            dense_gemv_fp8w_k,
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
        ) = match quant {
            MtpQuantization::Nvfp4 => (None, None, None, None, None, None),
            MtpQuantization::Fp8 => (
                // BF16 GEMV needed for gate (always BF16) + generic MoE dispatch
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                Some(gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?),
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
            MtpQuantization::Bf16 => (
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                None,
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
        };

        let effective_vocab = if mtp_vocab_size > 0 {
            (mtp_vocab_size as usize).min(config.vocab_size)
        } else {
            config.vocab_size
        };
        let ffn_kind: &str = if dense_ffn_generic.is_some() {
            "dense FFN"
        } else if moe_nvfp4.is_some() {
            "MoE (NVFP4 fused)"
        } else {
            "MoE (per-expert)"
        };
        tracing::info!(
            "MTP head: quant={:?}, fc=[{h},{h2}], attn Q=[{qd},{h}], ffn={ffn}, \
             {ne} experts, vocab={ev}/{fv} (LM head {lm:.1} MB)",
            quant,
            h2 = h * 2,
            qd = nq * hd * 2,
            ffn = ffn_kind,
            ne = if dense_ffn_generic.is_some() {
                0
            } else {
                config.num_experts
            },
            ev = effective_vocab,
            fv = config.vocab_size,
            lm = (effective_vocab * h / 2) as f64 / (1024.0 * 1024.0),
        );

        // Dedicated batched-prefill scratch (~50 MB at h=5120/nq=32/hd=256,
        // PREFILL_CHUNK=512 rows). Dedicated rather than aliased onto the
        // shared arena so the pass has zero aliasing hazards; allocated only
        // when a consumer exists.
        // The catch-up feed (ATLAS_MTP_CATCHUP) runs through the same batched
        // row writer as the drafter prefill and needs the same scratch.
        let prefill_scratch = if super::mtp_drafter_prefill_enabled(levers)
            || crate::speculative::mtp_catchup_enabled()
        {
            let c = super::prefill::PREFILL_CHUNK;
            let bf16 = 2usize;
            Some(super::MtpPrefillScratch {
                embed: gpu.alloc(c * h * bf16)?,
                normed_embed: gpu.alloc(c * h * bf16)?,
                normed_hidden: gpu.alloc(c * h * bf16)?,
                concat: gpu.alloc(c * 2 * h * bf16)?,
                fc_out: gpu.alloc(c * h * bf16)?,
                normed2: gpu.alloc(c * h * bf16)?,
                k_out: gpu.alloc(c * nkv * hd * bf16)?,
                v_out: gpu.alloc(c * nkv * hd * bf16)?,
                q_scratch: gpu.alloc(c * nq * hd * bf16)?,
                pos_dev: gpu.alloc(c * 4)?,
                slot_dev: gpu.alloc(c * 8)?,
            })
        } else {
            None
        };

        Ok(Self {
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding,
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden,
            input_layernorm: weights.input_layernorm,
            post_attn_layernorm: weights.post_attn_layernorm,
            norm: weights.norm,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            moe_nvfp4,
            moe_experts_generic,
            moe_shared_generic,
            moe_gate: weights.moe_gate,
            shared_expert_gate: weights.shared_expert_gate,
            dense_ffn_generic,
            quant,
            mtp_vocab_size,
            embed_tokens,
            lm_head_nvfp4,
            kv_cache: Mutex::new(kv_cache),
            attn_layer_idx: 0,
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            gemv_sw: crate::layers::ops::gemv_sw_from(
                std::env::var("ATLAS_NO_GEMV_SW").ok().as_deref(),
            ),
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            reshape_cache_k: if kv_bf16 {
                gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?
            } else {
                gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?
            },
            paged_decode_k: if kv_bf16 {
                gpu.kernel("paged_decode", "paged_decode_attn")?
            } else {
                gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")?
            },
            kv_bf16,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm")?,
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            embed_from_argmax_k: gpu.kernel("embed_from_argmax", "embed_from_argmax")?,
            draft_token_id_dev: gpu.alloc(4)?,
            last_conf_bits: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
            dense_gemv_k,
            dense_gemv_fp8w_k,
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv").ok(),
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
            // Batched BF16 GEMM for drafter prefill; 0-handle when the
            // target's kernel set lacks it (prefill then no-ops).
            dense_gemm_k: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16"),
            dense_gemm_pipelined_k: crate::layers::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            // Batched BF16 GEMV for the M=2..8 propose widths; 0-handle on
            // targets whose kernel set predates it (dispatch falls back to
            // the pipelined GEMM — see `row_dispatch`).
            dense_gemv_batchm_k: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemv_batch16_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch16",
            ),
            w4a16_gemv_batch32_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch32",
            ),
            // Propose-side tile-twin routing decided ONCE at construction
            // (process-static: handle + env + weight presence), so per-n CUDA
            // graph captures of the batched propose can never see the
            // selection flip. Kill switch is PRESENCE-style per the house
            // convention (`ATLAS_NO_MTP_LMHEAD_TGEMM=0` is NOT off).
            lm_head_nvfp4_t: lm_head_nvfp4_t
                .filter(|_| std::env::var_os("ATLAS_NO_MTP_LMHEAD_TGEMM").is_none()),
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            argmax_batch_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch"),
            argmax_batch_lp_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch_lp"),
            // Drafter attention metadata for the batched propose — a
            // dedicated allocation (32 x stride; 64 KB at the 2048 floor),
            // never an offset into the shared scratch arena (see the
            // `propose_meta` field docs). Sized from the dynamic stride so
            // long-context configs keep the batched propose.
            propose_meta: gpu.alloc(super::batch_caps::PROPOSE_META_SEQS * propose_meta_stride)?,
            propose_meta_stride,
            prefill_scratch,
        })
    }
}
