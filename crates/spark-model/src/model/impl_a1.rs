// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        // Runtime FP8 LM head (`--lm-head-dtype fp8`); `None` on NVFP4/BF16 paths.
        lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
        // Draft-only NVFP4 head for MTP when the main head is BF16; `None`
        // otherwise (the proposer then falls back to `lm_head_nvfp4`). Drafts
        // are verified by the main head, so it never affects an accepted token.
        mtp_lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        gpu: Box<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: crate::layers::MtpQuantization,
        use_speculative: bool,
        prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<crate::layers::VisionEncoder>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        let fp32_residual = config.use_fp32_residual();
        let rms_norm_kernel = if fp32_residual {
            gpu.kernel("norm", "rms_norm_f32")
                .or_else(|_| gpu.kernel("norm", "rms_norm"))?
        } else {
            gpu.kernel("norm", "rms_norm")?
        };
        let bf16_to_f32_kernel = if fp32_residual {
            gpu.kernel("residual_add", "bf16_to_f32")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0) // BF16 models don't need conversion
        };
        let dense_gemv_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
        // FP32-output dense GEMV — only loaded when LM head needs FP32 logits.
        // For models that don't use FP32 residual, this stays KernelHandle(0)
        // and the BF16 path is taken. The kernel lives in the same `gemv`
        // module as `dense_gemv_bf16` so this lookup is cheap.
        let dense_gemv_fp32out_kernel = if fp32_residual {
            gpu.kernel("gemv", "dense_gemv_bf16_fp32out")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0)
        };
        let w4a16_gemv_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
        let w4a16_gemv_logits_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_logits")?;
        let w4a16_gemm_kernel = gpu.kernel("w4a16", "w4a16_gemm")?;
        let w4a16_gemv_batch2_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?;
        // FP8 E4M3 LUT GEMV for the `--lm-head-dtype fp8` head (cheap handle; only used when set).
        let dense_gemv_fp8w_kernel = gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
        let dense_gemm_kernel = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let argmax_kernel = gpu.kernel("argmax", "argmax_bf16")?;
        let argmax_logits_kernel = gpu.kernel("argmax", "argmax_fp32")?;
        let batched_embed_kernel = if fp32_residual {
            gpu.kernel("embed_from_argmax", "batched_embed_f32")
                .or_else(|_| gpu.kernel("embed_from_argmax", "batched_embed"))?
        } else {
            gpu.kernel("embed_from_argmax", "batched_embed")?
        };
        let fill_slots_kernel = gpu.kernel("metadata_fill", "fill_slots_from_block_table")?;
        let profile = std::env::var("ATLAS_PROFILE").is_ok();
        let profile_first = std::env::var("ATLAS_PROFILE_FIRST").is_ok();

        tracing::info!(
            "TransformerModel: {} layers, vocab={}, hidden={}{}{}",
            layers.len(),
            config.vocab_size,
            config.hidden_size,
            if profile { " [PROFILE MODE]" } else { "" },
            if profile_first {
                " [PROFILE_FIRST]"
            } else {
                ""
            },
        );

        // Build SSM state pool (with MTP intermediate/checkpoint pools only if speculative decoding enabled)
        // num_intermediates = K (per-token SSM h/conv state snapshots).
        // For MTP K=2/3/4 verify: K = num_drafts + 1.
        // For DFlash K=γ verify: K = γ + 1 (drafter's γ drafts + 1 verified bonus slot).
        // Pool size = max of both so DFlash and MTP can coexist on the same model.
        let dflash_kgamma = if !config.dflash_capture_layers.is_empty() {
            // Drafter's γ is fixed in dflash config; use the largest known γ
            // (16 for `Qwen3.6-DFlash`). The +1 is the prefix bonus position
            // in the verify input `[last_token, draft_0, ..., draft_{γ-1}]`.
            17
        } else {
            0
        };
        // DFlash needs the SSM verify pools regardless of MTP weight presence
        // or lm_head quantization — its K=γ verify path checkpoints SSM state
        // for partial-accept rollback. Force `has_mtp` on whenever DFlash is
        // active so the checkpoint pools exist.
        // The MTP proposer needs an NVFP4 vocab head for drafting: either the
        // main head (NVFP4 default) or the draft-only head built when the main
        // head is BF16. `draft_lm_head_nvfp4` resolves to whichever is present.
        let draft_lm_head_nvfp4 = mtp_lm_head_nvfp4.or(lm_head_nvfp4);
        let has_mtp = self_speculative
            || (use_speculative && !mtp_weights.is_empty() && draft_lm_head_nvfp4.is_some())
            || dflash_kgamma > 0;
        let num_intermediates = if has_mtp {
            (num_drafts + 1).max(dflash_kgamma)
        } else {
            0
        };
        let ssm_pool = SsmStatePool::new(
            &config,
            max_batch_size,
            has_mtp,
            num_intermediates,
            gpu.as_ref(),
        )?;

        // SSM snapshot pool: Marconi prefix-cache slots + Phase-C
        // decode-rollback ring. The decode-rollback region is only sized
        // for SSM models — `num_ssm_layers == 0` makes both regions
        // collapse to empty. Per sequence the watchdog needs
        // `ROLLBACK_RESTEER_CAP + 1` snapshots (one per allowed rollback,
        // plus the current boundary); the region is sized for every
        // active-sequence pool slot (`max_batch_size`).
        let decode_ring_slots = if ssm_pool.num_ssm_layers > 0 {
            (atlas_kernels::ROLLBACK_RESTEER_CAP as usize) + 1
        } else {
            0
        };
        let ssm_snapshots = SsmSnapshotPool::new(
            ssm_cache_slots,
            ssm_pool.h_bytes,
            ssm_pool.conv_bytes,
            ssm_pool.num_ssm_layers,
            decode_ring_slots,
            max_batch_size,
            gpu.as_ref(),
        )?;
        if ssm_checkpoint_interval > 0 && ssm_cache_slots > 0 {
            tracing::info!(
                "Marconi intermediate checkpoints: every {} blocks ({} tokens at block_size={})",
                ssm_checkpoint_interval,
                ssm_checkpoint_interval * kv_cache.block_size(),
                kv_cache.block_size(),
            );
        }

        // Fixed metadata stride for CUDA graph compatibility
        let max_blocks_per_seq = (max_seq_len / kv_cache.block_size() + 1) as u32;

        // Permanent dummy KV block for padding sequences. Must be explicitly
        // zeroed: `gpu.alloc()` returns uninitialized memory, and any kernel
        // OOB-read (now routed here via the sentinel block_table_flat default
        // fill in upload_batch_metadata_*) would otherwise dequant random
        // bytes and inject garbage into attention scores.
        let mut kv_cache = kv_cache;
        let dummy_kv_block = kv_cache.alloc_block()?;
        kv_cache.zero_block(dummy_kv_block, gpu.as_ref(), gpu.default_stream())?;
        gpu.synchronize(gpu.default_stream())?;

        // Build MTP proposer (extracted to keep `new` under the file cap).
        let proposer: Option<Arc<dyn DraftProposer>> = super::impl_a1_init::build_mtp_proposer(
            use_speculative,
            mtp_weights,
            embed_tokens,
            draft_lm_head_nvfp4,
            &config,
            gpu.as_ref(),
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
        );

        if self_speculative {
            let num_ssm = config.num_ssm_layers();
            let num_attn = config.num_attention_layers();
            tracing::info!(
                "Self-speculative decoding: ENABLED (skipping {} SSM layers, keeping {} attention layers)",
                num_ssm,
                num_attn,
            );
        }

        // MTP hidden state save buffer (1 × hidden_size FP32)
        let mtp_hidden_save = gpu.alloc(config.hidden_size * 4)?;

        // DFlash 5-layer hidden-state stack. Allocated only when a
        // BlockDiffusionDraftHead is the active proposer (`config.dflash_capture_layers`
        // populated by the loader from the drafter's `dflash_config.target_layer_ids`).
        // Size: N_capture × hidden_size × bf16 (typically 5 × 2048 × 2 = 20 KB).
        let dflash_capture_layers: Vec<usize> = config.dflash_capture_layers.clone();
        let dflash_hidden_save = if dflash_capture_layers.is_empty() {
            None
        } else {
            let n = dflash_capture_layers.len();
            Some(gpu.alloc(n * config.hidden_size * 2)?)
        };

        // EP command buffer for token broadcast (4 bytes, u32)
        let ep_cmd_buf = gpu.alloc(4)?;

        // Secondary stream + event for pipelining checkpoint D2D with MTP propose.
        let secondary_stream = gpu.create_stream()?;
        let secondary_event = gpu.create_event()?;

        // EP: register moe_output buffer with NCCL and provide bf16_add kernel.
        if let Some(ref comm) = comm
            && comm.world_size() == 2
        {
            let moe_ptr = buffers.moe_output().0;
            let moe_bytes = buffers.sizes().moe_output;
            match comm.register_buffer(moe_ptr, moe_bytes) {
                Ok(_) => tracing::info!("Registered moe_output ({moe_bytes} B) with NCCL"),
                Err(e) => tracing::warn!("ncclCommRegister moe_output failed (non-fatal): {e}"),
            }
            match gpu.kernel("bf16_add", "bf16_add_inplace") {
                Ok(k) => comm.set_add_kernel(k.0),
                Err(e) => {
                    tracing::warn!("bf16_add_inplace kernel not found (send/recv disabled): {e}")
                }
            }
        }

        // Allocate pinned host staging buffer for batched metadata H2D.
        let pinned_bytes = buffers.sizes().scratch.max(64 * 1024);
        let pinned_ptr = gpu.alloc_host_pinned(pinned_bytes)?;
        tracing::info!("Pinned metadata staging: {} KB", pinned_bytes / 1024);
        let max_batch_tokens = buffers.max_batch_tokens();
        let pinned_staging = std::cell::UnsafeCell::new(PinnedMetaStaging {
            ptr: pinned_ptr,
            bytes: pinned_bytes,
            positions: Vec::with_capacity(max_batch_tokens),
            positions_h: Vec::with_capacity(max_batch_tokens),
            positions_w: Vec::with_capacity(max_batch_tokens),
            slots: Vec::with_capacity(max_batch_tokens),
        });

        // SSM state normalization kernel + pointer buffer (for chunked prefill).
        let ssm_norm_k = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
            .unwrap_or(KernelHandle(0));

        // Logit softcapping (Gemma-4: cap=30.0). Only load if model uses it.
        let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
            gpu.kernel("logit_softcap", "logit_softcap_bf16")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 softcap variant — only loaded when both softcap and FP32
        // residual are active (i.e. Gemma-4 dense). Other models keep the
        // BF16 softcap (or no softcap at all).
        let logit_softcap_fp32_kernel = if config.final_logit_softcapping > 0.0 && fp32_residual {
            gpu.kernel("logit_softcap", "logit_softcap_fp32")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap_fp32 kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 logits gate. The LM head produces FP32 (rather than BF16)
        // logits when the residual stream is FP32 AND the LM head is a
        // dense BF16 weight (no NVFP4 quant). NVFP4 LM heads keep their
        // existing path because that quantization is a much larger
        // precision floor than the BF16 store; FP32 wouldn't help there.
        // Today this only affects Gemma-4 dense (model_type=="gemma4",
        // num_experts==0, tied BF16 embed→lm_head).
        // Gemma-4-31B FP32 lm_head experiment. Disabled by default —
        // session 2026-05-01 verified the BF16 lm_head store is NOT the
        // source of Gemma-4's haiku argmax flip: FP32 view of step-1
        // logits keeps top1=` a` (21.85), top2=` waves` (21.706) — same
        // 0.14-margin tiebreak as BF16. The drift is upstream in attention
        // or MLP, not in the lm_head precision boundary. Code paths kept
        // wired so a future bisection (Phase 2 of the plan) can re-enable
        // via `ATLAS_GEMMA4_FP32_LMHEAD=1`. Keep `use_fp32_logits=false`
        // by default so the rest of the model behaves identically to the
        // pre-fix BF16 path on every model family.
        // FP32 lm_head + softcap. Default OFF — empirically the gain on
        // Gemma-4-31B is marginal (Creative occasionally cleaner; fib still
        // fails the same broken-indentation pattern) but the cost is huge:
        // FP32 forces host-side sampling (vocab=262144 × 4 bytes per
        // decode step → ~1 MB D2H per token) which crushes decode TPS
        // from ~35 tok/s to ~6 tok/s on Gemma-4-31B. Not worth it without
        // a GPU-side FP32 argmax kernel. `ATLAS_GEMMA4_FP32_LMHEAD=1`
        // re-enables for bisection / future work.
        //
        // The earlier "FP32 doesn't fix haiku" comment in this file was
        // arrived at via incomplete bisection (the scheduler readback
        // always assumed BF16 — see commit 16b2f3a's commit body). The
        // 2026-05-01 evening run with the dispatch wired confirmed the
        // bisection's *qualitative* conclusion: FP32 lm_head + softcap
        // doesn't materially fix Gemma-4's structural NVFP4 attention
        // drift on greedy code generation. Fix is upstream of lm_head.
        let env_override = std::env::var("ATLAS_GEMMA4_FP32_LMHEAD").ok();
        let fp32_requested = matches!(env_override.as_deref(), Some("1") | Some("true"));
        let use_fp32_logits = fp32_requested
            && fp32_residual
            && ((lm_head_nvfp4.is_none() && dense_gemv_fp32out_kernel.0 != 0)
                || (lm_head_nvfp4.is_some() && w4a16_gemv_logits_kernel.0 != 0));
        // Dedicated FP32 logits scratch — only the single-token decode path
        // uses it. Prefill and batched-decode lm_head still write BF16 to the
        // shared `buffers.logits()`. Sized for one row of `vocab_size` FP32.
        let logits_fp32_buf = if use_fp32_logits {
            let bytes = config.vocab_size * 4;
            let p = gpu.alloc(bytes)?;
            tracing::info!(
                "FP32 LM head + softcap active (model_type={}, vocab={}). \
                 Decode logits scratch: {} bytes.",
                config.model_type,
                config.vocab_size,
                bytes,
            );
            p
        } else {
            DevicePtr::NULL
        };

        // Embedding scale (Gemma-4: sqrt(hidden_size)). Only load if model uses it.
        let embed_scale_kernel = if config.embed_scale > 0.0 {
            gpu.kernel("embed_scale", "bf16_scale_inplace")
                .unwrap_or_else(|e| {
                    tracing::warn!("embed_scale kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        if config.embed_scale > 0.0 {
            tracing::info!(
                "Embedding scale: {:.4} (sqrt({}))",
                config.embed_scale,
                config.hidden_size
            );
        }
        let ssm_norm_ptrs = if ssm_pool.num_ssm_layers > 0 {
            gpu.alloc(ssm_pool.num_ssm_layers * 8)
                .unwrap_or(DevicePtr::NULL)
        } else {
            DevicePtr::NULL
        };

        // GDN prefill buffers: sized for max_batch_tokens (the prefill chunk size),
        // NOT max_seq_len. For prompts longer than this, prefill_twophase falls back
        // to standard chunked prefill which carries h_state/conv_state between chunks.
        // The GDN recurrence is sequential anyway, so chunking is mathematically identical.
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        // GDN buffers only needed when GDN linear attention layers exist
        // (conv_dim > 0). Mamba-2 models (Nemotron) have conv_dim=0 — skip alloc
        // to avoid cuMemAlloc(0) error.
        let gdn_buf_len = max_batch_tokens.min(max_seq_len);
        let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z) = if conv_dim > 0 {
            let qkv = gpu.alloc(gdn_buf_len * conv_dim * 2)?;
            let gb = gpu.alloc(gdn_buf_len * nv * 2 * 4)?;
            let o = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let z = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let total_mb =
                (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
            tracing::info!(
                "GDN prefill buffers: {total_mb} MB for {gdn_buf_len} tokens (chunked SSM prefill)"
            );
            (qkv, gb, o, z)
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        // FP8 calibration only runs when the cache is actually FP8 — the
        // observe() call in decode.rs sits inside the FP8 cache branch. For
        // BF16 or NVFP4 caches the MODEL.toml fp8_kv_calibration_tokens
        // value is dead code and must not suppress CUDA graphs.
        let has_fp8_calibration = config.fp8_kv_calibration_tokens > 0
            && kv_cache.dtype() == spark_runtime::kv_cache::KvCacheDtype::Fp8;
        Ok(Self {
            config,
            embed_tokens,
            final_norm,
            lm_head_weight,
            lm_head_nvfp4,
            lm_head_fp8,
            layers,
            buffers,
            kv_cache: Mutex::new(kv_cache),
            gpu,
            rms_norm_kernel,
            bf16_to_f32_kernel,
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_logits_kernel,
            w4a16_gemm_kernel,
            w4a16_gemv_batch2_kernel,
            dense_gemv_fp8w_kernel,
            dense_gemm_kernel,
            argmax_kernel,
            argmax_logits_kernel,
            batched_embed_kernel,
            fill_slots_kernel,
            decode_graph: Mutex::new(std::collections::HashMap::new()),
            batch_decode_graphs: Mutex::new(HashMap::new()),
            // Suppress graphs during FP8 calibration only. MLA used to be
            // suppressed because an internal sync was placed inside the graph
            // capture region — that sync is now conditional on eager mode
            // (see line ~3881), so graphs work for MLA too. The zero_all call
            // at line ~3751 runs in Phase 1 BEFORE begin_capture, so it is
            // naturally outside the captured region.
            suppress_graphs: std::sync::atomic::AtomicBool::new(
                has_fp8_calibration
                    || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true"),
            ),
            ssm_pool,
            ssm_snapshots,
            max_blocks_per_seq,
            dummy_kv_block,
            profile,
            profile_first_pending: std::sync::atomic::AtomicBool::new(profile_first),
            proposer,
            mtp_hidden_save,
            dflash_hidden_save,
            dflash_capture_layers,
            verify2_graph: Mutex::new(std::collections::HashMap::new()),
            verify3_graph: Mutex::new(std::collections::HashMap::new()),
            verify4_graph: Mutex::new(std::collections::HashMap::new()),
            verify_kgamma_graph: Mutex::new(std::collections::HashMap::new()),
            prefix_cache,
            secondary_stream,
            secondary_event,
            comm,
            ep_cmd_buf,
            ep_protocol_v2: matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2")),
            self_speculative,
            last_mtp_hidden_idx: std::sync::atomic::AtomicUsize::new(0),
            vision_encoder,
            vision_embed_patches: Mutex::new(0),
            vision_image_grids: Mutex::new(Vec::new()),
            pinned_staging,
            ssm_checkpoint_interval,
            ssm_state_norm_kernel: ssm_norm_k,
            ssm_norm_ptrs_buf: ssm_norm_ptrs,
            gdn_buf_qkv: gdn_qkv,
            gdn_buf_gate_beta: gdn_gate_beta,
            gdn_buf_out: gdn_out,
            gdn_buf_z: gdn_z,
            gdn_buf_max_len: gdn_buf_len,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
        })
    }
}
