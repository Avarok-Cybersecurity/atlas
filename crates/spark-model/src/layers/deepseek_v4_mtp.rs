// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash Multi-Token-Prediction (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the `DeepseekV4MtpModule` loaded by
//! `load_v4_mtp_module`. Unlike the
//! Qwen-shaped [`crate::layers::MtpHead`] (a hand-rolled single attention +
//! MoE block), the V4 MTP module's body is a full reused V4 layer
//! (MLA + manifold-constrained hyper-connections (mHC) + 256-expert NVFP4
//! MoE). The proposer therefore delegates the bulk of the forward to
//! `body.decode()` and only wraps it with the MTP-specific pieces.
//!
//! Forward (`propose()`, K = 1 since `num_nextn_predict_layers == 1`):
//!
//! ```text
//!   embed   = embed_tokens[last_token]                       // [hidden] BF16
//!   h_in    = e_proj · rms_norm(embed,  enorm)
//!           + h_proj · rms_norm(hidden, hnorm)               // combiner
//!   hc_expand(h_in → hc_streams)                             // is_first mHC
//!   body.decode(hc_streams, …, mtp_kv_cache, state.seq_len)  // MIDDLE mHC + MLA + MoE
//!   hc_head(hc_streams → h_out)                              // is_last mHC
//!   logits  = lm_head(rms_norm(h_out, norm))
//!   draft   = argmax(logits)                                 // grammar-masked when Some
//! ```
//!
//! The body was assembled with `layer_idx = num_hidden_layers`, so its
//! `decode_inner_hc` sees `is_first_layer == false` AND `is_last_layer ==
//! false`: it runs the middle mHC mixing (hc_pre → attn → hc_post → hc_pre →
//! ffn → hc_post) reading/writing `hc_streams`, but does NOT call `hc_expand`
//! or `hc_head`. The proposer supplies both ends.
//!
//! ## Separate KV cache + distinct metadata offset
//!
//! The MTP attention writes into its OWN single-layer MLA-shaped
//! [`PagedKvCache`] (num_kv_heads = 1, head_dim = kv_lora_rank +
//! qk_rope_head_dim), never the target's. The V4-Flash decode attention
//! (`attention_forward_v4`) reads positions / slot / seq_len / block_table
//! from `ctx.attn_metadata`, so the proposer uploads MTP-specific metadata to
//! `scratch().offset(MTP_META_OFFSET)` — distinct from the target metadata at
//! `32768` — and threads it through a derived [`ForwardContext`].

use std::any::Any;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::deepseek_v4::DeepseekV4MtpModule;
use crate::weight_map::DenseWeight;

mod prefill;

/// KV dtype for the V4 MTP draft module.
///
/// Default **FP8**: V4 MLA decode (`run_paged_decode`) only implements the
/// V4-specific FP8 path (V=K rope reconstruction + attention sink). BF16
/// falls through the generic GQA path, which cannot handle V4's MLA layout
/// (`kv_lora_rank=512`, single KV head) and hits `CUDA_ERROR_ILLEGAL_ADDRESS`.
/// Override with `ATLAS_V4_MTP_KV_DTYPE=bf16` only for experiments.
pub fn v4_mtp_kv_dtype(layer_kv_dtypes: &[KvCacheDtype]) -> KvCacheDtype {
    match std::env::var("ATLAS_V4_MTP_KV_DTYPE").ok().as_deref() {
        Some("bf16") => KvCacheDtype::Bf16,
        _ => layer_kv_dtypes.last().copied().unwrap_or(KvCacheDtype::Fp8),
    }
}

/// Scratch-buffer byte offset for the MTP attention metadata. Must be distinct
/// from the target model's metadata (`32768`) so a `propose()` call does not
/// clobber the in-flight target `attn_metadata`. Mirrors the Qwen `MtpHead`
/// choice of `49152` (the Qwen head uploads its own packed header there too).
pub(super) const MTP_META_OFFSET: usize = 49152;

pub(super) fn v4_mtp_k1_state_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_V4_MTP_K1_STATE").ok().as_deref() == Some("1"))
}

fn rollback_plan(num_drafted: usize, num_accepted: usize, k1_state_enabled: bool) -> usize {
    let trimmable_rows = if k1_state_enabled {
        num_drafted.saturating_sub(1)
    } else {
        num_drafted
    };
    trimmable_rows.saturating_sub(num_accepted)
}

fn rollback_pair_key(last_pair_key: Option<usize>, num_to_trim: usize) -> Option<usize> {
    last_pair_key.map(|key| key.saturating_sub(num_to_trim))
}

/// Per-sequence state for the DeepSeek-V4 MTP proposer.
pub struct DeepseekV4MtpProposerState {
    /// Block table for the MTP module's OWN KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in the MTP KV cache.
    pub seq_len: usize,
    /// Drafts produced by the last `propose()` (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// Newest sequence-space pair key written into the drafter KV (for catch-up).
    pub last_pair_key: Option<usize>,
    /// Per-layer state for the reused V4 body. MLA attention layers use
    /// `EmptyLayerState`, but we allocate it via `body.alloc_state` so any
    /// future stateful body type is handled correctly (no hard-coded assumption).
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for DeepseekV4MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// DeepSeek-V4 MTP draft proposer.
pub struct DeepseekV4MtpHead {
    /// The loaded MTP module: reused V4 body + combiner + final norm + hc_head.
    module: DeepseekV4MtpModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared LM head (BF16 — DeepSeek-V4-Flash keeps the head in BF16), from the
    /// target model. Every draft is re-verified by the target's head, so the
    /// draft head only affects acceptance, never an accepted token.
    lm_head: DenseWeight,
    /// Reduced vocab size for the draft LM-head GEMV (0 = full vocab).
    mtp_vocab_size: u32,
    /// Single-layer MLA-shaped KV cache for the MTP attention.
    kv_cache: Mutex<PagedKvCache>,

    // Kernel handles.
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemm_pipelined_k: KernelHandle,
    batched_embed_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    mtp_hc_f32_to_bf16_k: KernelHandle,
    mtp_hproj_batch4_k: KernelHandle,
    mtp_hproj_broadcast_add_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl DeepseekV4MtpHead {
    /// Build the proposer from a loaded `DeepseekV4MtpModule` and the shared
    /// embedding + NVFP4 LM head.
    pub fn new(
        module: DeepseekV4MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        kv_dtype: KvCacheDtype,
    ) -> Result<Self> {
        // MTP KV cache: single MLA-absorbed attention layer. Matches the
        // target's MLA cache shape (num_kv_heads = 1, head_dim = kv_lora_rank
        // + qk_rope_head_dim) so `write_kv_cache` / `run_paged_decode` in the
        // reused V4 body land at the correct strides. The dtype MUST match
        // the main attention layers (FP8 in practice): the V4 MLA decode
        // kernel only implements the V=K rope reconstruction + attention sink
        // on the FP8 KV path, so a BF16 draft cache silently corrupts every
        // draft (~16% acceptance measured on 2× GB10 EP=2 before this fix —
        // the earlier BF16 choice here dodged a Qwen-path FP8 issue that does
        // not apply to the V4 MLA cache layout).
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        // Private single-layer MLA cache. The body keeps attn_layer_idx =
        // num_hidden_layers for compress/hash/mHC is_first/is_last defaults,
        // but its kv_layer_idx is remapped to 0 (see load_v4_mtp_module) so
        // this 1-pool cache is addressed correctly.
        let num_layers = 1;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: mla_cache_dim,
            num_layers,
            dtype: kv_dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            // V4 ships HF-vanilla norm weights (enorm/hnorm/norm are loaded
            // exactly) — the offset-from-1 kernel would apply `1 + w`.
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm_pipelined_k: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            batched_embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            mtp_hc_f32_to_bf16_k: gpu.kernel("mtp_combiner", "mtp_hc_f32_to_bf16_legacy")?,
            mtp_hproj_batch4_k: gpu.kernel("mtp_combiner", "mtp_hproj_gemv_batch4")?,
            mtp_hproj_broadcast_add_k: gpu
                .kernel("mtp_combiner", "mtp_hproj_broadcast_add_batched")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// Allocate per-sequence state. Mirrors the body's own `alloc_state` for
    /// the body sub-state.
    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<DeepseekV4MtpProposerState> {
        Ok(DeepseekV4MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            last_pair_key: None,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    /// One MTP draft step. Returns the drafted token id.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut DeepseekV4MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = ctx.config.hc_mult as u32;
        let row_bytes = h as usize * 2;

        // Diagnostic: isolate the current target-conditioned MTP row from the
        // proposer's compacted history. Atlas does not yet feed every target
        // position through the V4 MTP layer the way vLLM does, so retained rows
        // can represent a sparse and acceptance-dependent history.
        static RESET_KV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *RESET_KV
            .get_or_init(|| std::env::var("ATLAS_V4_MTP_RESET_KV").ok().as_deref() == Some("1"))
        {
            state.seq_len = 0;
            state.last_pair_key = None;
        }

        // ATLAS_V4_MTP_DIAG_PASSTHROUGH=1: bisect diagnostic. Skip the
        // combiner + body + collapse entirely and compute
        // `argmax(lm_head(rms_norm(target_hidden, norm)))`. At temperature 0
        // this must ECHO `token` (the target's own argmax that produced it)
        // nearly always — if it does, the shared embed/norm/GEMV/lm_head
        // plumbing is correct and the draft corruption is inside the
        // combiner/body path; if it does not, the head's own output path is
        // broken. Diagnostic only; never enable in a real serve.
        static DIAG_PASSTHROUGH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diag_passthrough = *DIAG_PASSTHROUGH.get_or_init(|| {
            std::env::var("ATLAS_V4_MTP_DIAG_PASSTHROUGH")
                .ok()
                .as_deref()
                == Some("1")
        });
        if diag_passthrough {
            let final_normed = ctx.buffers.norm_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                target_hidden,
                &self.module.norm,
                final_normed,
                1,
                h,
                eps,
                stream,
            )?;
            let v = if self.mtp_vocab_size > 0 {
                self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
            } else {
                ctx.config.vocab_size as u32
            };
            let logits = ctx.buffers.logits();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                final_normed,
                &self.lm_head,
                logits,
                v,
                h,
                stream,
            )?;
            let out_ptr = ctx.buffers.scratch().offset(64);
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            let echo = u32::from_le_bytes(buf);
            tracing::info!(
                "V4_MTP_DIAG passthrough: token={token} pos={position} echo={echo} match={}",
                echo == token
            );
            state.seq_len += 1;
            return Ok(echo);
        }

        // ── 1. Embed last token (D2D gather from the shared table) ──
        // Use attn_output (not ssm_qkvz) — attention-only models like DeepSeek-V4
        // have no SSM layers, so ssm_* buffers can be undersized for some uses.
        // attn_output is [M, num_heads, head_dim] BF16 — far larger than hidden.
        let embed_out = ctx.buffers.attn_output();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // ── 2. Combiner (reference MTPBlock, model.py) ──
        //
        // Reference:
        //   e = e_proj(enorm(embed(token)))                         # [H]
        //   x = h_proj(hnorm(x_streams)) + e.unsqueeze(stream)     # [hc, H]
        // where `x_streams` is the TARGET multi-stream residual
        // (`hc_streams` AFTER the last main block, BEFORE main hc_head).
        //
        // Atlas previously collapsed to a single BF16 hidden, then hc_expand
        // equal copies. That mismatch made Sinkhorn residuals explode
        // (measured absmax 1e6–3e6) and left CUDA error-prone state mid-propose.
        //
        // ATLAS_V4_MTP_SINGLE_STREAM=1 restores the old expand path (debug).
        let hc_streams = ctx.buffers.hc_streams();
        let hc_elems = (hc_mult as usize) * (h as usize);
        // Default single-stream expand: body runs without multi-stream mHC
        // (see decode_inner mtp_skip_mhc). Multi-stream combiner only when
        // ATLAS_V4_MTP_USE_MHC=1 (and optionally ATLAS_V4_MTP_SINGLE_STREAM=0).
        let use_mhc = std::env::var("ATLAS_V4_MTP_USE_MHC").ok().as_deref() != Some("0");
        let single_stream =
            !use_mhc || std::env::var("ATLAS_V4_MTP_SINGLE_STREAM").ok().as_deref() == Some("1");

        // e_branch = e_proj(enorm(embed)) into h_in (BF16 [H]).
        let normed_embed = ctx.buffers.moe_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.module.enorm,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        let e_branch = ctx.buffers.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_embed,
            &self.module.e_proj,
            e_branch,
            h,
            h,
            stream,
        )?;

        if single_stream {
            // Legacy path: hnorm(target_hidden) → h_proj → + e → hc_expand.
            let normed_hidden = ctx.buffers.residual();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                target_hidden,
                &self.module.hnorm,
                normed_hidden,
                1,
                h,
                eps,
                stream,
            )?;
            let h_branch = ctx.buffers.norm_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_hidden,
                &self.module.h_proj,
                h_branch,
                h,
                h,
                stream,
            )?;
            // e_branch currently holds e; residual_add does e += h_branch → h_in.
            ops::residual_add(ctx.gpu, self.residual_add_k, e_branch, h_branch, h, stream)?;
            if use_mhc {
                ops::hc_expand(
                    ctx.gpu,
                    self.hc_expand_k,
                    e_branch,
                    hc_streams,
                    1,
                    h,
                    hc_mult,
                    stream,
                )?;
            }
        } else {
            // Multi-stream path (default, matches DeepSeek reference):
            // For each stream i: streams_f32[i] = h_proj(hnorm(bf16(streams_f32[i]))) + e
            static GPU_COMBINER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let gpu_combiner = *GPU_COMBINER.get_or_init(|| {
                std::env::var("ATLAS_V4_MTP_GPU_COMBINER").ok().as_deref() == Some("1")
            });
            if gpu_combiner {
                anyhow::ensure!(
                    hc_mult <= 4,
                    "V4 MTP GPU combiner supports hc_mult <= 4, got {hc_mult}"
                );
                let streams_bf16 = ctx.buffers.residual();
                let normed = ctx.buffers.norm_output();
                ops::mtp_hc_f32_to_bf16_legacy(
                    ctx.gpu,
                    self.mtp_hc_f32_to_bf16_k,
                    hc_streams,
                    streams_bf16,
                    hc_elems as u32,
                    stream,
                )?;
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    streams_bf16,
                    &self.module.hnorm,
                    normed,
                    hc_mult,
                    h,
                    eps,
                    stream,
                )?;
                ops::mtp_hproj_gemv_batch4(
                    ctx.gpu,
                    self.mtp_hproj_batch4_k,
                    normed,
                    &self.module.h_proj,
                    e_branch,
                    hc_streams,
                    hc_mult,
                    h,
                    h,
                    stream,
                )?;
            } else {
                ctx.gpu.synchronize(stream)?;
                let mut streams_f32 = vec![0f32; hc_elems];
                {
                    let bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            streams_f32.as_mut_ptr() as *mut u8,
                            hc_elems * 4,
                        )
                    };
                    ctx.gpu.copy_d2h(hc_streams, bytes)?;
                }
                // Detect dead/garbage streams → fall back to expand(target_hidden).
                let nan = streams_f32.iter().filter(|v| !v.is_finite()).count();
                let absmax = streams_f32.iter().fold(0f32, |m, v| m.max(v.abs()));
                let streams_ok = nan == 0 && absmax > 1e-8 && absmax < 1e4;
                if !streams_ok {
                    tracing::debug!(
                        "V4 MTP multi-stream residual unusable (nan={nan} absmax={absmax:.3});                      falling back to expand(target_hidden)"
                    );
                    let normed_hidden = ctx.buffers.residual();
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        target_hidden,
                        &self.module.hnorm,
                        normed_hidden,
                        1,
                        h,
                        eps,
                        stream,
                    )?;
                    let h_branch = ctx.buffers.norm_output();
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_hidden,
                        &self.module.h_proj,
                        h_branch,
                        h,
                        h,
                        stream,
                    )?;
                    // e_branch holds e; add h_branch, then expand.
                    ops::residual_add(ctx.gpu, self.residual_add_k, e_branch, h_branch, h, stream)?;
                    ops::hc_expand(
                        ctx.gpu,
                        self.hc_expand_k,
                        e_branch,
                        hc_streams,
                        1,
                        h,
                        hc_mult,
                        stream,
                    )?;
                } else {
                    // Per-stream: F32→BF16 → hnorm → h_proj → +e → BF16→F32.
                    // Single-row GPU workspaces only (safe for max_batch_tokens=1).
                    let row_bf16 = ctx.buffers.residual();
                    let normed = ctx.buffers.norm_output();
                    let h_branch = ctx.buffers.moe_output();
                    for i in 0..hc_mult as usize {
                        let base = i * (h as usize);
                        let mut bf16_row = vec![0u8; row_bytes];
                        for d in 0..(h as usize) {
                            let bits = streams_f32[base + d].to_bits();
                            let rounded = bits.wrapping_add(0x8000) >> 16;
                            let bf = (rounded as u16).to_le_bytes();
                            bf16_row[d * 2] = bf[0];
                            bf16_row[d * 2 + 1] = bf[1];
                        }
                        ctx.gpu.copy_h2d_async(&bf16_row, row_bf16, stream)?;
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            row_bf16,
                            &self.module.hnorm,
                            normed,
                            1,
                            h,
                            eps,
                            stream,
                        )?;
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed,
                            &self.module.h_proj,
                            h_branch,
                            h,
                            h,
                            stream,
                        )?;
                        // h_branch += e_branch
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            h_branch,
                            e_branch,
                            h,
                            stream,
                        )?;
                        ctx.gpu.synchronize(stream)?;
                        let mut out_bf16 = vec![0u8; row_bytes];
                        ctx.gpu.copy_d2h(h_branch, &mut out_bf16)?;
                        for d in 0..(h as usize) {
                            let hi = u16::from_le_bytes([out_bf16[d * 2], out_bf16[d * 2 + 1]]);
                            streams_f32[base + d] = f32::from_bits((hi as u32) << 16);
                        }
                    }
                    let bytes = unsafe {
                        std::slice::from_raw_parts(streams_f32.as_ptr() as *const u8, hc_elems * 4)
                    };
                    ctx.gpu.copy_h2d_async(bytes, hc_streams, stream)?;
                }
            }
        }

        // ATLAS_V4_MTP_DIAG_DUMP=1: per-stage magnitude stats for the first
        // few drafts. Sync-heavy; diagnostic only.
        static DIAG_DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static DUMP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let diag_dump = *DIAG_DUMP
            .get_or_init(|| std::env::var("ATLAS_V4_MTP_DIAG_DUMP").ok().as_deref() == Some("1"))
            && DUMP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6;
        let dump_stats = |label: &str, ptr: DevicePtr| -> Result<()> {
            if !diag_dump {
                return Ok(());
            }
            ctx.gpu.synchronize(stream)?;
            let n = h as usize;
            let mut raw = vec![0u8; n * 2];
            ctx.gpu.copy_d2h(ptr, &mut raw)?;
            let vals: Vec<f32> = raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let absmax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
            let l2 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nan = vals.iter().filter(|v| !v.is_finite()).count();
            tracing::info!(
                "V4_MTP_DUMP {label} (bf16): absmax={absmax:.4} l2={l2:.2} nan={nan} head=[{:.4},{:.4},{:.4},{:.4}]",
                vals[0],
                vals[1],
                vals[2],
                vals[3]
            );
            Ok(())
        };
        let dump_f32 = |label: &str, ptr: DevicePtr, n_elems: usize| -> Result<()> {
            if !diag_dump {
                return Ok(());
            }
            ctx.gpu.synchronize(stream)?;
            let mut vals = vec![0f32; n_elems];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(vals.as_mut_ptr() as *mut u8, n_elems * 4)
            };
            ctx.gpu.copy_d2h(ptr, bytes)?;
            let absmax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
            let l2 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nan = vals.iter().filter(|v| !v.is_finite()).count();
            let head = if vals.len() >= 4 {
                format!(
                    "[{:.4},{:.4},{:.4},{:.4}]",
                    vals[0], vals[1], vals[2], vals[3]
                )
            } else {
                format!("{:?}", &vals[..vals.len().min(4)])
            };
            tracing::info!(
                "V4_MTP_DUMP {label} (f32 n={n_elems}): absmax={absmax:.4} l2={l2:.2} nan={nan} head={head}"
            );
            Ok(())
        };
        dump_stats("target_hidden", target_hidden)?;
        dump_stats("embed_out", embed_out)?;
        dump_stats("e_branch", e_branch)?;
        dump_f32("hc_streams_after_combiner", hc_streams, hc_elems)?;

        // ── 4. Body decode: MIDDLE mHC + MLA attention (writes MTP KV cache)
        //       + MoE. Reads/writes `hc_streams` (hidden is a single-stream
        //       scratch). The body NEVER calls hc_expand/hc_head (layer_idx =
        //       num_hidden_layers ⇒ is_first == is_last == false). ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        // Upload MTP-specific attention metadata at a scratch offset that:
        //  (1) does not clobber target decode meta at 32768..33536
        //  (2) has room for the full block table (grows with mtp seq_len)
        //  (3) stays inside the scratch allocation (CUDA-700 if it doesn't)
        // Prefer MTP_META_OFFSET=49152 (mirrors Qwen MtpHead); if the growing
        // block table would overrun, slide the base back from the end of
        // scratch so the write always fits.
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;

        let bt_i32: Vec<i32> = state.block_table.iter().map(|&b| b as i32).collect();
        let bt_len = bt_i32.len() * 4;
        let meta_bytes = 256 + bt_len;
        let scratch_bytes = ctx.buffers.scratch_bytes();
        let meta_off = if MTP_META_OFFSET + meta_bytes <= scratch_bytes {
            MTP_META_OFFSET
        } else if meta_bytes + 64 <= scratch_bytes {
            // Keep clear of the low 64-byte MoE/argmax region.
            scratch_bytes - meta_bytes
        } else {
            anyhow::bail!("V4 MTP metadata ({meta_bytes} B) exceeds scratch ({scratch_bytes} B)");
        };
        let meta_base = ctx.buffers.scratch().offset(meta_off);
        let mut meta_buf = vec![0u8; meta_bytes];
        meta_buf[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
        meta_buf[16..20].copy_from_slice(&actual_seq_len.to_le_bytes());
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_len) };
        meta_buf[256..256 + bt_len].copy_from_slice(bt_bytes);
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot: spark_runtime::gpu::DevicePtr(0),
        };

        // The body's hash-MoE (if any) reads the decode token id from
        // `token_ids[0]`; upload this draft's input token there. The main
        // decode loop uploaded the target token earlier in the step, so we
        // must overwrite it for the MTP forward (and the main loop re-uploads
        // before the next target step / graph replay).
        if let Some(tid_buf) = ctx.token_ids {
            ctx.gpu
                .copy_h2d_async(&token.to_le_bytes(), tid_buf, stream)?;
        }

        // Derive a ForwardContext carrying the MTP metadata. CUDA-graph capture
        // is forced off for the MTP forward (its block-table / metadata are
        // host-built per call and the H2D uploads above are illegal under
        // capture).
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            // comm = None: the MTP draft runs ONLY on rank 0, so its MoE must NOT
            // issue an EP all-reduce (rank 1 never participates → the collective
            // hangs ~35s then corrupts CUDA). The MTP body is loaded with ALL
            // experts local (force_all_experts), so the no-EP MoE is correct.
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            routed_lora_layers: None, // #30: MTP draft body; no prefill LoRA route.
            midchunk_capture: None,
        };

        // `decode_inner_hc` reads the persistent multi-stream state from
        // `ctx.buffers.hc_streams()` directly (already populated by `hc_expand`
        // above) and uses the `hidden` ARG as a single-stream scratch (hc_pre
        // collapses into it). So `hidden` must be a SEPARATE buffer, NOT
        // `hc_streams` — aliasing them corrupts the persistent state. Reuse
        // `hidden_states()` (= the now-consumed `h_in` scratch).
        let body_scratch = ctx.buffers.hidden_states();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; ctx.config.num_hidden_layers + 1];
        let residual = ctx.buffers.residual();
        self.module.body.decode(
            body_scratch,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        // Surface async illegal-address faults at the body boundary (CUDA-700
        // was previously deferred to the next propose/verify sync).
        ctx.gpu.synchronize(stream).map_err(|e| {
            anyhow::anyhow!(
                "V4 MTP body.decode sync failed (pos={position} mtp_seq={}): {e}",
                state.seq_len
            )
        })?;
        drop(kv_cache);

        // BISECT: body outputs. hc_streams is F32 highway — use dump_f32.
        dump_f32("body_hc_streams", hc_streams, hc_elems)?;
        dump_stats("body_scratch", body_scratch)?;
        dump_stats("body_residual", residual)?;

        // ── 4b. mHC highway recovery (Bug 3 — measured real F32 blowup) ──
        // True F32 dumps on EP=2 (2026-07-24): body_scratch absmax~1–2 (fine)
        // but body_hc_streams absmax ~1e6–3e6 with thousands of NaNs. Clip
        // sanitize left ~70% of elements at ±1e4 → h_out still ~1e4 garbage.
        // When the multi-stream residual is corrupt, rebuild streams from the
        // good single-stream body output via hc_expand, then collapse with
        // hc_head (equal streams ≈ body_scratch). Disable recovery with
        // ATLAS_V4_MTP_NO_HC_RECOVER=1.
        static HC_RECOVER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // Skip recovery when multi-stream mHC is off (default MTP path): streams
        // still hold the *target* residual (absmax often 100+) which would
        // false-trigger recovery every step and burn a D2H sync.
        // Default OFF: the check forces sync+D2H every draft step.
        // Enable with ATLAS_V4_MTP_HC_RECOVER=1 if streams corrupt.
        let do_recover = use_mhc
            && *HC_RECOVER.get_or_init(|| {
                let explicit_on =
                    std::env::var("ATLAS_V4_MTP_HC_RECOVER").ok().as_deref() == Some("1");
                let explicit_off =
                    std::env::var("ATLAS_V4_MTP_NO_HC_RECOVER").ok().as_deref() == Some("1");
                explicit_on && !explicit_off
            });
        let mut recovered = false;
        if do_recover {
            ctx.gpu.synchronize(stream)?;
            let mut vals = vec![0f32; hc_elems];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(vals.as_mut_ptr() as *mut u8, hc_elems * 4)
            };
            ctx.gpu.copy_d2h(hc_streams, bytes)?;
            let nan = vals.iter().filter(|v| !v.is_finite()).count();
            let absmax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
            // Healthy residual streams are O(1–10); >64 is already pathological.
            const ABSMAX_OK: f32 = 64.0;
            if nan > 0 || absmax > ABSMAX_OK {
                tracing::warn!(
                    "V4_MTP hc_streams corrupt (nan={nan} absmax={absmax:.1});                      rebuilding from body_scratch via hc_expand"
                );
                ops::hc_expand(
                    ctx.gpu,
                    self.hc_expand_k,
                    body_scratch,
                    hc_streams,
                    1,
                    h,
                    hc_mult,
                    stream,
                )?;
                recovered = true;
                dump_f32("body_hc_streams_recovered", hc_streams, hc_elems)?;
            }
        }

        // ── 5. Collapse to h_out ──
        // Prefer hc_head when streams look usable; if we just recovered from
        // body_scratch expand, hc_head on equal streams is fine. As a final
        // belt-and-suspenders, allow ATLAS_V4_MTP_SKIP_HC_HEAD=1 to copy
        // body_scratch directly (bypass mHC entirely).
        let h_out = ctx.buffers.hidden_states();
        let skip_head = std::env::var("ATLAS_V4_MTP_SKIP_HC_HEAD").ok().as_deref() == Some("1");
        // When streams had to be recovered, prefer the body's single-stream
        // residual (pre-ffn collapse is still a better attractor than a
        // just-re-expanded highway). Opt out with ATLAS_V4_MTP_USE_HC_HEAD_AFTER_RECOVER=1.
        // Without multi-stream mHC on the body, the residual result lives in
        // body_scratch (= hidden). Skip hc_head collapse.
        let force_body = !use_mhc
            || (recovered
                && std::env::var("ATLAS_V4_MTP_USE_HC_HEAD_AFTER_RECOVER")
                    .ok()
                    .as_deref()
                    != Some("1"));
        if skip_head || force_body {
            ctx.gpu
                .copy_d2d_async(body_scratch, h_out, row_bytes, stream)?;
        } else if let Some(ref head) = self.module.hc_head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                h_out,
                1,
                h,
                hc_mult,
                eps,
                ctx.config.hc_eps,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2d_async(hc_streams, h_out, row_bytes, stream)?;
        }

        dump_stats("h_out(collapsed)", h_out)?;

        // ── 6. Final norm + shared LM head → logits ──
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_out,
            &self.module.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            final_normed,
            &self.lm_head,
            logits,
            v,
            h,
            stream,
        )?;

        // ── 7. Argmax (grammar-masked when a bitmask is supplied) ──
        // Park the result at scratch+64 so we never stomp the low MoE routing
        // region or any mid-scratch MTP meta that landed near the head.
        let out_ptr = ctx.buffers.scratch().offset(64);
        let token_id = if let Some(bitmask) = grammar_bitmask {
            argmax_grammar_masked(ctx.gpu, logits, v as usize, bitmask, position)?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        state.last_pair_key = Some(position);
        Ok(token_id)
    }
}

fn argmax_grammar_masked(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    bitmask: &[i32],
    position: usize,
) -> Result<u32> {
    let mut bf16_buf = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut bf16_buf)?;

    let mut best_tok = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let mut any_allowed = false;
    for tok in 0..vocab {
        let word = tok / 32;
        let bit = tok % 32;
        let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
        if !allowed {
            continue;
        }
        any_allowed = true;
        // BF16 → f32: BF16 is the upper 16 bits of an f32.
        let hi = u16::from_le_bytes([bf16_buf[2 * tok], bf16_buf[2 * tok + 1]]);
        let val = f32::from_bits((hi as u32) << 16);
        if val > best_val {
            best_val = val;
            best_tok = tok as u32;
        }
    }
    if !any_allowed {
        tracing::warn!(
            "V4 MTP grammar mask allowed zero tokens at pos {position}; \
             returning 0 as pad-draft (will be rejected at verify)."
        );
        return Ok(0);
    }
    Ok(best_tok)
}

impl DraftProposer for DeepseekV4MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;
        for i in 0..num_drafts {
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "V4 MTP grammar-masked drafting with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            // The scheduler passes the target sequence length after the hidden
            // row was produced. V4 MTP pairs that hidden with the next token
            // at the hidden row's position, one less than that length.
            let row_position = if v4_mtp_k1_state_enabled() {
                position.saturating_sub(1) + i
            } else {
                position + i
            };
            let draft = self.forward_one(
                current_token,
                current_hidden,
                row_position,
                v4_state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            tracing::debug!(
                "V4 MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                row_position,
                v4_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // Subsequent drafts feed on the MTP head's own collapsed hidden.
            current_hidden = ctx.buffers.hidden_states();
        }
        v4_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .and_then(|s| s.last_pair_key)
    }

    fn take_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<(Vec<u32>, usize, Option<usize>)> {
        let st = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()?;
        if st.block_table.is_empty() || st.seq_len == 0 {
            return None;
        }
        let blocks = std::mem::take(&mut st.block_table);
        let rows = st.seq_len;
        let key = st.last_pair_key;
        st.seq_len = 0;
        st.last_pair_key = None;
        st.last_num_drafted = 0;
        Some((blocks, rows, key))
    }

    fn install_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
        blocks: Vec<u32>,
        rows: usize,
        last_pair_key: Option<usize>,
    ) -> bool {
        let Some(st) = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
        else {
            return false;
        };
        if !st.block_table.is_empty() || st.seq_len != 0 {
            return false;
        }
        st.block_table = blocks;
        st.seq_len = rows;
        st.last_pair_key = last_pair_key;
        true
    }

    fn free_drafter_kv(&self, blocks: &[u32]) {
        if !blocks.is_empty() {
            self.kv_cache.lock().free_blocks(blocks);
        }
    }

    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        // Opt-in: full-layer catchup via forward_one is expensive and was
        // observed to trip CUDA_ERROR_ILLEGAL_ADDRESS under EP=2 until the
        // mHC residual path is fixed. Enable with ATLAS_V4_MTP_CATCHUP=1.
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED
            .get_or_init(|| std::env::var("ATLAS_V4_MTP_CATCHUP").ok().as_deref() == Some("1"))
        {
            return Ok(0);
        }
        let v4_state = match state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
        {
            Some(s) => s,
            None => return Ok(0),
        };
        if v4_state.seq_len != row_base || tokens.len() < 2 {
            return Ok(0);
        }
        let h = ctx.config.hidden_size;
        let rows = tokens.len() - 1;
        let t0 = std::time::Instant::now();
        for r in 0..rows {
            let tok = tokens[r + 1];
            let hidden = hiddens.offset(r * h * 2);
            let pos = pos_base + r;
            let _draft = self.forward_one(tok, hidden, pos, v4_state, ctx, stream, None)?;
        }
        tracing::info!(
            "V4 MTP catchup_drafter: wrote {rows} rows in {:.1} ms (row_base={row_base} pos_base={pos_base})",
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(rows)
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.catchup_drafter(prompt_tokens, hiddens, 0, 1, state, ctx, stream)
    }

    fn prefill_v4_stream_rows(
        &self,
        next_tokens: &[u32],
        target_streams: DevicePtr,
        first_position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        self.prefill_stream_rows_inner(
            next_tokens,
            target_streams,
            first_position,
            v4_state,
            ctx,
            stream,
        )
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        // Trim `drafted - accepted` rejected entries from the MTP KV cache by
        // rolling back `seq_len` (the slots are overwritten on the next
        // propose). Mirrors `MtpHead::after_verify`.
        let num_drafted = v4_state.last_num_drafted.max(1);
        // Row 0 is conditioned on an already target-verified token and is
        // valid even when its predicted draft is rejected. Only recursive
        // rows 1.. are speculative inputs. The legacy formula incorrectly
        // dropped row 0 on every K1 rejection.
        let num_to_trim = rollback_plan(num_drafted, num_accepted, v4_mtp_k1_state_enabled());
        let old_sl = v4_state.seq_len;
        if num_to_trim > 0 {
            v4_state.seq_len = v4_state.seq_len.saturating_sub(num_to_trim);
            v4_state.last_pair_key = rollback_pair_key(v4_state.last_pair_key, num_to_trim);
        }
        tracing::debug!(
            "V4 MTP after_verify: accepted={num_accepted} drafted={num_drafted} \
             trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            v4_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        if !v4_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&v4_state.block_table);
            v4_state.block_table.clear();
        }
        v4_state.seq_len = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{rollback_pair_key, rollback_plan};

    #[test]
    fn k1_rejection_keeps_target_conditioned_row() {
        assert_eq!(rollback_plan(1, 0, true), 0);
    }

    #[test]
    fn recursive_rejections_roll_back_rows_and_pair_keys() {
        assert_eq!(rollback_plan(3, 1, true), 1);
        assert_eq!(rollback_plan(3, 0, false), 3);
        assert_eq!(rollback_pair_key(Some(17), 3), Some(14));
        assert_eq!(rollback_pair_key(Some(1), 3), Some(0));
    }

    #[test]
    fn accepted_rows_are_never_trimmed() {
        assert_eq!(rollback_plan(2, 2, true), 0);
        assert_eq!(rollback_plan(2, 2, false), 0);
    }
}
