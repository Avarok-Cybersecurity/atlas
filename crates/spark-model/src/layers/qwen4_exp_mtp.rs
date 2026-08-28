// SPDX-License-Identifier: AGPL-3.0-only

//! qwen4_exp MTP draft head — the combiner + body forward.
//!
//! ```text
//!   normed  = grouped_rms_norm(target_hc_streams, pre_fc_norm_hidden)  // [hc,H]
//!   streams = fc_hidden · normed
//!           + broadcast(fc_embedding · rms_norm(embed, pre_fc_norm_embedding))
//!   body.decode(streams, …, own kv cache, seq_len)     // MIDDLE mHC + QSA + MoE
//!   h_out   = hc_head(streams)                          // is_last mHC
//! ```
//!
//! The caller finishes with its own `final_norm_apply` + `lm_head` + argmax, so
//! this file never duplicates the target's LM-head quantization handling.
//!
//! # Why the combiner consumes STREAMS, not a collapsed hidden
//!
//! DeepSeek-V4's MTP norms a collapsed `[H]` hidden and then calls `hc_expand`
//! to replicate it into streams. qwen4_exp does NOT: its
//! `mtp.pre_fc_norm_hidden` is `[hc_mult*hidden]` = `[10240]`, and a norm
//! weight's width is what it normalizes. `HcLowRank::norm_w` documents that
//! exact shape as "a GROUPED RMSNorm scale: the streams normalize
//! independently inside the vector, group_size = hidden" — so this combiner
//! takes the four-stream highway directly and its OUTPUT is the body's
//! highway. There is no `hc_expand` on this path.
//!
//! That reading is what makes `fc_hidden [2560,2560]` consistent: it is applied
//! PER STREAM, not to a 10240-wide vector (which it could not consume). The
//! alternative — collapse-then-expand — would require `pre_fc_norm_hidden` to
//! be `[2560]`. It is not.
//!
//! ⚠ UNVERIFIED AGAINST A GOLDEN. HF's `modeling_qwen4_exp.py` carries
//! `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and ships no MTP class,
//! so no reference activations can be generated. The argument above is
//! structural, not numerical. `ATLAS_QWEN4EXP_MTP_SHADOW=1` exists to settle it
//! empirically: a wrong combiner costs ~100% of ACCEPTANCE and 0% of
//! correctness, so a near-zero shadow accept rate falsifies it.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;
use crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule;
use crate::weight_map::DenseWeight;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

/// Scratch offset for the MTP body's attention metadata. Mirrors the
/// DeepSeek-V4 head's use of a DISTINCT offset so the draft forward does not
/// clobber the target metadata the main decode wrote at 32768.
const MTP_META_OFFSET: usize = 40960;

/// Per-sequence MTP draft state.
pub struct Qwen4ExpMtpState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub body_state: Box<dyn LayerState>,
    /// Shadow mode: the token drafted at the PREVIOUS decode step, awaiting
    /// comparison against the token the target actually emits this step.
    pub pending_draft: Option<u32>,
}

/// Private device buffers. The head owns every buffer it writes so it cannot
/// alias the target's live state — the failure mode that "does not error, it
/// yields ~0% accept".
struct MtpBuffers {
    /// `[hc_mult * hidden]` — the target's streams, copied in.
    streams: DevicePtr,
    /// `[hc_mult * hidden]` — grouped-normed streams.
    normed_streams: DevicePtr,
    /// `[hidden]` scratch: embedding, its norm, its projection.
    embed: DevicePtr,
    normed_embed: DevicePtr,
    embed_proj: DevicePtr,
    /// `[hidden]` single-stream scratch the body collapses into.
    body_scratch: DevicePtr,
    residual: DevicePtr,
    /// `[hc_mult * hidden]` low-rank head scratch.
    head_scratch: DevicePtr,
}

pub struct Qwen4ExpMtpHead {
    module: Qwen4ExpMtpModule,
    embed_tokens: DenseWeight,
    kv_cache: Mutex<PagedKvCache>,
    buf: MtpBuffers,
    rms_norm_k: KernelHandle,
    rms_norm_strided_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_head_k: KernelHandle,
    /// Shadow counters: drafts made, drafts that matched the target's token.
    shadow_drafts: AtomicU64,
    shadow_hits: AtomicU64,
}

/// `ATLAS_QWEN4EXP_MTP_SHADOW=1` — run the draft head alongside normal decode
/// and log how often its draft matches the token the target actually emits.
/// Produces NO speculation: nothing is fed back, the scheduler is untouched.
pub fn shadow_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_SHADOW").as_deref() == Ok("1"))
}

impl Qwen4ExpMtpHead {
    pub fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let hc = config.hc_mult.max(1);
        let row = h * 2;
        let streams_bytes = hc * row;

        // The body was built with `attn_idx = 0`, so this pool needs exactly
        // ONE layer — and the body must NEVER be handed the main model's pool,
        // where index 0 is full-attention layer 0's live K/V.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = max_seq_len / kv_config.block_size + 2;
        let (bs, kvh, hd) = (
            kv_config.block_size,
            kv_config.num_kv_heads,
            kv_config.head_dim,
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        tracing::info!(
            "qwen4_exp MTP head: private KV pool {} blocks x {} tok = {} tokens, \
             {} kv_heads x {} head_dim BF16 (~{:.2} GB). This is allocated AFTER \
             the main pool and is therefore OUTSIDE the util pledge.",
            num_blocks,
            bs,
            num_blocks * bs,
            kvh,
            hd,
            (num_blocks * bs * kvh * hd * 2 * 2) as f64 / 1e9,
        );

        Ok(Self {
            module,
            embed_tokens,
            kv_cache: Mutex::new(kv_cache),
            buf: MtpBuffers {
                streams: gpu.alloc(streams_bytes)?,
                normed_streams: gpu.alloc(streams_bytes)?,
                embed: gpu.alloc(row)?,
                normed_embed: gpu.alloc(row)?,
                embed_proj: gpu.alloc(row)?,
                body_scratch: gpu.alloc(row)?,
                residual: gpu.alloc(row)?,
                head_scratch: gpu.alloc(streams_bytes.max(row * 4))?,
            },
            // Atlas's offset-from-1 rms_norm, NOT V4's `rms_norm_vanilla`:
            // this checkpoint's norm weights are offset-from-1 like the rest of
            // the qwen4_exp tree.
            rms_norm_k: gpu.kernel("rms_norm", "rms_norm")?,
            rms_norm_strided_k: gpu.kernel("rms_norm", "rms_norm_strided")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            shadow_drafts: AtomicU64::new(0),
            shadow_hits: AtomicU64::new(0),
        })
    }

    pub fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Qwen4ExpMtpState> {
        Ok(Qwen4ExpMtpState {
            block_table: Vec::new(),
            seq_len: 0,
            body_state: self.module.body.alloc_state(gpu)?,
            pending_draft: None,
        })
    }

    /// Record a shadow observation and return the running accept rate.
    pub fn shadow_observe(&self, drafted: Option<u32>, actual: u32) {
        if let Some(d) = drafted {
            self.shadow_drafts.fetch_add(1, Ordering::Relaxed);
            if d == actual {
                self.shadow_hits.fetch_add(1, Ordering::Relaxed);
            }
            let n = self.shadow_drafts.load(Ordering::Relaxed);
            if n.is_multiple_of(32) {
                let hits = self.shadow_hits.load(Ordering::Relaxed);
                tracing::info!(
                    "qwen4_exp MTP shadow: {hits}/{n} drafts matched the target \
                     ({:.1}% accept). NO speculation is running — this measures \
                     whether the combiner reading is right.",
                    100.0 * hits as f64 / n as f64
                );
            }
        }
    }

    /// One draft step. Writes the draft's final hidden state (post-mHC-head,
    /// pre-LM-head) into `h_out`; the caller applies its own final norm and LM
    /// head. `target_streams` is the target's four-stream highway for the
    /// position that just produced `last_token`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_hidden(
        &self,
        last_token: u32,
        target_streams: DevicePtr,
        position: usize,
        state: &mut Qwen4ExpMtpState,
        h_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let hc = ctx.config.hc_mult.max(1) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row = h as usize * 2;

        // Copy the target's streams into private memory FIRST. The body writes
        // through its own highway, and the target's buffer must not be touched.
        ctx.gpu
            .copy_d2d_async(target_streams, self.buf.streams, hc as usize * row, stream)?;

        // ── 1. Embedding branch ──
        let src = self.embed_tokens.weight.offset(last_token as usize * row);
        ctx.gpu.copy_d2d_async(src, self.buf.embed, row, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            self.buf.embed,
            &self.module.pre_fc_norm_embedding,
            self.buf.normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.buf.normed_embed,
            &self.module.fc_embedding,
            self.buf.embed_proj,
            h,
            h,
            stream,
        )?;

        // ── 2. Hidden branch: GROUPED norm, then fc_hidden PER STREAM ──
        // `rms_norm_strided` applies ONE weight to every row it normalizes, but
        // each stream owns its own `[hidden]` slice of the `[hc*hidden]` scale
        // — so this is one call per stream, each over that stream's rows with
        // `row_stride = hc*hidden`. At T=1 that is a single row per call.
        for i in 0..hc as usize {
            let off = i * row;
            let w = DenseWeight {
                weight: self.module.pre_fc_norm_hidden.weight.offset(off),
            };
            ops::rms_norm_strided(
                ctx.gpu,
                self.rms_norm_strided_k,
                self.buf.streams.offset(off),
                &w,
                self.buf.normed_streams.offset(off),
                1,
                1,
                h,
                eps,
                hc * h,
                stream,
            )?;
            // fc_hidden is shared across streams; apply it to this stream and
            // add the (broadcast) embedding projection in place.
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                self.buf.normed_streams.offset(off),
                &self.module.fc_hidden,
                self.buf.streams.offset(off),
                h,
                h,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                self.buf.streams.offset(off),
                self.buf.embed_proj,
                h,
                stream,
            )?;
        }

        // ── 3. Body decode against the module's OWN cache ──
        let mut kv_cache = self.kv_cache.lock().expect("mtp kv cache poisoned");
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        // Same layout + same shared packer the DeepSeek-V4 head uses, at a
        // DISTINCT scratch offset so this never clobbers the target metadata.
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let meta_buf = super::mtp_meta::pack_mtp_attn_meta(
            position as u32,
            global_slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let meta = crate::layer::AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };

        let mtp_ctx = ForwardContext {
            attn_metadata: Some(meta),
            // The draft body must not issue an EP all-reduce: it is rank-0 only
            // and `ensure_loadable` refuses ep_world_size > 1 outright.
            comm: None,
            // Host-built metadata + H2D uploads are illegal under capture.
            graph_capture: false,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            ..*ctx
        };

        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        self.module.body.decode(
            self.buf.body_scratch,
            self.buf.residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // ── 4. mHC head: collapse the module's streams → h_out ──
        // qwen4_exp's head is LOW-RANK, which is exactly the arm DeepSeek-V4's
        // MTP asserts against; call the low-rank collapse directly.
        let head = self
            .module
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp MTP: module has no hc_head"))?;
        let lowrank = head.lowrank.as_ref().ok_or_else(|| {
            anyhow::anyhow!("qwen4_exp MTP: hc_head is not low-rank; this model's is")
        })?;
        ops::hc_head_lowrank(
            ctx.gpu,
            self.hc_head_k,
            self.buf.streams,
            lowrank,
            h_out,
            self.buf.head_scratch,
            1,
            h,
            hc,
            eps,
            stream,
        )?;

        state.seq_len += 1;
        Ok(())
    }
}
