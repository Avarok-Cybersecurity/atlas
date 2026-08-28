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
//! structural, not numerical.
//!
//! # STATUS: NOT WORKING. The SHARING MODEL is wrong — do not keep patching.
//!
//! **1. The body forward corrupts the target, and buffer-by-buffer isolation
//! does NOT fix it.** Bisect via `ATLAS_QWEN4EXP_MTP_SHADOW_STAGE` (250-token
//! greedy completion against a shadow-off control):
//! ```text
//!   observe  -> output matches control exactly      INERT
//!   combine  -> output matches control exactly      INERT
//!   body     -> degenerate / empty, watchdog fires  CORRUPTS
//! ```
//! Four separate causes were found and fixed, and the corruption SURVIVED all
//! of them:
//!   - `hc_streams` is FP32 (`m*hc_mult*h*4`); a BF16-sized save/restore covered
//!     only half the highway. Fixed — the buffer diff now shows hc_streams
//!     unchanged across the body, so that restore is proven correct.
//!   - `head_scratch` was under-allocated by `hc_lowrank*4` = 1280 B, because
//!     `hc_head_lowrank`'s split layout is `t*(hc*h + rank)*4`, not `hc*h*4`.
//!     A real heap overflow. Fixed.
//!   - a local `MTP_META_OFFSET = 40960` sat inside the target's metadata block
//!     table region; now uses the shared 49152. Latent at long context. Fixed.
//!   - `hc_post` / `hc_lowrank_scratch` / `norm_output` (every buffer the diff
//!     named) are now saved and restored around the body. Fixed.
//!
//! ★ CONCLUSION: the corruption is NOT in `ctx.buffers`. Enumerating buffers to
//! save is the wrong strategy — the draft body inherits the target's whole
//! `ForwardContext` via `..*ctx` and shares everything reachable through it,
//! including state that is not a buffer this file can name. The sound fix is
//! ISOLATION BY CONSTRUCTION: give the draft its OWN `BufferArena` (sized for
//! T=1, a few MB) and build its `ForwardContext` from that, so it cannot reach
//! the target's state at all. Do that before spending another round here.
//!
//! **2. The combiner is dtype-mismatched.** Same root discovery: the highway is
//! FP32, but `rms_norm_strided` / `dense_gemv` / `residual_add` are BF16 ops, so
//! this reads the FP32 streams as BF16 and writes BF16 back into an FP32 buffer.
//! The draft is therefore numerically meaningless and the 0% accept rate
//! measured so far says NOTHING about the combiner reading. Needs FP32-aware
//! ops (no BF16->FP32 convert primitive exists today).
//!
//! **3. The combiner READING is genuinely open — two candidates.** The `[10240]`
//! grouped-norm width argues for the per-stream form implemented here. But the
//! loader's own note says the body runs "MIDDLE mHC mixing only — the proposer
//! owns BOTH ENDS", i.e. the proposer is expected to supply an `hc_expand` at
//! the front, which this form has no place for; that argues for a collapse-then-
//! `hc_expand` form, which would equally explain the square `fc_hidden`. Decide
//! it by implementing both behind `ATLAS_QWEN4EXP_MTP_COMBINER` and comparing
//! shadow accept rates — but only AFTER (1) and (2), or the comparison is noise.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;
use crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule;
use crate::weight_map::DenseWeight;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

// Use the SHARED constant (49152), not a local one. An earlier local 40960 sat
// INSIDE the region the target's metadata slab reserves for its block table
// (32768 + 256 + 4 bytes/block ⇒ 49152 covers 4096 blocks = 65536 tokens). It
// happens not to collide at short contexts — the target only fills as many
// entries as it has blocks — which is exactly what makes it the kind of latent
// bug that shows up first at long context. mtp_meta.rs says the constant "wanted
// to be shared rather than mirrored"; this is the third caller.
use crate::layers::mtp_meta::MTP_META_OFFSET;

/// Bytes of `hc_post` (and `hc_comb`) saved around the draft body. Both are
/// `m * hc_mult * {1,hc_mult} * 4`; the draft is a single row, so a small fixed
/// slab covers the rows it can touch.
const HC_POST_SAVE_BYTES: usize = 256;

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
    /// Save slots for the OTHER shared buffers the draft body writes. The
    /// buffer diff (`ATLAS_QWEN4EXP_MTP_DIFF=1`) named exactly these three, and
    /// reasoning that they "look transient" is what kept the corruption alive
    /// through several rounds — so they are saved and restored like the highway
    /// rather than argued about.
    hc_post_save: DevicePtr,
    hc_lowrank_save: DevicePtr,
    norm_output_save: DevicePtr,
    /// `[vocab]` BF16. The shadow step reuses the TARGET's `lm_head` (so the
    /// draft's logits go through the same quantization ladder the real token
    /// did), and that writes `buffers.logits()` — the buffer the scheduler is
    /// about to sample from. The target's logits are parked here first and put
    /// back afterwards, so the draft cannot change what the model emits.
    logits_stash: DevicePtr,
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

/// How far the shadow step runs — a BISECTION handle, not a feature.
///
/// Shadow mode was observed to change the target's own output (a thinking-loop
/// degeneration, watchdog-forced `</think>`), which means something in the draft
/// step mutates state the target still needs. Rather than guess which buffer,
/// this walks the step forward one stage at a time and the operator watches for
/// the first stage whose output stops matching the shadow-off control.
///
/// `ATLAS_QWEN4EXP_MTP_SHADOW_STAGE` = observe | combine | body | full (default).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShadowStage {
    /// argmax the target's logits and count only. Touches NOTHING else.
    Observe,
    /// + the combiner (writes `hc_streams`, then restores it).
    Combine,
    /// + the draft body forward.
    Body,
    /// + the draft's own final norm / LM head / argmax.
    Full,
}

pub fn shadow_stage() -> ShadowStage {
    static S: std::sync::OnceLock<ShadowStage> = std::sync::OnceLock::new();
    *S.get_or_init(
        || match std::env::var("ATLAS_QWEN4EXP_MTP_SHADOW_STAGE").as_deref() {
            Ok("observe") => ShadowStage::Observe,
            Ok("combine") => ShadowStage::Combine,
            Ok("body") => ShadowStage::Body,
            _ => ShadowStage::Full,
        },
    )
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
        // FP32 highway (4 B/elem) — see the note in `draft_hidden`.
        let streams_bytes = hc * h * 4;

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
                // `hc_head_lowrank`'s decode-split layout is
                // `t * (hc_mult*h + hc_lowrank) * 4` — normed FP32 [t, hc*H]
                // THEN low FP32 [t, rank]. Sizing this `hc*h*4` (the streams
                // alone) under-allocates by `rank*4` = 1280 B and the collapse
                // writes past the end of the allocation. Measured, not guessed.
                head_scratch: gpu.alloc((hc * h + config.hc_lowrank.max(1)) * 4)?,
                hc_post_save: gpu.alloc(HC_POST_SAVE_BYTES)?,
                hc_lowrank_save: gpu.alloc((hc * h + config.hc_lowrank.max(1)) * 4)?,
                norm_output_save: gpu.alloc(row)?,
                logits_stash: gpu.alloc(config.vocab_size * 2)?,
            },
            // Atlas's offset-from-1 rms_norm, NOT V4's `rms_norm_vanilla`:
            // this checkpoint's norm weights are offset-from-1 like the rest of
            // the qwen4_exp tree.
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_strided_k: gpu.kernel("norm", "rms_norm_strided")?,
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

    /// Park the target's logits so the draft's `lm_head` cannot change what the
    /// model emits. MUST be paired with [`Self::restore_logits`].
    pub fn stash_logits(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        gpu.copy_d2d_async(logits, self.buf.logits_stash, vocab * 2, stream)
    }

    /// Put the target's logits back after the draft has used the buffer.
    pub fn restore_logits(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        gpu.copy_d2d_async(self.buf.logits_stash, logits, vocab * 2, stream)
    }

    /// Record a shadow observation and return the running accept rate.
    pub fn shadow_observe(&self, drafted: Option<u32>, actual: u32) {
        if let Some(d) = drafted {
            self.shadow_drafts.fetch_add(1, Ordering::Relaxed);
            if d == actual {
                self.shadow_hits.fetch_add(1, Ordering::Relaxed);
            }
            let n = self.shadow_drafts.load(Ordering::Relaxed);
            if n.is_multiple_of(16) {
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

        // ★ THE BODY'S HIGHWAY IS `ctx.buffers.hc_streams()`, NOT a buffer we
        // can hand it. `decode_inner_hc` reads and writes the multi-stream state
        // through that buffer directly — the `hidden` argument is only the
        // single-stream scratch hc_pre collapses into. Two consequences, both
        // measured the hard way (0/240 accept AND a corrupted, truncated target
        // response when this was got wrong):
        //
        //   1. the combiner's output must be written INTO `hc_streams`, or the
        //      body silently runs on the target's highway and the draft is
        //      meaningless;
        //   2. that highway is PERSISTENT per-sequence state, so it must be
        //      saved and put back, or the draft destroys the target's next step.
        //
        // `buf.streams` is the save slot; the combiner reads from it and writes
        // into `hc_streams`.
        //
        // ★★ THE HIGHWAY IS FP32, NOT BF16. `BufferSizes` allocates it as
        // `m * hc_mult * h * 4` with the comment "the residual streams grow
        // large across the blocks ... so BF16 storage swamps the small
        // per-layer signal at scale and collapses generation".
        //
        // Sizing this as BF16 was a REAL corruption: the save/restore covered
        // only the first HALF of the highway, so the body's full-width FP32
        // writes survived the restore and poisoned every later target step
        // (measured: a thinking-loop degeneration, watchdog-forced `</think>`,
        // truncated answer). The bisect is the proof — the `combine` stage was
        // clean precisely BECAUSE it only ever touched, and then restored, that
        // same first half.
        let hc_bytes = hc as usize * h as usize * 4;

        // ── DIAGNOSTIC (ATLAS_QWEN4EXP_MTP_DIFF=1) ──
        // The bisect proved the BODY forward dirties state the target still
        // needs, but not WHICH buffer. Rather than keep guessing, fingerprint
        // the shared buffers either side of the call and name the ones that
        // changed. Taken BEFORE the combiner runs, so the baseline is the
        // TARGET's state — an earlier version sampled it after the combiner had
        // already written hc_streams, which made hc_streams a false positive.
        let diff = std::env::var("ATLAS_QWEN4EXP_MTP_DIFF").as_deref() == Ok("1");
        let probes: Vec<(&str, DevicePtr, usize)> = if diff {
            ctx.gpu.synchronize(stream).ok();
            vec![
                ("hc_streams", ctx.buffers.hc_streams(), hc_bytes.min(4096)),
                ("hc_post", ctx.buffers.hc_post(), 256),
                ("hc_comb", ctx.buffers.hc_comb(), 256),
                ("hc_lowrank_scratch", ctx.buffers.hc_lowrank_scratch(), 4096),
                ("hidden_states", ctx.buffers.hidden_states(), row),
                ("residual", ctx.buffers.residual(), row),
                ("norm_output", ctx.buffers.norm_output(), row),
                (
                    "scratch@target_meta",
                    ctx.buffers.scratch().offset(32768),
                    4096,
                ),
            ]
        } else {
            Vec::new()
        };
        let before: Vec<u64> = probes
            .iter()
            .map(|(_, p, n)| crate::speculative::hidden_fingerprint(ctx.gpu, *p, *n / 2))
            .collect();

        let body_streams = ctx.buffers.hc_streams();
        ctx.gpu
            .copy_d2d_async(target_streams, self.buf.streams, hc_bytes, stream)?;

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
            //
            // Destination is `body_streams` (= ctx.buffers.hc_streams()), which
            // is where the body will look. `buf.streams` must stay untouched —
            // it is the target's saved highway, restored below.
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                self.buf.normed_streams.offset(off),
                &self.module.fc_hidden,
                body_streams.offset(off),
                h,
                h,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                body_streams.offset(off),
                self.buf.embed_proj,
                h,
                stream,
            )?;
        }

        // BISECTION: stop after the combiner, restoring the highway. If the
        // target's output is clean here but dirty at `body`, the body forward
        // owns the corruption.
        if shadow_stage() < ShadowStage::Body {
            ctx.gpu
                .copy_d2d_async(self.buf.streams, body_streams, hc_bytes, stream)?;
            return Ok(());
        }

        // ★ SAVE the other shared buffers the body writes. Named by the buffer
        // diff, not by reasoning: hc_post, hc_lowrank_scratch, norm_output.
        let lr_bytes = (hc as usize * h as usize + ctx.config.hc_lowrank.max(1)) * 4;
        let saves: [(DevicePtr, DevicePtr, usize); 3] = [
            (
                ctx.buffers.hc_post(),
                self.buf.hc_post_save,
                HC_POST_SAVE_BYTES,
            ),
            (
                ctx.buffers.hc_lowrank_scratch(),
                self.buf.hc_lowrank_save,
                lr_bytes,
            ),
            (ctx.buffers.norm_output(), self.buf.norm_output_save, row),
        ];
        for (src, dst, n) in saves {
            ctx.gpu.copy_d2d_async(src, dst, n, stream)?;
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
        // Result captured, NOT `?`: an early return here would leave the
        // draft's streams in the target's persistent highway.
        let body_res = self.module.body.decode(
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
        );
        drop(kv_cache);
        if body_res.is_err() {
            ctx.gpu
                .copy_d2d_async(self.buf.streams, body_streams, hc_bytes, stream)?;
            for (dst, src, n) in saves {
                ctx.gpu.copy_d2d_async(src, dst, n, stream)?;
            }
            body_res?;
        }

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
        // Collapse the DRAFT's highway (which the body just wrote through
        // `body_streams`), not the saved target one.
        let collapse = ops::hc_head_lowrank(
            ctx.gpu,
            self.hc_head_k,
            body_streams,
            lowrank,
            h_out,
            self.buf.head_scratch,
            1,
            h,
            hc,
            eps,
            stream,
        );

        // ★ RESTORE THE TARGET'S HIGHWAY, unconditionally and before returning
        // any error. `hc_streams` is persistent per-sequence state: leaving the
        // draft's streams in it corrupts every subsequent target step, which is
        // exactly how this bug first showed up (a truncated, rewritten answer).
        ctx.gpu
            .copy_d2d_async(self.buf.streams, body_streams, hc_bytes, stream)?;
        // …and the other three, same contract: restore before any return.
        for (dst, src, n) in saves {
            ctx.gpu.copy_d2d_async(src, dst, n, stream)?;
        }

        if diff {
            ctx.gpu.synchronize(stream).ok();
            let changed: Vec<&str> = probes
                .iter()
                .zip(before.iter())
                .filter(|((_, p, n), b)| {
                    crate::speculative::hidden_fingerprint(ctx.gpu, *p, *n / 2) != **b
                })
                .map(|((name, _, _), _)| *name)
                .collect();
            tracing::info!(
                "qwen4_exp MTP diff: buffers changed across the draft body = {:?} \
                 (hc_streams should NOT appear — it is restored above; anything \
                 else that appears is shared state the target still needs)",
                changed
            );
        }
        collapse?;

        state.seq_len += 1;
        Ok(())
    }
}
