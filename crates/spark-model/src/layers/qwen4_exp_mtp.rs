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
//! # STATUS: the draft head WORKS — 86.5% shadow accept, harness inert.
//!
//! Measured (250-token greedy completion, `ATLAS_QWEN4EXP_MTP_SHADOW=1`):
//! ```text
//!   shadow off -> full answer, finish: stop
//!   shadow on  -> BYTE-IDENTICAL output, finish: stop, 0 failures
//!   accept      -> 83/96 drafts matched the target's next token = 86.5%
//! ```
//! ★ That accept rate VALIDATES the per-stream combiner reading empirically.
//! The alternative (collapse-then-`hc_expand`) is not needed and was removed.
//!
//! What it took, in the order the bugs were found — every one invisible to
//! review, each caught only by running it:
//!   1. kernel module is `norm::rms_norm`, not `rms_norm::rms_norm` (the engine
//!      refused to start rather than serve on a null handle).
//!   2. the hook must live in `decode_a.rs`: `decode_batch_dispatch` returns
//!      early for n == 1, so a hook on the batched path never fires here.
//!   3. the body's highway is `ctx.buffers.hc_streams()`, not a buffer you can
//!      hand it, and it is PERSISTENT per-sequence state.
//!   4. that highway is FP32 (`m*hc_mult*h*4`), not BF16.
//!   5. `head_scratch` needs `t*(hc*h + hc_lowrank)*4`; sizing it `hc*h*4` is a
//!      1280 B heap overflow.
//!   6. `MTP_META_OFFSET` must be the SHARED 49152.
//!   7. ★ the decisive one: buffer-by-buffer isolation NEVER converged. A full
//!      decoder layer touches state the caller cannot enumerate. Giving the
//!      draft its OWN `BufferArena` (T=1, <1 MB) fixed it at once and let every
//!      save/restore be deleted. ISOLATE STRUCTURALLY; DO NOT ENUMERATE.
//!   8. the combiner must be dtype-correct across the FP32/BF16 seam:
//!      `hc_pre_stage_bf16` (FP32 highway -> BF16 grouped norm, offset-from-1,
//!      per-stream RMS — the model's own kernel for exactly this) then BF16
//!      GEMVs then `qhc_mtp_combine_streams` (BF16 -> FP32 highway). Running
//!      BF16 ops directly over the FP32 highway is SILENT garbage, and it is
//!      what made every earlier accept measurement meaningless.
//!
//! # NOT YET SPECULATION
//!
//! `--speculative` still produces no speedup: this drafts and scores, nothing
//! is fed back. The verify path is the remaining work — `decode_batched` and
//! `decode_verify_multi` both refuse under the highway (they keep their own
//! residual, which the highway replaces), so K-row verification has to route
//! through `prefill_inner_hc`, and a rejected draft needs QSA/PLE rewind that
//! has no API today. 86.5% accept is what makes that work worth doing.

mod sampling;
mod state;

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

/// Per-sequence MTP draft state.
pub struct Qwen4ExpMtpState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub body_state: Box<dyn LayerState>,
    /// Shadow mode: the token drafted at the PREVIOUS decode step, awaiting
    /// comparison against the token the target actually emits this step.
    pub pending_draft: Option<u32>,
    /// How many drafts the last `propose` produced, so `after_verify` knows how
    /// many rows to unwind when some are rejected.
    pub last_num_drafted: usize,
    /// Rows the draft must unwind before its next round. Set by `after_verify`
    /// (which gets no GPU handle) and APPLIED at the top of the next `propose`,
    /// where `ctx.gpu` is available. Deferring is safe: nothing reads the draft
    /// body's state in between.
    pub pending_rewind: usize,
    /// The draft BODY's aux carry (its own QSA indexer state) snapshotted before
    /// this round's drafts. The draft body advances its carry exactly like the
    /// target does, and its ingest asserts `pos == ingested` — so a rejected
    /// draft must rewind it or the NEXT draft trips that assert.
    pub pre_draft_aux: Option<Vec<u8>>,
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
    /// `[hc_mult * hidden]` BF16 — `fc_hidden` applied per stream, before the
    /// combine tail writes the FP32 highway.
    per_stream: DevicePtr,
    /// `[hc_mult * hidden]` low-rank head scratch.
    head_scratch: DevicePtr,
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
    /// ★ THE DRAFT'S OWN BUFFER ARENA — isolation by CONSTRUCTION.
    ///
    /// The draft body used to inherit the target's `ForwardContext` (and so its
    /// arena) via `..*ctx`. Four rounds of "find the shared buffer, save and
    /// restore it" each fixed a real bug and none of them stopped the target's
    /// output being corrupted, because a full decoder layer touches state this
    /// file cannot enumerate. Sized for ONE token, so the draft physically
    /// cannot reach anything the target owns.
    arena: spark_runtime::buffers::BufferArena,
    buf: MtpBuffers,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    hc_head_k: KernelHandle,
    /// The target's NVFP4 vocab head, shared (Copy pointers). The draft writes
    /// its logits into the DRAFT arena, so it never touches the buffer the
    /// scheduler samples from — the stash/restore the shadow path needed is
    /// unnecessary once the arena is private.
    lm_head_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    /// The target's NATIVE EXL3 vocab head, BORROWED (`Arc`), not copied.
    ///
    /// Under `ATLAS_EXL3_NATIVE` the lm_head is served from packed trellis and
    /// there is no NVFP4 head at all — `build.rs` sets all three NVFP4/FP8
    /// head slots to `None` — so without this arm the draft errored on EVERY
    /// propose and speculation silently degenerated to serial.
    ///
    /// Borrowing is also the CORRECT reading of the checkpoint: the 4.05bpw
    /// tensor map ships exactly one `lm_head` trellis `[248320,2560]` K=6 and
    /// no `mtp.lm_head` — the MTP block SHARES the target's vocab head. A
    /// second materialized copy would be both wrong and ~325 MB wasted.
    ///
    /// The shared head carries the model-wide `Exl3LaunchState` (one locks
    /// buffer, one host mutex, one cross-stream fence), so the draft is a
    /// third caller of the SAME section — never a second launch state. It
    /// projects into the DRAFT's private arena through the head's reserved
    /// draft scratch row, so the private-arena isolation (PR #782) is intact.
    lm_head_exl3: Option<std::sync::Arc<crate::model::lm_head_exl3::Exl3LmHead>>,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_sw_k: KernelHandle,
    argmax_k: KernelHandle,
    /// FP32 highway -> BF16 grouped norm (the combiner's hidden branch).
    hc_stage_k: KernelHandle,
    /// BF16 per-stream + broadcast -> FP32 highway (the combiner's tail).
    combine_k: KernelHandle,
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
    // `pub(crate)`: the signature now names `Exl3LmHead`, which is a
    // crate-private type (the native head is an internal dispatch arm). The
    // only caller is `model/impl_b3_accessors.rs`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: Option<crate::weight_map::QuantizedWeight>,
        lm_head_exl3: Option<std::sync::Arc<crate::model::lm_head_exl3::Exl3LmHead>>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        max_sequences: usize,
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
        let num_blocks =
            state::draft_pool_blocks(max_seq_len, kv_config.block_size, max_sequences)?;
        let (bs, kvh, hd) = (
            kv_config.block_size,
            kv_config.num_kv_heads,
            kv_config.head_dim,
        );
        let kv_bytes = [num_blocks, bs, kvh, hd, 2, 2]
            .into_iter()
            .try_fold(1usize, |bytes, dim| bytes.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("Qwen MTP KV pool byte size overflow"))?;
        anyhow::ensure!(
            kv_bytes <= gpu.free_memory()?,
            "Qwen MTP private KV pool needs {kv_bytes} bytes for {max_sequences} sequence slots; insufficient free GPU memory"
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        tracing::info!(
            "qwen4_exp MTP head: private KV pool {} blocks x {} tok = {} tokens, \
             {} kv_heads x {} head_dim BF16 (~{:.2} GB), capacity {max_sequences} sequence slots ({kv_bytes} bytes). This is allocated AFTER \
             the main pool and is therefore OUTSIDE the util pledge.",
            num_blocks,
            bs,
            num_blocks * bs,
            kvh,
            hd,
            kv_bytes as f64 / 1e9,
        );

        // T=1 arena for the draft. `max_batch_tokens = 1` keeps every
        // token-scaled buffer at one row; `max_seq_len` still sizes the scratch
        // block-table region, and kv_block_size must match this head's own pool.
        let free_before = gpu.free_memory().unwrap_or(0);
        let arena = spark_runtime::buffers::BufferArena::new(config, 1, max_seq_len, 16, 1, gpu)?;
        let free_after = gpu.free_memory().unwrap_or(0);
        tracing::info!(
            "qwen4_exp MTP head: private T=1 buffer arena costs {:.3} GB. The draft \
             runs entirely inside it, so it cannot reach the target's buffers.",
            (free_before.saturating_sub(free_after)) as f64 / 1e9,
        );

        Ok(Self {
            module,
            embed_tokens,
            kv_cache: Mutex::new(kv_cache),
            arena,
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
                per_stream: gpu.alloc(hc * h * 2)?,
                head_scratch: gpu.alloc((hc * h + config.hc_lowrank.max(1)) * 4)?,
                logits_stash: gpu.alloc(config.vocab_size * 2)?,
            },
            // Atlas's offset-from-1 rms_norm, NOT V4's `rms_norm_vanilla`:
            // this checkpoint's norm weights are offset-from-1 like the rest of
            // the qwen4_exp tree.
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            lm_head_nvfp4,
            lm_head_exl3,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: KernelHandle(0),
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            hc_stage_k: gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?,
            combine_k: gpu.kernel("hyper_connection", "qhc_mtp_combine_streams")?,
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
            last_num_drafted: 0,
            pending_rewind: 0,
            pre_draft_aux: None,
        })
    }

    /// Snapshot the draft body's own carry before a round of drafts.
    pub fn snapshot_draft_aux(
        &self,
        st: &Qwen4ExpMtpState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.module
            .body
            .snapshot_aux(st.body_state.as_ref(), gpu, stream)
    }

    /// Rewind the draft body's own state by `rows` after a rejected draft.
    ///
    /// Mirrors the target-side rollback: the draft advanced its seq_len, its KV
    /// and its QSA carry for every row it produced, but only the accepted ones
    /// are real. The KV past `seq_len` is left alone — the next draft overwrites
    /// it — but the QSA carry must be restored, because its ingest asserts hard
    /// on `pos == ingested`.
    pub fn rewind_draft(
        &self,
        st: &mut Qwen4ExpMtpState,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        st.seq_len = st.seq_len.saturating_sub(rows);
        // MARK REWIND, not snapshot/restore. `snapshot_aux` returns None until
        // the sequence has reached the layer's ingest, so the snapshot pair
        // cannot undo the FIRST draft — it leaves the carry ahead of the
        // sequence and the next draft dies on
        // `QSA: decode at pos 0 but 1 tokens ingested`. Measured, not theorised.
        self.module
            .body
            .rewind_aux(st.body_state.as_mut(), rows, gpu, stream)?;
        st.pre_draft_aux = None;
        Ok(())
    }

    /// Scratch the draft writes its post-mHC-head hidden into. Lives in the
    /// draft's own arena, so it cannot collide with the target's.
    pub fn draft_h_out(&self) -> DevicePtr {
        self.arena.hidden_states()
    }

    /// The DRAFT's highway after a `draft_hidden` call. Chaining a second draft
    /// feeds this back in as the next step's input — draft j+1 continues from
    /// the body's own state, not the target's.
    pub fn draft_streams(&self) -> DevicePtr {
        self.arena.hc_streams()
    }

    /// Turn the draft's final hidden into a token id, entirely inside the
    /// draft's own arena.
    ///
    /// qwen4_exp sets `final_norm_identity`, so there is NO final norm here —
    /// the mHC head's own `hc_norm` plays that role. Applying one would be an
    /// uninvited extra RMS divide (a bug this model already shipped once).
    pub fn draft_token(&self, h_out: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<u32> {
        self.draft_token_with_grammar(h_out, ctx, stream, None)
    }

    /// Grammar state belongs to the scheduler. Only the first draft can use
    /// its current mask; later draft rows are checked by target verification.
    pub(super) fn draft_token_with_grammar(
        &self,
        h_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let vocab = ctx.config.vocab_size as u32;
        let h = ctx.config.hidden_size as u32;
        let logits = self.arena.logits();
        // NATIVE EXL3 FIRST. Under `ATLAS_EXL3_NATIVE` there is no NVFP4 head
        // to fall back to, and the borrowed trellis head is the SAME head the
        // target samples from — which is the whole point of scoring a draft.
        // `project_draft` writes ONE row into the DRAFT's own arena using the
        // head's reserved scratch row, inside a section of the model-shared
        // `Exl3LaunchState`.
        if let Some(exl3) = self.lm_head_exl3.as_ref() {
            exl3.project_draft(ctx.gpu, h_out, logits, stream)?;
        } else {
            let w = self.lm_head_nvfp4.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "qwen4_exp MTP: no NVFP4 and no native-EXL3 lm_head for the draft head"
                )
            })?;
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                false,
                h_out,
                w,
                logits,
                vocab,
                h,
                stream,
            )?;
        }
        if let Some(bitmask) = grammar_bitmask {
            return sampling::grammar_argmax(ctx.gpu, logits, vocab as usize, bitmask);
        }
        let out_ptr = self.arena.scratch();
        ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, vocab, stream)?;
        let mut b = [0u8; 4];
        ctx.gpu.copy_d2h(out_ptr, &mut b)?;
        Ok(u32::from_le_bytes(b))
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

        // The body reads its highway through the private arena. Snapshot the
        // input before the combiner writes that arena: chained drafts read
        // from the same arena, so input and output can alias. Highway elements
        // are FP32, even though collapsed hidden rows are BF16.
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

        // The draft's highway lives in the DRAFT's arena, not the target's.
        // Nothing below writes a buffer the target owns.
        let body_streams = self.arena.hc_streams();
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

        // ── 2. Hidden branch — DTYPE-CORRECT end to end ──
        // The highway is FP32; the projections are BF16 GEMVs. So:
        //   a) `hc_pre_stage_bf16` reads the FP32 streams and writes the GROUPED
        //      norm as BF16. It is the model's own kernel for exactly this
        //      (per-stream RMS, offset-from-1 scale, `[hc*H]` weight) — which is
        //      also independent confirmation that `pre_fc_norm_hidden [10240]`
        //      normalizes the four-stream highway.
        //   b) `fc_hidden` is applied PER STREAM as a BF16 GEMV.
        //   c) `qhc_mtp_combine_streams` writes the FP32 highway from those
        //      per-stream BF16 rows plus the broadcast embedding projection.
        // An earlier version ran BF16 ops directly over the FP32 buffer — silent
        // garbage, and the reason the first accept measurements were meaningless.
        ops::hc_pre_stage_bf16_norm(
            ctx.gpu,
            self.hc_stage_k,
            self.buf.streams,
            self.module.pre_fc_norm_hidden.weight,
            self.buf.normed_streams,
            1,
            h,
            hc,
            eps,
            stream,
        )?;
        for i in 0..hc as usize {
            let off = i * row;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                self.buf.normed_streams.offset(off),
                &self.module.fc_hidden,
                self.buf.per_stream.offset(off),
                h,
                h,
                stream,
            )?;
        }
        ops::qhc_mtp_combine_streams(
            ctx.gpu,
            self.combine_k,
            self.buf.per_stream,
            self.buf.embed_proj,
            body_streams,
            h,
            hc,
            stream,
        )?;

        // No save/restore of target buffers: the draft runs in its own arena.
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
            // This private arena contains one draft row, regardless of the
            // accepted row selected from the target's verification highway.
            hc_row_offset: 0,
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
            // ★ the draft's OWN arena — the whole point of this design.
            buffers: &self.arena,
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

        // Keep the private arena's output highway for the next autoregressive
        // draft. The input snapshot belongs to the combiner; restoring it here
        // would make every later draft consume the preceding draft's INPUT.

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
                 (target hc_streams should NOT appear — the draft arena is private; anything \
                 else that appears is shared state the target still needs)",
                changed
            );
        }
        collapse?;

        state.seq_len += 1;
        Ok(())
    }
}
