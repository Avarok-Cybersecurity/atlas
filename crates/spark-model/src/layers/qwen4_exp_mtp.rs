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
    /// Batched propose (see `draft_tokens_batched`): `[batch_cap, hidden]`
    /// BF16 draft hiddens, one row per sequence, so n bodies can run before
    /// one LM head scores them all.
    batch_h_out: DevicePtr,
    /// `[batch_cap, vocab]` BF16 draft logits, private to the draft.
    batch_logits: DevicePtr,
    /// `[batch_cap]` u32 argmax results — ONE D2H for the whole batch.
    batch_tok: DevicePtr,
}

pub struct Qwen4ExpMtpHead {
    module: Qwen4ExpMtpModule,
    embed_tokens: DenseWeight,
    /// ★ THE DRAFTER'S OWN CONFIG — full pre-shard head counts, tp_world_size 1.
    ///
    /// The drafter is REPLICATED across TP ranks: every rank runs the whole
    /// draft locally, which is why its `ForwardContext` carries `comm: None`.
    /// But that context is built with `..*ctx`, so it used to inherit the
    /// TARGET's config — and under TP=2 `topology.rs` has already halved
    /// `num_attention_heads` there. The drafter then computed 12-head attention
    /// over its own 24-head weights: no error, no crash, correct-looking text
    /// (rejected drafts still emit the target's token), and acceptance falling
    /// from p1 0.83 to 0.42 while propose kept costing full price.
    cfg: atlas_core::config::ModelConfig,
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
    /// Rows the private arena carries (= sequences one batched propose can
    /// run). `max_sequences` capped at `BATCH_CAP`.
    batch_cap: usize,
    /// Batched LM-head GEMV tiers (4..8 rows) for `draft_tokens_batched`.
    w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers,
    /// Batched argmax over `[n, vocab]` draft logits. 0-handle = unbatched only.
    argmax_batch_k: KernelHandle,
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

/// Sequences one batched propose can score in a single LM-head pass. Sized
/// for the C<=16 shapes this model serves; the scheduler chunks wider batches.
pub(crate) const BATCH_CAP: usize = 16;

impl Qwen4ExpMtpHead {
    pub fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Qwen4ExpMtpState> {
        Ok(Qwen4ExpMtpState {
            block_table: Vec::new(),
            seq_len: 0,
            body_state: self.module.body.alloc_state(gpu)?,
            pending_draft: None,
            last_num_drafted: 0,
            pending_rewind: 0,
        })
    }

    /// Diagnostic-only copy of the draft prefix for the old-snapshot A/B.
    /// Rejected drafts use `rewind_aux` marks and never consume this blob.
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

    /// Whether `draft_tokens_batched` can run: an NVFP4 head to batch over
    /// (the EXL3 trellis head is scored one row at a time by its own
    /// `project_draft` and is not batched here), the batchm GEMV family, and
    /// the batched argmax kernel.
    pub fn batch_ready(&self) -> bool {
        if self.argmax_batch_k.0 == 0 {
            return false;
        }
        // Native EXL3: the trellis head projects n rows through the SAME
        // `project` the one-row draft uses — it only needed the reserved
        // scratch rows to sit on (`EXL3_DRAFT_ROWS`). No NVFP4 head exists
        // under native EXL3, and manufacturing one would mean a second
        // 318 MB vocab copy scoring drafts through a different approximation
        // than the target samples from.
        if self.lm_head_exl3.is_some() {
            return true;
        }
        self.lm_head_nvfp4.is_some() && self.w4a16_batchm.has_base()
    }

    pub fn batch_cap(&self) -> usize {
        self.batch_cap
    }

    /// Staged draft-hidden row `i` — pass as `draft_hidden`'s `h_out`.
    pub fn batch_h_out_row(&self, i: usize, hidden: usize) -> DevicePtr {
        self.buf.batch_h_out.offset(i * hidden * 2)
    }

    /// Arena highway row `i` (FP32, `hc_mult * hidden` per row): the batched
    /// body's input for sequence i, and its output after the body ran.
    pub fn arena_streams_row(&self, i: usize, hc_mult: usize, hidden: usize) -> DevicePtr {
        self.arena.hc_streams().offset(i * hc_mult * hidden * 4)
    }
}

#[path = "qwen4_exp_mtp_hidden.rs"]
mod qwen4_exp_mtp_hidden;

#[path = "qwen4_exp_mtp_combine.rs"]
mod qwen4_exp_mtp_combine;

#[path = "qwen4_exp_mtp_draft_token.rs"]
mod qwen4_exp_mtp_draft_token;

#[path = "qwen4_exp_mtp_new.rs"]
mod qwen4_exp_mtp_new;
