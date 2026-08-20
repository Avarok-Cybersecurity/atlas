// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative decoding abstraction (SDD).
//!
//! Defines the [`DraftProposer`] trait for speculative decoding strategies.
//! MTP implements this first; EAGLE-3 can implement later without engine changes.

pub mod ladder;
pub mod tree_shape;
pub mod verify_key;

pub use ladder::{mtp_ladder_disabled, mtp_ladder_drafts, mtp_max_seqs};

use std::any::Any;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;

/// Per-sequence state owned by a [`DraftProposer`].
///
/// Stores KV cache, hidden states, or whatever the proposer needs
/// across decode steps. Follows the same downcasting pattern as `LayerState`.
pub trait ProposerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A draft token proposer for speculative decoding.
///
/// The engine calls `propose()` after each target decode to get draft tokens,
/// then verifies them with the target model. `after_verify()` lets the
/// proposer trim state (e.g., KV cache) based on how many drafts were accepted.
/// Confidence floor for submitting drafts to verification
/// (`ATLAS_MTP_DRAFT_CONF`, default 0.0 = disabled). When the drafter's
/// chain confidence (min top-1 softmax prob across the drafts of one
/// propose) is below this, the drafts are discarded and the next step
/// decodes serially — skipping a verify that would most likely reject.
/// Economics at K=1 on the 35B MoE: verify ≈ 35 ms for 1+acc tokens vs
/// decode+propose ≈ 21 ms for 1, so a draft is only worth verifying when
/// p(accept) ≳ 0.66 — the threshold to calibrate around. Staged OFF until
/// its measured A/B (same discipline as ATLAS_SNAP_EVICT_ALPHA).
pub fn draft_conf_tau() -> f32 {
    std::env::var("ATLAS_MTP_DRAFT_CONF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|t| t.clamp(0.0, 0.99))
        .unwrap_or(0.0)
}

/// Shadow top-k draft instrumentation (`ATLAS_MTP_SHADOW_TOPK=k`, default
/// 0 = off, clamp k ≤ 8). Observational only — token selection untouched.
/// Each drafter `forward_one` D2H's its logits (same ~200 µs the conf path
/// pays) and logs the top-k candidate ids + softmax probs per position;
/// the verify steps log the target argmaxes under the same gate. Joining
/// the two offline yields the per-depth conditional top-k coverage that
/// gates the tree-speculation build (Phase 0 of the tree-spec plan).
/// Value-parsed, not presence-checked (`=0` really is off).
///
/// The SSOT parse. Both `ModelLevers::shadow_topk` (spark-model) and
/// `SchedLevers::shadow_topk` (spark-server) resolve through it once per run
/// rather than caching the answer in a `OnceLock` that a swap would pin.
pub fn shadow_topk() -> usize {
    std::env::var("ATLAS_MTP_SHADOW_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(8)
}

/// True when the MTP cap is raised above one sequence. The catchup ring
/// (`types.rs` `mtp_catchup_ring` + meta), the refeed label convention, and
/// the carry slot (`mtp_carry.rs`) are SINGLE-SEQUENCE structures — one ring,
/// one label range, one slot. Running them with n concurrently-verifying
/// sequences interleaves unrelated hiddens under one label space and breaks
/// `after_verify`'s env-keyed trim contract (`mtp_rows_to_trim`). Disabling
/// them process-wide when the cap > 1 is the only consistent option; a
/// slot-keyed ring is recorded follow-up work (acceptance debt at n>1).
pub fn mtp_multi_seq_mode() -> bool {
    mtp_max_seqs() > 1
}

/// `ATLAS_MTP_ACCEPT_DEBUG` (PRESENCE): per-BATCH-WIDTH acceptance telemetry.
///
/// The shipped K ladder gives `k_drafts == 2` at n in [5, 8], and the existing
/// positional counters (`k4_record_positional`) are gated on `k_drafts == 3`,
/// so at the C=8 operating point NOTHING reported p1 — only the na histogram.
/// This gate turns on a per-n line reporting p1, mean accepted and the derived
/// tokens/step, which is the quantity the C=8 arithmetic is written in.
/// Counters only (no D2H, no sync), but it logs per period, so keep it off in
/// timed legs unless the leg IS the accept measurement.
pub fn mtp_accept_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MTP_ACCEPT_DEBUG").is_ok())
}

/// Drafter catch-up feed on serial->speculative transitions
/// (`ATLAS_MTP_CATCHUP=1`, staged off). During serial-decode stretches the
/// scheduler rings the per-step final hiddens; on the next propose the gap
/// rows are batch-fed into the drafter KV so it never runs stale. Wrong
/// feeds cannot corrupt output (verification rejects bad drafts) — the
/// stake is acceptance only, which is the flip gate's metric.
/// Force-off in multi-seq MTP mode (single-sequence ring; see
/// [`mtp_multi_seq_mode`]).
pub fn mtp_catchup_enabled() -> bool {
    std::env::var("ATLAS_MTP_CATCHUP").ok().as_deref() == Some("1") && !mtp_multi_seq_mode()
}

/// Re-feed ACCEPTED draft rows with the target's TRUE hidden state
/// (`ATLAS_MTP_REFEED_ACCEPTED=1`, default OFF). Requires `ATLAS_MTP_CATCHUP=1`.
///
/// WHY. The MTP head is one module run autoregressively. Draft 1 consumes the
/// TARGET's verified hidden (`mtp_hidden_save`); every later draft consumes the
/// drafter's OWN single-block residual (`mtp_head.rs`, `current_hidden =
/// ctx.buffers.hidden_states()`). The drafter KV row written for draft d >= 2
/// therefore pairs the right token with the WRONG hidden — and on ACCEPT that
/// row is kept forever: `after_verify` trims only REJECTED rows. So every
/// accepted draft permanently contaminates the drafter's own context.
///
/// Measured on dgx2 (W4A4 27B, gate disarmed, seq_len ~10k, n=700/config):
/// unconditional per-position acceptance 0.660 -> 0.485 -> 0.407, i.e. the
/// FIRST autoregressive step costs x0.735 while the second costs only x0.838 —
/// the loss is concentrated exactly at the hidden-state handoff. Neither
/// existing lever touches it: `ATLAS_MTP_CATCHUP=1` alone is bit-identical
/// (its ring is only written on SERIAL decode steps, and with the throughput
/// gate disarmed there are none), and dropping `ATLAS_MTP_DRAFTER_PREFILL`
/// costs only 0.017/0.030 (~1 sd).
///
/// WHAT THIS DOES. After a verify, the target's true hidden for every accepted
/// position is sitting in the verify hidden buffer. Ring those hiddens under
/// the same label convention the serial path uses, and have `after_verify`
/// additionally drop the `num_accepted - 1` accepted rows that were written
/// with a drafter hidden. The next propose's catch-up feed then rebuilds
/// exactly those rows from the ring, with the TARGET's hidden, through the
/// already-exercised `catchup_drafter` batch path. No new kernel, no new
/// state machine — it reuses the gap-fill machinery for a gap that was never
/// being detected.
///
/// SAFETY. A wrong feed cannot corrupt output: verification rejects bad
/// drafts. The stake is acceptance only.
///
/// ## STATUS 2026-07-21 (SUPERSEDES the earlier "refuted" note). STAGED OFF.
///
/// The earlier note claimed the pair-key -> hidden mapping was wrong, inferred
/// from a sign reversal between a 67%-delivery and a 99%-delivery arm at
/// n=700 (+0.021 -> −0.023 on p2_uncond). **That inference is withdrawn.** The
/// two arms differed by only ~1.7 sd, neither was more than 1 sd from the
/// baseline, and they are not paired samples (each arm emits different text).
///
/// The mapping has since been VERIFIED DIRECTLY, with dumped hidden
/// fingerprints (`ATLAS_MTP_REFEED_DEBUG=1`, FNV-1a over each BF16 row), on
/// dgx2 / W4A4 27B / nd=2 / gate disarmed:
///
/// | check | result |
/// |---|---|
/// | ring D2D landed (`fp_src == fp_dst`) | 658 / 658 |
/// | fed hidden == live ring content at that label | 422 / 422 |
/// | `label == key + 1` and `RoPE == key + 1` on every feed | always |
/// | `fp(ring[position]) == fp(mtp_hidden_save)` at each propose | 302 / 304 (the 2 are a run's first propose) |
/// | **feed(key k) == `mtp_hidden_save` at the propose whose position was k+1** | **93 / 93** |
///
/// The last row is the non-tautological one: it compares the hidden this
/// feature feeds for pair key `k` against the hidden the drafter's own
/// `forward_one` consumed as `target_hidden` when it wrote pair key `k` —
/// two different code paths, bit-identical on every checkable case. So the
/// convention "ring label n holds hidden_{n−1}, hence pair key k reads label
/// k+1" is confirmed against an independently-exercised consumer.
///
/// What the earlier session DID find is real and is now fixed: the exclusive
/// `0..num_accepted` bound left one label unwritten per step, collapsing the
/// ring's contiguous window (458 fed / 231 missed = 67%). The bound is now
/// `0..=num_accepted` on both K=3 and K=4 (K=4 matters because
/// `mtp_rows_to_trim`'s extra trim is K-agnostic — without a K=4 ring write,
/// nd=3 would drop accepted drafter rows with nothing rebuilding them).
///
/// ## POWERED A/B (2026-07-21, dgx2): the pre-registered threshold is MET.
///
/// nd=2, gate disarmed, 16 documents x 8 turns, ~10k verify steps per arm
/// (with `ATLAS_MTP_GATE_FORCE=1` the engine is bit-reproducible, so n rises
/// only with NEW CONTENT, never with repetitions).
///
/// | arm | n | p1 | p2_uncond | tokens/verify step |
/// |---|---|---|---|---|
/// | OFF | 10,400 | 0.6100 | 0.4182 | 1.882 |
/// | ON  | 10,100 | 0.6262 | **0.4452** | 1.926 |
/// | delta | | +0.016 (2.4 sd) | **+0.027 (3.9 sd)** | **+2.3%** |
///
/// Criterion, pre-registered before the run: `p2_uncond` up by ≥ 0.015 at
/// ≥ 3 sd. Met. At n=700 — the sample that produced the earlier "refuted"
/// verdict — this same effect is ~1.0 sd, i.e. invisible. That verdict was a
/// power problem, not a mapping problem.
///
/// Caveat kept deliberately: the arms emit different text, so the binomial sd
/// understates the true variance. Content is matched (identical documents and
/// questions in both arms) but this is one measurement, not a replication.
/// STAYS DEFAULT OFF pending the standard gates (C2 smoke, A 35B
/// webserver_ok, B/D ST-995).
///
/// ## SIZE IT AGAINST THE REAL PRIZE BEFORE SPENDING ANY MORE TIME HERE
///
/// This lever is small BY CONSTRUCTION: it repairs at most
/// `num_accepted − 1` drafter KV **history** rows per step, while the measured
/// p1->p2 cliff happens WITHIN a single `propose`, where a history repair
/// cannot act at all. Two larger effects were measured the same night:
///
/// 1. **The drafter's own INPUT hidden at draft position >= 2** (dgx1's
///    teacher-forced oracle probe, `ATLAS_MTP_ORACLE_P2`): feeding draft 2 the
///    TARGET's true hidden instead of the MTP head's own takes p2_cond
///    0.5265 -> 0.7196, McNemar z = +18.4, recovering 1.40x the p1−p2 gap.
///    "Exposure bias" is refuted — the drafter is not mis-calibrated, it is
///    fed the wrong vector. That is **+0.193**, about **7x** this flag's
///    +0.027.
/// 2. **Drafter context blindness on WARM turns** (dgx2): the drafter holds
///    only **142 KV rows at sequence position 10,098**, because
///    `try_mtp_prefill_capture` no-ops whenever a prefill starts at a
///    reused-prefix boundary and the drafter prompt-prefill is then skipped.
///    Prefilling it on every turn measured **+0.086 p1 / +0.101 p2_uncond /
///    +10.2% accepted tokens per verify step** at n ~ 10k per arm, of which a
///    de-confounding pair (drafter coverage held at zero, prefix caching the
///    only variable) attributes **+0.079 p1 / +0.089 p2_uncond — 92% / 88% —
///    to drafter coverage** and the small remainder to warm restore.
///
/// Both dwarf this flag, and (2) also changes what this flag is worth: a
/// drafter that can actually see the prompt is a different drafter. **Build
/// (2) first, then re-measure this.**
///
/// Force-off in multi-seq MTP mode: the refeed label space is single-sequence
/// (see [`mtp_multi_seq_mode`]).
pub fn mtp_refeed_accepted_enabled() -> bool {
    std::env::var("ATLAS_MTP_REFEED_ACCEPTED").ok().as_deref() == Some("1") && !mtp_multi_seq_mode()
}

/// Deliberate off-by-N perturbation of the re-feed's ring LABEL
/// (`ATLAS_MTP_REFEED_SHIFT`, default 0 = the derived mapping).
///
/// This is a MAPPING-VALIDATION hatch, not a tuning knob. The label
/// convention (`label n holds hidden_{n-1}`, so pair key `k` reads label
/// `k+1`) cannot be falsified by any self-consistent checksum: the only
/// independently-exercised consumer of the verify hidden buffer is
/// `save_hidden_for_mtp(num_accepted)`, which reads the SAME buffer at the
/// SAME offset formula as the re-feed's `t = num_accepted` write, and the
/// sequence positions strictly between two propose positions are never
/// observed by any other code path. So the mapping is tested BEHAVIOURALLY
/// instead: shift every re-fed label by ±1 and measure acceptance. A uniform
/// shift keeps the ring contiguous (delivery is unchanged) but hands pair key
/// `k` the hidden of position `k ± 1`. If the derived mapping is right,
/// `shift = 0` must be the maximum of the three arms; if `+1` or `−1` wins,
/// that arm IS the correct mapping. If all three are indistinguishable, the
/// drafter's KV-history rows do not carry enough signal for this lever to
/// work at all — which is itself the answer.
pub fn mtp_refeed_shift() -> isize {
    std::env::var("ATLAS_MTP_REFEED_SHIFT")
        .ok()
        .and_then(|v| v.parse::<isize>().ok())
        .unwrap_or(0)
        .clamp(-4, 4)
}

/// `ATLAS_MTP_REFEED_DEBUG=1`: fingerprint every hidden that enters and
/// leaves the catch-up ring, so the pair-key -> hidden mapping can be read
/// off the serve log instead of argued about. Costs a D2H of one hidden row
/// (`h * 2` bytes) plus a stream sync per event — NEVER enable it in a timed
/// leg. See `mtp_refeed_shift` for why the fingerprints alone cannot falsify
/// the mapping, and what they DO establish (the ring's slot arithmetic and
/// the pair-key bookkeeping round-trip).
pub fn mtp_refeed_debug() -> bool {
    std::env::var("ATLAS_MTP_REFEED_DEBUG").ok().as_deref() == Some("1")
}

/// FNV-1a over a BF16 GPU row, for `mtp_refeed_debug` fingerprints.
pub fn hidden_fingerprint(gpu: &dyn GpuBackend, p: DevicePtr, h: usize) -> u64 {
    let mut b = vec![0u8; h * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return 0;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in &b {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

pub trait DraftProposer: Send + Sync {
    /// Allocate per-sequence proposer state.
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>>;

    /// [`Self::alloc_state`] with the sequence's KNOWN token budget
    /// (`prompt_len + max_tokens`), so a proposer whose per-sequence state
    /// scales with context can size to what this request can actually reach
    /// instead of the global `--max-seq-len` ceiling. That distinction is what
    /// OOMs a high-concurrency long-context serve: the ceiling is per-sequence
    /// and paid n times, while a typical request needs a fraction of it.
    ///
    /// `usize::MAX` means "unknown, use the ceiling". Defaults to
    /// `alloc_state`, so proposers with fixed-size state need not implement it.
    fn alloc_state_for(
        &self,
        gpu: &dyn GpuBackend,
        budget_tokens: usize,
    ) -> Result<Box<dyn ProposerState>> {
        let _ = budget_tokens;
        self.alloc_state(gpu)
    }

    /// The proposer's trained block size γ, when it is a block-diffusion
    /// drafter (DFlash/DFlash2). The serve layer derives num_drafts from
    /// this — the head resolved it from the drafter checkpoint and is the
    /// SSOT. `None` = not a block drafter.
    fn block_gamma(&self) -> Option<usize> {
        None
    }

    /// Chain confidence of the most recent `propose` (min top-1 softmax prob
    /// across its drafts), when the proposer computes it (`draft_conf_tau` >
    /// 0). `None` = not computed; callers must not gate on it then.
    fn last_confidence(&self) -> Option<f32> {
        None
    }

    /// Current drafter KV length (rows), for the catch-up append point.
    /// 0 = unknown / not applicable (catch-up is skipped).
    fn drafter_rows(&self, _state: &mut dyn ProposerState) -> usize {
        0
    }

    /// Sequence-space pair key of the newest drafter row (`None` = untracked;
    /// catch-up is skipped). The drafter row space is compacted, so `rows`
    /// cannot locate the drafter in the sequence — this can.
    fn last_pair_key(&self, _state: &mut dyn ProposerState) -> Option<usize> {
        None
    }

    /// ATLAS_MTP_CARRY_DRAFTER: move this sequence's drafter KV blocks OUT of
    /// its proposer state, so `free_state` releases nothing and the model can
    /// hold them for the next turn. Returns `(blocks, rows, last_pair_key)`;
    /// `None` = unsupported or nothing to carry. After this call the state
    /// must behave as if freshly allocated.
    fn take_drafter_kv(
        &self,
        _state: &mut dyn ProposerState,
    ) -> Option<(Vec<u32>, usize, Option<usize>)> {
        None
    }

    /// Inverse of [`Self::take_drafter_kv`]: install carried blocks into a fresh
    /// proposer state. Returns false when unsupported (caller must then free
    /// the blocks itself).
    fn install_drafter_kv(
        &self,
        _state: &mut dyn ProposerState,
        _blocks: Vec<u32>,
        _rows: usize,
        _last_pair_key: Option<usize>,
    ) -> bool {
        false
    }

    /// Release drafter KV blocks that no proposer state owns (a carried entry
    /// being replaced or dropped).
    fn free_drafter_kv(&self, _blocks: &[u32]) {}

    /// Append drafter rows at KV slots `row_base ..` with RoPE positions
    /// `pos_base ..` from `(tokens, hiddens)` pairs — the catch-up feed.
    /// Returns rows written (0 = unsupported/no-op).
    #[allow(clippy::too_many_arguments)]
    fn catchup_drafter(
        &self,
        _tokens: &[u32],
        _hiddens: DevicePtr,
        _row_base: usize,
        _pos_base: usize,
        _state: &mut dyn ProposerState,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<usize> {
        Ok(0)
    }

    /// Propose up to `num_drafts` tokens autoregressively.
    ///
    /// # Arguments
    /// * `last_token` - The last verified token (target model output)
    /// * `target_hidden` - Target model's hidden states after final norm [1, hidden_size] BF16
    /// * `position` - Current sequence position (for RoPE)
    /// * `num_drafts` - Maximum number of draft tokens to produce
    /// * `state` - Per-sequence proposer state
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    /// * `grammar_bitmask` - Optional XGrammar bitmask (ceil(vocab_size/32) i32
    ///   words). When `Some`, drafts are constrained to tokens the grammar
    ///   accepts at the current matcher position; bit `tok` set ⇒ allowed.
    ///   `None` preserves the unconstrained fast path.
    /// * `target_hidden_stack` - Optional pointer to a contiguous buffer of
    ///   `5 × target_hidden × bf16` containing the most-recently-decoded
    ///   token's hidden states captured at the drafter's `target_layer_ids`
    ///   (DFlash uses this; MTP ignores). Layout matches vLLM's
    ///   `combine_hidden_states` input: shallow-to-deep concatenation along
    ///   the feature axis.
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>>;

    /// Batched cross-sequence propose: draft `num_drafts` tokens for each of
    /// `n = last_tokens.len()` sequences, reading every drafter weight ONCE
    /// per draft position instead of once per sequence (the measured C=4
    /// serialization: 12 x ~5 ms per-seq drafter forwards per batched verify
    /// step, ~62 ms of the ~180 ms step).
    ///
    /// Row i of every slice belongs to sequence i; `target_hiddens[i]` is
    /// that sequence's accepted-position hidden ([1, hidden] BF16, may be
    /// non-contiguous across i). Chains autoregressively per sequence like
    /// `propose` — position j uses (draft_{j-1}, drafter's own hidden row i).
    ///
    /// Returns `Ok(None)` when unsupported (caller falls back to the per-seq
    /// `propose` loop); `Ok(Some(drafts))` with `drafts[i].len() ==
    /// num_drafts` on success. Grammar-constrained sequences must not reach
    /// this path (callers gate on grammarless).
    #[allow(clippy::too_many_arguments)]
    fn propose_batch(
        &self,
        _last_tokens: &[u32],
        _target_hiddens: &[DevicePtr],
        _positions: &[usize],
        _num_drafts: usize,
        _states: &mut [&mut dyn ProposerState],
        _ctx: &ForwardContext,
        _stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        Ok(None)
    }

    /// The widest batch [`Self::propose_batch`] can carry in ONE drafter
    /// forward per draft position, derived from this proposer's resolved
    /// kernels and the arena's row capacities. `1` = per-sequence only.
    ///
    /// Callers chunk by this instead of a hardcoded constant: a fixed cap of
    /// 4 made a 16-sequence step run 4 drafter forwards per position, each
    /// re-reading the whole drafter — the batched-propose lever's own cost
    /// re-introduced by its caller.
    fn propose_batch_max(&self, _buffers: &BufferArena, _config: &ModelConfig) -> usize {
        1
    }

    /// Prefill the drafter's own context (KV cache) over the prompt, before
    /// the first `propose()` of a sequence (ATLAS_MTP_DRAFTER_PREFILL).
    ///
    /// * `prompt_tokens` — the prompt token ids `t_0..t_{P-1}`.
    /// * `hiddens` — device buffer `[P, hidden_size]` BF16; row `i` is the
    ///   target's final-layer (pre-final-norm) hidden after processing `t_i`.
    ///
    /// Returns the number of drafter positions written (0 = unsupported /
    /// already prefilled / nothing to do). Default: no-op.
    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let _ = (prompt_tokens, hiddens, state, ctx, stream);
        Ok(0)
    }

    /// Read the draft token ID stored on GPU by the last `propose()` call
    /// that used `draft_embed_target = Some(...)`. Returns 0 if not supported.
    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let _ = gpu;
        Ok(0)
    }

    /// Called after target verification to trim proposer state.
    ///
    /// `num_accepted` indicates how many draft tokens were accepted.
    /// The proposer should trim its KV cache / state to match.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()>;

    /// Free per-sequence proposer state (KV cache blocks, device buffers, etc.).
    ///
    /// Must be called when a sequence is finished to avoid resource leaks.
    /// `gpu` is threaded in (symmetric with `alloc_state`) so implementations
    /// can release raw device allocations stored on the state — `DevicePtr`
    /// has no `Drop`, so anything `alloc_state` allocated leaks unless it is
    /// explicitly freed here.
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let _ = (gpu, state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProposerState {
        tokens_proposed: Vec<u32>,
    }

    impl ProposerState for MockProposerState {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![42, 99],
        });
        let mock = state.as_any().downcast_ref::<MockProposerState>().unwrap();
        assert_eq!(mock.tokens_proposed, vec![42, 99]);
    }

    #[test]
    fn test_proposer_state_downcast_mut() {
        let mut state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![],
        });
        let mock = state
            .as_any_mut()
            .downcast_mut::<MockProposerState>()
            .unwrap();
        mock.tokens_proposed.push(7);
        assert_eq!(mock.tokens_proposed, vec![7]);
    }
}
