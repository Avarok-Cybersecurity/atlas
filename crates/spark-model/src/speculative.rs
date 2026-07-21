// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative decoding abstraction (SDD).
//!
//! Defines the [`DraftProposer`] trait for speculative decoding strategies.
//! MTP implements this first; EAGLE-3 can implement later without engine changes.

use std::any::Any;

use anyhow::Result;
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

/// Drafter catch-up feed on serial->speculative transitions
/// (`ATLAS_MTP_CATCHUP=1`, staged off). During serial-decode stretches the
/// scheduler rings the per-step final hiddens; on the next propose the gap
/// rows are batch-fed into the drafter KV so it never runs stale. Wrong
/// feeds cannot corrupt output (verification rejects bad drafts) — the
/// stake is acceptance only, which is the flip gate's metric.
pub fn mtp_catchup_enabled() -> bool {
    std::env::var("ATLAS_MTP_CATCHUP").ok().as_deref() == Some("1")
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
/// It stays default OFF pending its powered A/B (~10k verify steps per arm;
/// with `ATLAS_MTP_GATE_FORCE=1` the engine is bit-reproducible, so n rises
/// only with NEW CONTENT, never with repetitions).
///
/// CONTEXT FOR WHOEVER PICKS THIS UP: this lever is small by construction. It
/// repairs at most `num_accepted − 1` drafter KV rows per step, and the
/// measured p1->p2 cliff happens WITHIN a single `propose` — where a KV-history
/// repair cannot act at all. Measured on the same box the same night, the
/// drafter holds only **142 rows at sequence position 10,098** on a warm turn,
/// because `try_mtp_prefill_capture` no-ops whenever a prefill starts at a
/// reused-prefix boundary and the drafter prompt-prefill is then skipped. Cold
/// (prefilled) vs warm (blind) acceptance measured +0.043 p1 / +0.067
/// p2_uncond at n=10,400. Fix that first.
pub fn mtp_refeed_accepted_enabled() -> bool {
    std::env::var("ATLAS_MTP_REFEED_ACCEPTED").ok().as_deref() == Some("1")
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
