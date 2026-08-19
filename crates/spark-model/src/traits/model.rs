// SPDX-License-Identifier: AGPL-3.0-only

//! `Model` trait — the interface the scheduler talks to.
//!
//! ## Dispatch contract
//!
//! Per request, the scheduler invokes:
//!
//! 1. [`Model::prefill`] (or [`Model::prefill_chunk`] for chunked prefill)
//!    once per sequence. Returns logits at the last prompt position;
//!    populates the sequence's KV cache and SSM state.
//! 2. [`Model::decode`] once per emitted token. Returns next-token logits;
//!    extends KV/SSM state by one position. May be replaced by
//!    [`Model::decode_batch`] when multiple sequences are co-scheduled.
//! 3. Optional speculative-decode verify path: [`Model::decode_verify_graphed`]
//!    (K=2), [`Model::decode_verify_graphed_k3`] (K=3),
//!    [`Model::decode_verify_graphed_k4`] (K=4), or
//!    [`Model::decode_verify_graphed_kgamma`] (DFlash γ-token).
//!    These take [last_token, draft0, ..] and return per-position logits;
//!    the scheduler picks accept/reject and rolls back state on reject.
//! 4. [`Model::mixed_forward`] fuses one decode step + one prefill chunk
//!    through a single weight load; used by the scheduler to amortize
//!    weight-streaming cost when both phases are pending.
//!
//! Implementors live under `crates/spark-model/src/model/trait_impl/`,
//! split per phase (prefill_a/b/c/d, decode_a/b, verify_a/b/c/d) per
//! ADR-0006's multi-file module idiom.
//!
//! ## Concurrency
//!
//! `Model: Send + Sync` — a single instance handles all sequences
//! concurrently. Per-sequence state lives in [`SequenceState`].

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::{MixedBatchResult, MixedForwardResult, PrefillSlice, SequenceState};

/// One beam-search request for a translation model (NLLB). Carries the resolved
/// per-request parameters the scheduler stamps onto the sequence; the model runs
/// the whole beam search to completion and returns the winning hypothesis.
#[derive(Debug, Clone)]
pub struct BeamReq {
    /// Raw source subword ids (the model adds `[src_lang] … </s>` itself).
    pub prompt_tokens: Vec<u32>,
    /// Per-request source/target language token ids (`0` = deployment default).
    pub src_lang_id: u32,
    pub tgt_lang_id: u32,
    /// Per-request LoRA slot (`>=0` apply, `-1` base).
    pub adapter_slot: i32,
    pub num_beams: usize,
    pub max_new: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

/// The multi-sequence batch padding ladder — the SSOT for `padded_n`.
///
/// Batched decode pads the live sequence count up to a small set of captured
/// sizes so that (a) CUDA graphs (`ATLAS_DECODE_GRAPHS_MULTISEQ`) are keyed by
/// a handful of stable shapes instead of one per exact n, and (b) the batched
/// kernels see a bounded set of widths. Padding rows point at the dummy SSM
/// slot / dummy KV block and cost one wasted lane each.
///
/// This expression used to be duplicated at FOUR call sites
/// (`decode_a2.rs:168`, `decode_b.rs:52` and `:110`,
/// `phase_continue_prefills.rs:142` — the last one in a different crate), which
/// is exactly how the ladder would have drifted when a step was added. All four
/// now call here.
///
/// `12` and `16` were added for the `C=[1,2,4,8,16]` concurrency work
/// (2026-07-25): previously any n ≥ 9 fell through to `padded_n = n`, so at
/// C=16 every distinct batch composition minted its OWN CUDA graph (n=9, 10,
/// ... 16 each a separate capture) and the buffer-fit guards were computed on
/// exact n.
///
/// `24` and `32` were added for native bs=32 (2026-07-30): n=17..32 now pads
/// to two stable graph shapes instead of minting one graph per exact n.
///
/// `48`, `64`, `96` and `128` were added for native bs=64+ (2026-07-31,
/// wave-14a): the decode-metadata layout is now DERIVED from the serve
/// `max_batch_size` (`spark_runtime::buffers::DecodeMetaLayout`, rows =
/// max(32, bs), ceiling `DECODE_META_MAX_ROWS`), and
/// `upload_batch_metadata_fixed` ensures `padded_n <= rows`. Rungs above the
/// boot's `max_batch_size` are unreachable (the scheduler admits at most
/// `max_batch_size` active sequences), so every bs<=32 boot never pads past
/// 32 — byte-identical by construction. Rungs <=32 unchanged.
/// Above 128 the fall-through behaviour is unchanged (guarded downstream).
#[inline]
pub fn padded_batch_n(n: usize) -> usize {
    [2usize, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        .iter()
        .copied()
        .find(|&s| s >= n)
        .unwrap_or(n)
}

pub trait Model: Send + Sync {
    /// Release the device memory this model owns, in reverse construction
    /// order.
    ///
    /// Called by the host when the model is being replaced, **after** the
    /// scheduler has drained and the stream is synchronised — the only point at
    /// which a device free is safe on GB10, where a free interleaved with other
    /// allocation traffic corrupts neighbouring allocations. See
    /// `atlas_core::scope` for why this is not `Drop`: `Drop` can express
    /// neither the ordering nor the failure.
    ///
    /// Default: a no-op returning `Ok`, which is honest for the mock and
    /// translation models that own no pooled device memory. A model that DOES
    /// own pools and leaves this unimplemented leaks them — loudly, as the next
    /// load failing to fit, never as wrong output.
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }

    /// Poll TQ+ InnerQ calibration for this model. Called once per prefill
    /// chunk. Default: a no-op, which is every model without a driver — the
    /// scheduler used to reach a process-wide `OnceLock` for this, which meant
    /// the driver could outlive the model whose device symbols it writes.
    fn poll_innerq(&self) {}

    /// True when this model implements run-to-completion beam search
    /// ([`Self::generate_beam_batch`]). Default `false` — only encoder-decoder
    /// translation models (NLLB) override it.
    fn supports_beam(&self) -> bool {
        false
    }

    /// Run beam search to completion for each request, returning each one's
    /// winning hypothesis token ids (EOS-terminated). Called from the prefill
    /// path for `num_beams > 1` requests, bypassing the token-by-token decode
    /// loop. Default: unsupported.
    fn generate_beam_batch(&self, _reqs: &[BeamReq]) -> Result<Vec<Vec<u32>>> {
        bail!("this model does not support beam search")
    }

    /// Run prefill: process all prompt tokens through the model.
    ///
    /// Returns logits DevicePtr for the last token position.
    /// Updates KV cache and SSM states for the sequence.
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// Process `chunk_len` tokens starting at `chunk_start` in the prompt.
    /// `is_last_chunk` runs final norm + LM head; intermediate chunks return
    /// `DevicePtr::NULL`. KV blocks alloc incrementally; SSM state carries
    /// across chunks; attention uses FA on chunk 0, paged decode after.
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr>;

    /// Run one decode step: process a single new token.
    ///
    /// Returns logits DevicePtr for the new token.
    /// Updates KV cache and SSM states.
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// Run batched decode: process one token per sequence.
    ///
    /// Returns logits DevicePtr for [batch_size, vocab_size].
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr>;

    /// Process N decode tokens + an M-token prefill chunk in one pass through
    /// the same weight loads. Returns decode logits `[N, vocab]` and prefill
    /// logits `[1, vocab]` (when `is_last`). Default: serial decode + prefill.
    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<MixedForwardResult> {
        // Default: serial execution (no weight sharing)
        let decode_logits = if !decode_tokens.is_empty() {
            self.decode_batch(decode_tokens, decode_seqs, stream)?
        } else {
            spark_runtime::gpu::DevicePtr::NULL
        };
        let prefill_logits = self.prefill_chunk(
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )?;
        Ok(MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// Process N concurrent prefill chunks in one forward pass (same weight
    /// load amortised across N streams). The default implementation falls
    /// back to a per-stream loop calling `prefill_chunk` — implementors that
    /// support kernel-level batched prefill should override this.
    ///
    /// Returns a `Vec<DevicePtr>` parallel to `streams`: each entry is the
    /// last-token logits pointer for that stream when its chunk is
    /// `is_last_chunk`, or `DevicePtr::NULL` otherwise.
    ///
    /// Tracks issue Q12 in
    /// `/workspace/atlas-internal/qwen-refactor/notes.md`.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        // Default: serialized per-stream prefill_chunk. This preserves
        // current behavior for any model that doesn't override; only the
        // weight-streaming amortisation is lost vs a true batched path.
        let mut out = Vec::with_capacity(streams.len());
        for slice in streams.iter_mut() {
            let logits = self.prefill_chunk(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                slice.is_last_chunk,
                stream,
            )?;
            out.push(logits);
        }
        Ok(out)
    }

    /// Like `prefill_batch_chunk`, but each finishing stream's first-token
    /// logits land in row `row_base + stream_idx` of the shared logits arena
    /// instead of row `stream_idx`.
    ///
    /// CROSS-REQUEST CORRUPTION (the reason this exists). `decode_batch`
    /// writes lane `i`'s logits to row `i` of `buffers.logits()`, and a
    /// finishing prefill stream writes row `stream_idx` of the SAME arena —
    /// byte-identical addresses. In a mixed step decode runs first and the
    /// caller samples the decode rows AFTER the prefill sub-pass, so every
    /// active decode lane whose index collides with a finishing prefill
    /// stream samples THAT REQUEST'S first-token distribution instead of its
    /// own. Symptoms are exactly what a foreign distribution looks like: a
    /// stray `<tool_call>` opener at the head of a reply (the foreign stream
    /// was tool-enabled), or a reply that veers onto another user's topic.
    /// It cannot happen sequentially — a mixed step needs >=2 prefills and
    /// >=1 active decode in the same tick.
    ///
    /// Giving prefill a disjoint row window is enough to fix it. The arena
    /// holds `min(max_batch_tokens, 32)` rows against `n_decode + n_prefill
    /// <= max_num_seqs`, so the shifted window fits with room to spare;
    /// implementations MUST bounds-check and fall back to `row_base = 0`
    /// rather than write past the arena.
    ///
    /// Default ignores `row_base` (models with no batched prefill of their
    /// own can't alias, since the serial path returns one row).
    fn prefill_batch_chunk_rows(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        _row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        self.prefill_batch_chunk(streams, stream)
    }

    /// Generalised mixed forward: M decode tokens + N concurrent prefill
    /// chunks fused into one forward pass. Default: delegates to
    /// `decode_batch` + `prefill_batch_chunk` serially. Models that
    /// implement true mixed batching should override.
    fn mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<MixedBatchResult> {
        // Default: serial execution.
        let decode_logits = if !decode_tokens.is_empty() {
            let lg = self.decode_batch(decode_tokens, decode_seqs, stream)?;
            // #110: decode_batch runs its whole forward on the DEFAULT stream
            // — both `decode_batch_compute_main` (n>=2) and the n==1 graph path
            // ignore the `stream` arg and hardcode `gpu.default_stream()`. The
            // batched prefill below reuses the SAME shared arena buffers
            // (hidden_states/residual/scratch/gdn) but submits on `stream`
            // (prefill_stream). With no barrier the two sub-passes execute
            // concurrently on two different streams and race over those
            // buffers — corrupting the batched prefill's slot table into wild
            // KV-cache indices and faulting with a CUDA illegal access
            // (status 700). Synchronize the decode stream so its buffer use is
            // fully retired before prefill overwrites them. This runs once per
            // mixed step (active+prefilling), never in the hot decode loop.
            self.synchronize(self.default_stream())?;
            lg
        } else {
            spark_runtime::gpu::DevicePtr::NULL
        };
        // Prefill rows start ABOVE the decode lanes: decode owns rows
        // 0..decode_tokens.len(), so a finishing prefill stream can no longer
        // overwrite a lane whose logits the caller has not sampled yet. See
        // `prefill_batch_chunk_rows`.
        let prefill_logits =
            self.prefill_batch_chunk_rows(prefill_streams, stream, decode_tokens.len())?;
        Ok(MixedBatchResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// Normalize SSM h_state norms to prevent catastrophic state explosion
    /// during long chunked prefill. Called between chunks by the scheduler.
    /// Default: no-op (models without SSM layers don't need normalization).
    fn normalize_ssm_states(&self, _seq: &SequenceState, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Per-layer chunked prefill: SSM layers use three phases (proj →
    /// single-launch GDN → post) so the recurrence sees the full sequence
    /// in one launch; attention layers use standard chunked prefill.
    /// Returns last-token logits. Default: single-chunk prefill (no SSM).
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Default: single-chunk prefill (no two-phase benefit without SSM)
        self.prefill_chunk(tokens, seq, 0, tokens.len(), true, stream)
    }

    /// Vocab size (for sampler allocation).
    fn vocab_size(&self) -> usize;

    /// Runtime LoRA adapter rotation: select the resident adapter named `name`
    /// as active (re-points the delta pool pointers). MUST be called at a
    /// scheduler quiescent point (no in-flight decode). Graph-safety is via the
    /// eager-on-rotate gate. Default: unsupported (non-LoRA or non-rotatable).
    fn set_active_lora(&mut self, _name: &str) -> Result<()> {
        bail!("this model does not support LoRA adapter rotation")
    }

    /// Task #24: stable adapter_id (KV/prefix-cache identity) for a per-request
    /// pool-slot selector. `slot` follows `SequenceState.adapter_slot`: `>= 0`
    /// picks that resident slot, `-1` defers to the installed active adapter.
    /// The default (no LoRA) returns the base sentinel `0`, keeping the prefix
    /// cache byte-identical to the pre-LoRA path.
    fn adapter_id_for(&self, _slot: i32) -> u64 {
        0
    }

    /// Task #25: acquire a per-slot ref when a sequence begins using its adapter
    /// (at prefill), resolving `-1 -> active` like [`Self::adapter_id_for`].
    /// Returns the RESOLVED pool index the ref was taken on (store it, release
    /// EXACTLY that index at terminal free — immune to a rotate changing active).
    /// Default (no LoRA) returns `-1` "nothing acquired" so the release guard
    /// skips and the base path is byte-identical.
    fn acquire_adapter_slot(&self, _slot: i32) -> i32 {
        -1
    }

    /// Task #25: release a per-slot ref acquired by [`Self::acquire_adapter_slot`],
    /// by the RESOLVED index it returned. `-1` is a no-op. Default: no-op.
    fn release_adapter_slot(&self, _resolved: i32) {}

    /// Runtime LoRA adapter dynamic-load: load the adapter at `dir` INTO pool
    /// `slot` and make it resident there (pool-size-1 per-request weight change).
    /// MUST be called at a scheduler quiescent point; needs rotation armed.
    /// Default: unsupported (non-LoRA or non-rotatable).
    fn swap_lora_from_disk(
        &mut self,
        _dir: &std::path::Path,
        _name: &str,
        _slot: usize,
    ) -> Result<()> {
        bail!("this model does not support LoRA disk swap")
    }

    /// Task #27 (demand-driven promotion): RDMA-promote the adapter `name`
    /// (staged on `peer_addr` at `adapter_id`) from the peer into a cache pool
    /// slot and make it active, returning `(slot, evicted_name)`. Runs at a
    /// scheduler quiescent point. `peft` supplies the r/alpha/scaling the peer
    /// manifest does not carry. Default: unsupported (non-LoRA / non-cuda).
    fn promote_lora_from_peer(
        &mut self,
        _peer_addr: &str,
        _adapter_id: &str,
        _name: &str,
        _peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        bail!("this model does not support LoRA peer promotion")
    }

    /// Demand-driven DISK promotion (no RDMA/peer): load the adapter `name` from
    /// `adapter_dir` into a cache pool slot (LRU victim) and make it active,
    /// returning `(slot, evicted_name)`. Local-disk sibling of
    /// [`Self::promote_lora_from_peer`]; the swap re-parses the dir's
    /// `adapter_config.json`, so no `peft` arg. Runs at a scheduler quiescent
    /// point; needs rotation armed. Default: unsupported.
    fn promote_lora_from_disk(
        &mut self,
        _adapter_dir: &std::path::Path,
        _name: &str,
    ) -> Result<(usize, Option<String>)> {
        bail!("this model does not support LoRA disk promotion")
    }

    /// Dims for the `--high-speed-swap` orchestrator (installed thread-local
    /// after `bind_gpu_to_thread`). `None` for legacy/non-attention models.
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        None
    }

    /// Bind the GPU context to the current thread.
    /// Must be called from any thread other than the one that created the model.
    fn bind_gpu_to_thread(&self) -> Result<()>;

    /// Allocate a new SequenceState with SSM states.
    fn alloc_sequence(&self) -> Result<SequenceState>;

    /// [`Self::alloc_sequence`] told what this request can actually reach
    /// (`prompt_len + max_tokens`). Proposer state that scales with context is
    /// sized to THAT instead of `--max-seq-len`; see
    /// `DraftProposer::alloc_state_for`. Defaults to the unsized form.
    fn alloc_sequence_for(&self, budget_tokens: usize) -> Result<SequenceState> {
        let _ = budget_tokens;
        self.alloc_sequence()
    }

    /// Copy logits from device to host buffer (for CPU-side sampling).
    ///
    /// `logits_ptr` points to `[vocab_size]` BF16 values on device.
    /// `dst` must be at least `vocab_size * 2` bytes.
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// FP32 logits flag (host buffer needs `vocab*4` bytes, reinterpret `&[f32]`).
    /// True only for Gemma-4 dense single-token decode `lm_head`; default false.
    fn logits_ptr_is_fp32(&self, _logits_ptr: DevicePtr) -> bool {
        false
    }

    /// Base pointer of the on-device logits buffer (`[k, vocab]` BF16 after
    /// `decode_verify_graphed`). Lets the scheduler read logits for temp
    /// sampling even though graphs bake in argmax.
    fn logits_buffer_ptr(&self) -> DevicePtr;

    /// GPU argmax: 4-byte D2H copy vs 304KB BF16 D2H + CPU argmax.
    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32>;

    /// GPU batched argmax over `[N, vocab]` BF16; returns N token IDs.
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>>;

    /// Return the hidden state after final norm from the last decode step.
    ///
    /// Used by MTP speculative decoding: the MTP head takes the target model's
    /// post-norm hidden states as input alongside the token embedding.
    fn hidden_after_norm(&self) -> DevicePtr;

    /// L2-resident multi-token verification: per-position argmax token IDs;
    /// each token advances KV/SSM state. All tokens go through each layer
    /// before moving on so weights stay in L2.
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>>;

    /// Checkpoint SSM states before speculative verification.
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()>;

    /// Rollback SSM states after partial acceptance.
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()>;

    /// True when this model has recurrent SSM / Mamba layers whose
    /// `h_state` + `conv_state` are advanced in-place every decoded
    /// token.
    ///
    /// Pure-attention models return `false` (the default): their only
    /// per-token state is the paged KV cache, which the Phase-C
    /// boundary rollback rewinds by lowering `seq_len`. Hybrid models
    /// (Qwen3.6-A3B, MiniMax, Nemotron-nano) return `true` — for those
    /// the scheduler MUST also restore the SSM state from a decode-time
    /// snapshot, because the recurrent state cannot be undone by
    /// lowering a cursor.
    fn has_ssm_layers(&self) -> bool {
        false
    }

    /// Verify DRAFT capacity of the MTP state pools for a sequence
    /// occupying SSM pool slot `slot_idx` — the deepest `num_drafts` a
    /// speculative step may dispatch to it without overflowing its slot's
    /// per-token H-intermediate allocation (tiered since 2026-08-16; SSOT
    /// `ssm_reserve::verify_slot_h_intermediates`). The scheduler clamps
    /// every spec step's draft count to the MINIMUM capacity across the
    /// active slots. Default `usize::MAX`: no SSM verify pools to
    /// constrain (pure-attention models, spec off).
    fn mtp_slot_draft_capacity(&self, _slot_idx: usize) -> usize {
        usize::MAX
    }

    /// Number of decode-rollback SSM snapshot slots reserved **per
    /// active sequence** (Phase-C). The scheduler's per-sequence
    /// snapshot ring is sized from this. `0` (the default) means the
    /// model keeps no decode-rollback snapshots — appropriate for
    /// pure-attention models and for SSM models when the snapshot pool
    /// has no capacity reserved. SSM models with a populated pool
    /// override to `ROLLBACK_RESTEER_CAP + 1`.
    fn decode_rollback_ring_slots(&self) -> usize {
        0
    }

    /// Save `seq`'s live SSM `h_state` + `conv_state` (all SSM layers)
    /// into the decode-rollback snapshot slot `ring_slot`.
    ///
    /// `ring_slot` is a per-sequence ring index in
    /// `[0, decode_rollback_ring_slots())`; the model maps it to a
    /// concrete snapshot-pool slot keyed by `seq.slot_idx`. Reuses the
    /// same `SsmSnapshotPool` D2D copy primitive as Marconi prefix
    /// caching and MTP verify (SSOT — one snapshot mechanism).
    ///
    /// Default: no-op `Ok(())` for pure-attention models, which have no
    /// SSM state to snapshot.
    fn save_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// Restore `seq`'s SSM `h_state` + `conv_state` (all SSM layers)
    /// from the decode-rollback snapshot slot `ring_slot` previously
    /// written by [`Self::save_decode_ssm_snapshot`].
    ///
    /// Default: no-op `Ok(())` for pure-attention models.
    fn restore_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// Speculative decoding via the model's internal MTP proposer; falls
    /// back to regular decode when no proposer is wired up.
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult>;

    /// Check if speculative decoding is available (MTP or self-speculative).
    fn has_proposer(&self) -> bool;
    /// The installed DFlash drafter's block size γ, when one is installed.
    /// The serve layer derives `num_drafts = γ - 1` from THIS (the head is
    /// the SSOT — it resolved the drafter config's trained block size),
    /// never from a CLI default that may not match the checkpoint.
    fn dflash_gamma(&self) -> Option<usize> {
        None
    }

    /// Check if self-speculative decoding is enabled.
    fn has_self_speculative(&self) -> bool;

    /// Eager decode skipping SSM layers. Used by self-speculative drafting.
    /// Returns logits pointer for argmax. Advances seq_len by 1.
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// Insert the full token sequence (prompt + generated) into the prefix
    /// cache. Call BEFORE `free_sequence()` (block indices must still be
    /// valid). Benefits multi-turn agentic sessions that resend full history.
    fn cache_sequence(&self, seq: &SequenceState);

    /// #155 iter3: during decode, save a block-aligned Marconi SSM snapshot
    /// at checkpoint-interval boundaries so the NEXT turn's warm prefix-cache
    /// hit restores from decode-produced state near the conversation's end —
    /// instead of replaying decode-produced tokens through the prefill kernel
    /// (the warm-hit drift ratchet, issue #155). Called from the scheduler
    /// after each decode step's live SSM state is canonical (post-commit on
    /// the MTP path). Default no-op (non-hybrid models / caching disabled).
    fn decode_marconi_checkpoint(&self, _seq: &mut SequenceState) {}

    /// Free all GPU resources associated with a sequence.
    ///
    /// Releases KV cache blocks and returns SSM state pool slot.
    /// Must be called when a sequence is no longer needed.
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()>;

    /// Move a sequence's SSM states to a different pool slot.
    ///
    /// Copies h_state and conv_state across all SSM layers from the current
    /// slot to `new_slot`. Used by the scheduler for slot compaction after
    /// swap_remove to keep active sequences at contiguous slots [0..N).
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()>;

    /// Disown a retired sequence's SSM pool slot after `compact_sequence`
    /// migrated it to a surviving sequence.
    ///
    /// Sets the `slot_idx` reuse sentinel AND neutralizes the sequence's
    /// internal slot-release guard so the migrated slot is NOT released when
    /// this sequence is later freed or dropped (the surviving sequence now owns
    /// it). The scheduler MUST call this — instead of mutating `slot_idx`
    /// directly — immediately after a `compact_sequence` that reuses this
    /// sequence's slot, so a subsequent early-return/drop cannot double-release.
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState);

    /// CUDA-graphed K=2 verify: 2 tokens, capture-then-replay. Returns
    /// `[verified_0, verified_1]` argmax IDs. SSM intermediates saved for
    /// partial rollback via `rollback_ssm_states`.
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]>;

    /// CUDA-graphed K=3 verify (1 verified + 2 drafts). Returns 3 argmax IDs.
    /// SSM intermediates `[0]` and `[1]` are saved for partial rollback.
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 3]>;

    /// CUDA-graphed K=4 verify (1 verified + 3 drafts). Returns 4 argmax IDs.
    /// SSM intermediates [0..3] saved for partial rollback.
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 4]>;

    /// Whether [`Self::decode_verify_batched`] can run for `ks.len()`
    /// sequences at `ks[i]` verify rows each (one more than that sequence's
    /// draft count; the K-vs-batch ladder passes 2..=4, and D-Cut makes the
    /// vector RAGGED — uniform is just the special case).
    ///
    /// Default `false`: the scheduler MUST fall back to the per-sequence
    /// `decode_verify_graphed_k{2,3,4}` loop. There is deliberately NO
    /// default loop impl of the batched form — a loop over the per-seq
    /// verify would leave the shared logits buffer holding only the LAST
    /// sequence's rows and silently poison row-based pipeline picks.
    fn can_batch_verify(&self, _ks: &[usize]) -> bool {
        false
    }

    /// Batched K-row verify: `ks.len()` sequences × `ks[i]` rows in ONE eager
    /// forward (flat seq-major rows, `tokens.len() == Σ ks`). Weight matrices
    /// are read once for all `Σ ks` rows. Sequence i occupies rows
    /// `[off_i, off_i + ks[i])` where `off_i = Σ_{t<i} ks[t]`, holding
    /// `[last_verified, d0, .., d_{ks[i]-2}]`. Returns the `Σ ks` argmax IDs
    /// in the same flat order. On success each sequence's `tokens`/`seq_len`
    /// advance by its own `ks[i]` (rewind is the caller's verdict arithmetic,
    /// same as the per-seq path). On Err NO sequence state has been advanced.
    ///
    /// Callers must gate on [`Self::can_batch_verify`].
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<u32>> {
        let _ = (tokens, ks, seqs, stream);
        bail!("decode_verify_batched: unsupported by this model")
    }

    /// Copy raw-hidden rows `rows[i]` of the just-run batched verify forward
    /// into stash slot `i` (`verify_hidden_stash`), BEFORE any propose
    /// clobbers the shared `hidden_states` buffer. Companion of
    /// [`Self::decode_verify_batched`].
    fn stash_verify_hidden_rows(&self, rows: &[usize], stream: u64) -> Result<()> {
        let _ = (rows, stream);
        bail!("stash_verify_hidden_rows: unsupported by this model")
    }

    /// Stashed-row variant of [`Self::save_hidden_for_mtp`]: copy stash slot
    /// `idx` (written by [`Self::stash_verify_hidden_rows`]) into the MTP
    /// input buffer. Used by the batched-verify verdict path, whose propose
    /// calls have already overwritten the live verify rows.
    fn save_hidden_for_mtp_from_stash(&self, idx: usize, stream: u64) -> Result<()> {
        let _ = (idx, stream);
        bail!("save_hidden_for_mtp_from_stash: unsupported by this model")
    }

    /// Batched cross-sequence MTP propose for the batched K=4 verify path:
    /// `num_drafts` drafts for each of `tokens.len()` sequences, reading
    /// every drafter weight once per draft position instead of once per
    /// sequence. `stash_idx[i]` names the verify-stash slot holding sequence
    /// i's accepted-position hidden (written by
    /// [`Self::stash_verify_hidden_rows`]); `positions[i]` is the propose
    /// position (post-rewind `seq_len`), matching the per-seq
    /// [`Self::run_mtp_propose_multi`] contract. Grammarless sequences only.
    ///
    /// `out_conf`, when `Some`, receives each draft's top-1 LOG-probability
    /// (`ln p`, same shape as the returned drafts) — the D-Cut ranking key.
    /// It is filled with zeros (certainty) when the drafter cannot measure
    /// confidence, so a caller ranking by prefix product never prunes on a
    /// value nobody produced.
    ///
    /// `Ok(None)` = unsupported (caller falls back to the per-seq propose
    /// loop, re-saving each stash slot first). Default: unsupported.
    #[allow(clippy::too_many_arguments)]
    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let _ = (
            tokens, positions, stash_idx, num_drafts, seqs, stream, out_conf,
        );
        Ok(None)
    }

    /// Widest batch [`Self::run_mtp_propose_batched`] can carry in ONE
    /// drafter forward per draft position. `1` = per-sequence only.
    /// Schedulers chunk their propose groups by this — never by a constant.
    fn mtp_propose_batch_max(&self) -> usize {
        1
    }

    /// DFlash K=γ graphed verify (γ+1 tokens). Specialization of the K=2/3/4
    /// pattern for arbitrary K. Default impl falls back to eager
    /// `decode_verify`. Models can override for CUDA-graph speedup keyed by
    /// `(slot_idx, K)`.
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify(tokens, seq, stream)
    }

    /// DFlash γ-token verification: 1 verified + γ drafts → per-position
    /// argmax. Variable-length γ (vs fixed K=2/3/4) because it's a drafter
    /// config field. CUDA-graph capture keyed by `(slot_idx, tokens.len())`.
    /// Default routes to `decode_verify_graphed_kgamma`.
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        // Phase 2.5e: route to the K=γ graphed path. Models that don't
        // override `decode_verify_graphed_kgamma` get the eager fallback
        // for free (the trait default does that).
        self.decode_verify_graphed_kgamma(tokens, seq, stream)
    }

    /// DFlash fused decode+verify: one M=(1+k) forward replacing separate
    /// M=1 decode + M=k verify on the DFlash path.
    ///
    /// `tokens[0]` = accepted/decode token; `tokens[1..]` = draft block.
    /// `try_dflash_capture` fires at row 0 so the DFlash drafter conditions
    /// on the confirmed-accepted token's per-layer hidden, never on a
    /// potentially-rejected draft's hidden.
    ///
    /// CUDA-graph cache keyed by `(slot_idx, tokens.len())`. Default falls
    /// back to `decode_verify_graphed_kgamma` (which itself falls back to
    /// eager `decode_verify`) for models that don't override.
    fn decode_and_verify_fused(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma(tokens, seq, stream)
    }

    /// Save the post-norm hidden state at `token_idx` (0 or 1) to a
    /// dedicated MTP input buffer. Must precede `run_mtp_propose` — MTP
    /// overwrites shared buffers including `norm_output`.
    fn save_hidden_for_mtp(&self, token_idx: usize, stream: u64) -> Result<()>;

    /// ATLAS_MTP_CATCHUP: ring-capture a serially decoded token's final
    /// hidden at `pos` for the drafter catch-up feed. Default no-op.
    fn save_hidden_for_catchup(&self, _token_idx: usize, _pos: usize) -> Result<()> {
        Ok(())
    }

    /// Capture `hidden_states[token_idx]` from every DFlash capture layer
    /// into `dflash_hidden_save`. Called after gamma verify Phase 3 D2H
    /// sync (bonus position known). No-op when DFlash is disabled.
    fn save_dflash_hidden_for_propose(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Append the accepted draft's hidden state (row 1 of dflash_hidden_save)
    /// into the proposer context. Base primitive for both legacy and Eagle paths.
    /// Default no-op for models without a DFlash drafter.
    fn dflash_accept_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// EAGLE-fix (K=2 accept): append row 0 @ N then row 1 @ N+1 BEFORE propose
    /// so forward_block conditions on row 1 (the hidden that generated bonus).
    /// Default no-op for models without a DFlash drafter.
    fn dflash_eagle_accept_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// EAGLE-fix (K=gamma): append rows 0..=num_accepted at positions
    /// base_pos..=base_pos+num_accepted. Row num_accepted is appended LAST ->
    /// freshest ctx slot = the hidden that generated the bonus (EAGLE).
    /// Default no-op for models without a DFlash drafter.
    fn dflash_eagle_kgamma_append(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _base_pos: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Ctx-holes fix (serial decode): append the just-decoded token's
    /// captured per-layer hidden (`dflash_hidden_save` row 0, filled by
    /// `try_dflash_capture` inside the decode layer loop) into the seq's
    /// DFlash ctx accumulator, stamped at its true position
    /// (`seq.seq_len - 1`, matching propose.rs's decode-append convention).
    ///
    /// Called from the scheduler's serial bootstrap path when adaptive
    /// speculation has SUSPENDED this seq — propose() never runs there, so
    /// without this hook every serially-decoded token's target hidden is
    /// overwritten (single-slot model capture) and permanently lost,
    /// leaving holes in the drafter's ctx at spec re-entry (measured
    /// -0.42 accepted/step on think-gated vs spec-through-think content).
    ///
    /// Sets `skip_next_decode_append` so a propose() firing later (re-probe)
    /// does not double-append the same capture. Graceful no-op when DFlash
    /// is disabled or the seq has a non-DFlash proposer state.
    fn dflash_serial_ctx_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// Unified DFlash ctx commit (ATLAS_DFLASH_UNIFIED_CTX=1). Copies
    /// `num_committed` scratch rows (`dflash_hidden_save` rows
    /// `scratch_row..scratch_row+num_committed`) into `ctx_hidden_acc` at the
    /// CURRENT TAIL (`ctx_len`), stamping RoPE positions
    /// `base_pos..base_pos+num_committed`, folding the watermark slide in
    /// first. `base_pos` is the RoPE position, NOT the acc row index (they
    /// diverge after a watermark slide — DDD §4.1 landmine). `scratch_row` is
    /// 0 on every single-sequence path; batched decode (n>1) captures ALL
    /// batch rows, so seq i commits from scratch row i. The single structural
    /// replacement for the ~5 fragmented appends. Default no-op for models
    /// without a DFlash drafter.
    fn commit_ctx(
        &self,
        _seq: &mut SequenceState,
        _num_committed: usize,
        _base_pos: usize,
        _scratch_row: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Rows per per-sequence capture BAND in the DFlash hidden scratch (γ+1).
    /// Sequence `i` of a batched K=γ verify captures into band `i`, so its
    /// `commit_ctx` `scratch_row` is `i * dflash_capture_band()`. Returning
    /// the model's own stride keeps the capture and the commit from ever
    /// disagreeing. `0` when there is no DFlash drafter.
    fn dflash_capture_band(&self) -> usize {
        0
    }

    /// Run the MTP proposer for one draft token off the saved hidden state.
    /// `None` when no proposer is wired.
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<u32>>;

    /// Run the MTP proposer to generate multiple draft tokens.
    ///
    /// Uses the hidden state previously saved via `save_hidden_for_mtp`.
    /// Returns empty vec if no MTP proposer is available.
    ///
    /// `grammar_bitmask`: when `Some`, drafts are constrained to the allowed
    /// token set of an XGrammar matcher at its current position. Format is
    /// `ceil(vocab_size / 32)` i32 words; bit `tok` set ⇒ allowed. `None`
    /// preserves the unconstrained GPU-argmax fast path.
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>>;

    /// Read the draft token ID stored on GPU by the last `run_mtp_propose_multi`
    /// call (which used `embed_from_argmax` to write the draft embedding and
    /// token ID directly on GPU). Returns 0 if no proposer is available.
    fn read_deferred_draft_token(&self) -> Result<u32> {
        Ok(0)
    }

    /// Encode images through the vision encoder and store embeddings for the next prefill.
    ///
    /// Each tuple is `(pixels: Vec<f32>, grid_h: usize, grid_w: usize)`.
    /// Pixels are laid out [P, C×T×Hp×Wp] matching `vision_preprocess::preprocess_image`.
    /// Must be called before `prefill_chunk` when the prompt contains `<|image_pad|>` tokens.
    ///
    /// Default: no-op (text-only models).
    fn prepare_vision_embed(&self, _images: &[crate::VisionItem]) -> Result<()> {
        Ok(())
    }

    /// Batched vision encode across N requests' images in ONE `forward_batched`
    /// call (block GEMM weights read once over Σpatches). `per_request[i]` is
    /// request i's images. Returns one `(patch_row_offset, grid_index_offset,
    /// num_images, patch_row_count)` per request, in request order, locating
    /// its slice of the shared packed `buf_out`. Default: no-op (text models).
    fn prepare_vision_embed_batched(
        &self,
        _per_request: &[Vec<crate::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        Ok(Vec::new())
    }

    /// Set the co-dispatched batched-ViT slice base for the NEXT prefill_chunk
    /// (row offset into buf_out, grid index offset, image count owned). Pass
    /// (0,0,0) to reset to the legacy single-request behaviour. Default: no-op.
    fn set_vision_slice_base(&self, _row_base: usize, _grid_base: usize, _owned_images: usize) {}

    /// EP worker step: receive a (seq_id, cmd) preamble from rank 0 and
    /// execute the command in the addressed slot.
    ///
    /// Returns false when the worker should shut down.
    /// Only valid on rank > 0 with EP enabled.
    ///
    /// `slots` must be sized to `args.max_batch_size` (same as the head's
    /// scheduler `active` capacity); commands with `seq_id >= slots.len()`
    /// fail loudly rather than corrupt unrelated state.
    fn ep_worker_step(&self, _slots: &mut [Option<SequenceState>]) -> Result<bool> {
        Ok(true) // no-op for non-EP models
    }

    /// Check whether expert parallelism (EP) is enabled (multi-GPU MoE).
    ///
    /// When true, the scheduler must use separate decode + prefill commands
    /// with explicit EP broadcasts rather than mixed_forward (which has no
    /// EP broadcast protocol defined).
    fn is_ep(&self) -> bool {
        false
    }

    /// True when single-token decode `lm_head` writes FP32 logits to a
    /// dedicated FP32 scratch buffer (rather than the shared BF16 logits
    /// buffer). Callers that consume those logits must read from
    /// [`Self::decode_logits_ptr`] using 4 bytes/element. Defaults false;
    /// only Gemma-4 dense overrides today (gated by
    /// `ATLAS_GEMMA4_FP32_LMHEAD=1`).
    fn decode_logits_fp32(&self) -> bool {
        false
    }

    /// Buffer pointer the single-token decode `lm_head` last wrote to. The
    /// returned dtype is FP32 when [`Self::decode_logits_fp32`] is true,
    /// BF16 otherwise. The default impl returns the shared BF16 logits
    /// buffer used by every existing model. Override on models that route
    /// the lm_head output through an FP32 scratch (Gemma-4 + softcap).
    fn decode_logits_ptr(&self) -> DevicePtr {
        // Default: shared BF16 logits buffer. Models with FP32 lm_head
        // override.
        // NOTE: this default panics when the trait method is invoked on
        // models that don't implement either accessor. TransformerModel
        // overrides both. If a future model needs only one, it must
        // override both for consistency.
        unreachable!(
            "Model::decode_logits_ptr() must be overridden alongside \
             decode_logits_fp32() — default cannot return a valid pointer."
        )
    }

    /// Multi-head Latent Attention guard. When true, chunked prefill MUST run
    /// as a single chunk — Atlas has no paged-MLA prefill kernel and
    /// multi-chunk MLA silently corrupts attention output (see Mistral-Small-4
    /// 2026-05-01 sweep: 8K collapses to "The\nThe…").
    fn is_mla(&self) -> bool {
        false
    }

    /// Tokens per paged-KV block, or `None` when the model has no paged KV.
    /// The scheduler uses this to land a prefill chunk boundary exactly on the
    /// block boundary a warm turn will match at (see
    /// `spark_runtime::ssm_tail_boundary`).
    fn kv_block_size(&self) -> Option<usize> {
        None
    }

    /// EP broadcast: send a command (u32) to all worker ranks.
    ///
    /// Called by rank 0 before each model operation to synchronize workers.
    /// Only valid when EP is enabled.
    fn ep_broadcast_cmd(&self, _cmd: u32) -> Result<()> {
        Ok(()) // no-op for non-EP models
    }

    /// EP broadcast: send a `(seq_id, cmd)` pair to all worker ranks.
    ///
    /// Use this at the *first* broadcast of a logical command sequence
    /// (e.g. the K=2 verify marker, prefill start, decode token, etc.).
    /// Follow-up broadcasts within the same command (chunk metadata, more
    /// tokens, accept/reject result) keep using [`Self::ep_broadcast_cmd`]
    /// — the worker consumes the preamble once per command and routes
    /// subsequent reads through the slot it identified.
    ///
    /// When [`Self::ep_protocol_v2`] returns false (the default), the
    /// `seq_id` is ignored on the wire and behaviour matches the legacy
    /// single-sequence broadcast.
    fn ep_broadcast_cmd_for_seq(&self, _seq_id: u32, _cmd: u32) -> Result<()> {
        Ok(()) // no-op for non-EP models
    }

    /// Returns true if this model's EP comm path is using the v2 protocol
    /// (slot-aware seq_id preamble). Default false — pre-PR behaviour.
    fn ep_protocol_v2(&self) -> bool {
        false
    }

    /// EP bulk broadcast: send an array of u32 tokens to all worker ranks.
    /// Uses a single NCCL broadcast instead of per-token broadcasts.
    fn ep_broadcast_tokens(&self, _tokens: &[u32]) -> Result<Vec<u32>> {
        Ok(Vec::new()) // no-op for non-EP models
    }

    /// Trim the MTP proposer's KV cache after verification.
    ///
    /// Called on rejection to discard the rejected draft's MTP KV entry.
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()>;

    /// Launch SSM state checkpoint D2D copies on a secondary CUDA stream.
    ///
    /// Non-blocking: returns immediately. The copies can overlap with MTP
    /// propose on the default stream since they access disjoint memory.
    /// Call `sync_secondary` before the next verify to ensure completion.
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        // Default: fall back to synchronous checkpoint.
        self.checkpoint_ssm_states(seq)
    }

    /// Launch SSM state rollback + checkpoint on the secondary stream.
    ///
    /// Used on the reject path: rollback to `intermediate[0]`, then checkpoint
    /// the rolled-back state for the next verify iteration.
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        // Default: fall back to synchronous operations.
        self.rollback_ssm_states(seq, num_accepted)?;
        self.checkpoint_ssm_states(seq)
    }

    /// Wait for all work on the secondary stream to complete.
    fn sync_secondary(&self) -> Result<()> {
        Ok(()) // No-op if no secondary stream.
    }

    /// Item #2 (STree-style in-place verify commit): commit the surviving
    /// prefix of a verify pass directly onto the canonical `h_state` /
    /// `conv_state`. Full accept (`num_accepted == k`) is a no-op (the
    /// kernel's final state is already live); partial accept is a single
    /// index-select of `h_state_intermediates[num_accepted-1]`. No-op
    /// default for backends without the dual-buffer SSM state.
    /// Runs on `secondary_stream`; pair with `sync_secondary`.
    fn commit_accepted_prefix(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _k: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Save KV blocks + SSM state to writer. Does NOT free resources.
    ///
    /// Format: `[KV layers × blocks × (K + V)]` then `[SSM layers × (h + conv)]`.
    /// The model owns the serialization format.
    fn save_sequence_state(
        &self,
        _seq: &SequenceState,
        _writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// Restore KV blocks + SSM state from reader into an allocated sequence.
    ///
    /// Allocates `num_blocks` new KV blocks, fills from reader, restores SSM.
    fn restore_sequence_state(
        &self,
        _seq: &mut SequenceState,
        _num_blocks: usize,
        _reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// Whether `tokens` contains a vision pad token for this model — i.e.
    /// the KV at those positions came from image/video EMBEDDINGS that a
    /// plain token re-prefill cannot reproduce. Decode-time preemption uses
    /// this to exclude vision sequences from the requeue-with-re-prefill
    /// path (the spill path, which saves KV verbatim, stays eligible).
    /// Default false: pure-text models are always re-prefillable.
    fn tokens_contain_vision_pad(&self, _tokens: &[u32]) -> bool {
        false
    }

    /// Number of free KV cache blocks available for allocation.
    fn num_free_blocks(&self) -> usize {
        0
    }

    /// Total KV blocks in the paged cache (denominator for occupancy
    /// gauges). Default 0 for backends without a paged cache.
    fn num_total_blocks(&self) -> usize {
        0
    }

    /// Reclaim up to `num_blocks` blocks from the prefix cache, returning how
    /// many actually became free.
    ///
    /// The prefill/decode allocators reclaim implicitly (`try_alloc` → evict →
    /// retry), but swap-in cannot: it gates on `num_free_blocks()` BEFORE
    /// attempting a restore, so cached-but-evictable capacity is invisible to
    /// it and a swapped-out sequence waits for blocks that are never
    /// volunteered. Cached blocks are legitimately held (the cache owns one ref
    /// per radix node), so nothing frees them on its own — the swap-in path has
    /// to ask. Returns 0 when nothing is evictable, which the caller must treat
    /// as "no progress possible" rather than retrying forever.
    fn reclaim_prefix_blocks(&self, _num_blocks: usize) -> usize {
        0
    }

    /// Return the default CUDA stream handle.
    fn default_stream(&self) -> u64 {
        0
    }

    /// Create a new CUDA stream (for overlapping prefill with decode).
    fn create_stream(&self) -> Result<u64> {
        Ok(0)
    }

    /// Create a CUDA event (for inter-stream synchronization).
    fn create_event(&self) -> Result<u64> {
        Ok(0)
    }

    /// Record an event on a stream (marks a point in the stream's work).
    fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Make a stream wait for an event (GPU-side sync, CPU does not block).
    fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
        Ok(())
    }

    /// Block the host until all work submitted to `stream` has completed.
    /// Used by `mixed_forward_batch` to retire the decode pass (which runs on
    /// the default stream) before the batched prefill reuses the shared arena
    /// buffers on another stream (#110). Default no-op for non-CUDA mocks.
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod padded_batch_n_tests {
    use super::padded_batch_n;

    /// Rungs <= 32 must be UNCHANGED by the wave-14a widening (byte-identity
    /// for every bs <= 32 boot), and the new rungs must cover n=33..128.
    #[test]
    fn ladder_rungs() {
        // Legacy rungs (aacd29cb and earlier) — must not move.
        for (n, want) in [
            (1usize, 2usize),
            (2, 2),
            (3, 4),
            (5, 8),
            (9, 12),
            (13, 16),
            (16, 16),
            (17, 24),
            (25, 32),
            (32, 32),
        ] {
            assert_eq!(padded_batch_n(n), want, "n={n}");
        }
        // Wave-14a rungs (only reachable when the boot's max_batch_size
        // admits that many active sequences).
        for (n, want) in [
            (33usize, 48usize),
            (48, 48),
            (49, 64),
            (64, 64),
            (65, 96),
            (96, 96),
            (97, 128),
            (128, 128),
        ] {
            assert_eq!(padded_batch_n(n), want, "n={n}");
        }
        // Above the ladder: fall-through unchanged.
        assert_eq!(padded_batch_n(129), 129);
    }
}
