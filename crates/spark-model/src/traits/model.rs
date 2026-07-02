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

pub trait Model: Send + Sync {
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
        let prefill_logits = self.prefill_batch_chunk(prefill_streams, stream)?;
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

    /// Save the post-norm hidden state at `token_idx` (0 or 1) to a
    /// dedicated MTP input buffer. Must precede `run_mtp_propose` — MTP
    /// overwrites shared buffers including `norm_output`.
    fn save_hidden_for_mtp(&self, token_idx: usize, stream: u64) -> Result<()>;

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
    fn prepare_vision_embed(&self, _images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        Ok(())
    }

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

    /// F62 (2026-04-27): copy canonical SSM state from `*_checkpoint` into
    /// `*_state` BEFORE verify so the kernel can scratch-write it. Runs on
    /// default_stream (FIFO ordering with the next kernel). No-op default
    /// for non-MTP backends.
    fn pre_verify_copy_async(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
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

    /// F62 (2026-04-27): commit a verify pass to the canonical SSM state.
    /// `num_accepted ∈ [0, k]`: full accept → copy `h_state` → checkpoint;
    /// partial → copy `h_state_intermediates[num_accepted-1]`; full reject →
    /// no-op. Runs on secondary_stream; pair with `sync_secondary`.
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        // Default: fall back to synchronous behavior compatible with the
        // legacy NGram path. Backends without dual-buffer support use the
        // pre-existing checkpoint/rollback machinery.
        if num_accepted == k {
            self.checkpoint_ssm_states(seq)
        } else if num_accepted > 0 {
            self.rollback_ssm_states(seq, num_accepted)
        } else {
            Ok(())
        }
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

    /// Number of free KV cache blocks available for allocation.
    fn num_free_blocks(&self) -> usize {
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
