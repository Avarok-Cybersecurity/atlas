// SPDX-License-Identifier: AGPL-3.0-only

//! Shared prefill-path `Model` stub for the scheduler tests.
//!
//! Lifted out of `prefill_fifo_tests.rs` when a second test file needed the
//! same ~150-line `Model` impl. Copying it would have satisfied the file-size
//! cap and broken SSOT: two stubs drifting apart is how a harness starts
//! testing something the engine does not do.
//!
//! `test_support.rs` is the home for scheduler fixtures generally; this stub
//! lives beside it rather than inside it only because that file is already at
//! the 500-LoC cap.

use anyhow::Result;
use spark_model::traits::{BatchedPrefillDeclined, Model, PrefillSlice, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The sampled first token. Not an EOS in the fixtures that use it, so
/// promotion pushes onto `active` rather than finishing.
pub(super) const FIRST: u32 = 7;

/// A NON-NULL sentinel logits pointer. It is never dereferenced — the greedy
/// fast path (temperature 0, no suppressed ids) answers from
/// `argmax_on_device`, which this stub scripts. It must not be `NULL` because
/// the batched paths treat NULL-on-a-last-chunk as "the model returned no
/// logits" and fail the stream, which is a real contract worth keeping: a stub
/// returning NULL would make every batched test look like a dropped request.
const LOGITS: DevicePtr = DevicePtr(0x1000);

/// How the stub answers `prefill_batch_chunk` for a batch of 2+ streams.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchedBehaviour {
    /// Run the trait's default per-stream loop (what a model with no batched
    /// path of its own does).
    #[default]
    PerStream,
    /// Refuse with a `BatchedPrefillDeclined` — the model looked at the wave
    /// and said no BEFORE touching a stream. The scheduler must re-run the
    /// wave one stream at a time and complete every request.
    Decline,
    /// Refuse with a plain error — the batch was admitted and something failed
    /// with state already committed. The scheduler must NOT retry, and every
    /// affected request must still receive an error rather than a closed
    /// channel.
    HardError,
}

/// Minimal `Model`: every prefill chunk succeeds and the greedy sampler
/// (temperature 0.0, no suppressed ids) answers from `argmax_on_device`.
/// Everything else is unreachable on the prefill paths this harness drives.
///
/// `batched` scripts the N>=2 batched-prefill answer so the scheduler's
/// decline/failure handling can be driven without a GPU; `single_calls`
/// counts the N==1 dispatches, which is how a test proves the per-stream
/// fallback actually ran rather than the wave silently succeeding.
#[derive(Default)]
pub(super) struct PrefillStubModel {
    pub(super) batched: BatchedBehaviour,
    pub(super) single_calls: AtomicUsize,
    pub(super) batched_calls: AtomicUsize,
}

impl PrefillStubModel {
    pub(super) fn with_batched(batched: BatchedBehaviour) -> Self {
        Self {
            batched,
            ..Default::default()
        }
    }
}

impl Model for PrefillStubModel {
    /// The only override this harness needs: mirror the real dispatcher's
    /// contract that an N==1 call is the single-stream path, and script the
    /// N>=2 answer.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        if streams.len() <= 1 {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            self.batched_calls.fetch_add(1, Ordering::Relaxed);
            match self.batched {
                BatchedBehaviour::PerStream => {}
                BatchedBehaviour::Decline => {
                    return Err(anyhow::Error::new(BatchedPrefillDeclined::new(
                        "scripted decline (test)",
                    )));
                }
                BatchedBehaviour::HardError => {
                    anyhow::bail!("scripted hard batched-prefill failure (test)");
                }
            }
        }
        let mut out = Vec::with_capacity(streams.len());
        for slice in streams.iter_mut() {
            out.push(self.prefill_chunk(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                slice.is_last_chunk,
                stream,
            )?);
        }
        Ok(out)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        _is_last: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        // Mirror the real contract: the chunk's tokens land in the sequence.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = seq.tokens.len();
        Ok(LOGITS)
    }
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        Ok(FIRST)
    }
    fn vocab_size(&self) -> usize {
        32
    }
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _l: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        unreachable!("greedy fast path never reads logits back")
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("chunked prefill only")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
    fn argmax_batch(&self, _l: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        unreachable!("batched decode is not driven here")
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
}
