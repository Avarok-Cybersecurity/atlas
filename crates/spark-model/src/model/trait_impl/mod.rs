// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for TransformerModel` — thin trait impl that delegates to
//! `<method>_dispatch` helpers split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `TransformerModel`
//! inherent impl. The trait impl below is purely one-line delegators.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{AttnMetadataDev, LayerState};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights};

mod async_chkpt;
mod decode_a;
mod decode_a2;
mod decode_b;
mod decode_b2;
mod ep_misc;
mod meta;
mod prefill_a;
mod prefill_b;
mod prefill_c;
mod prefill_d;
mod sequence;
mod speculative;
mod verify_a;
mod verify_b;
mod verify_c;
mod verify_c2;
mod verify_d;

impl Model for TransformerModel {
    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        self.prepare_vision_embed_dispatch(images)
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.prefill_dispatch(tokens, seq, stream)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream)
    }
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.decode_dispatch(token, seq, _stream)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.decode_batch_dispatch(tokens, seqs, stream)
    }
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
    ) -> Result<crate::traits::MixedForwardResult> {
        self.mixed_forward_dispatch(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size_dispatch()
    }
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        self.high_speed_swap_dims_dispatch()
    }
    fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        self.normalize_ssm_states_dispatch(seq, stream)
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_gpu_to_thread_dispatch()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.alloc_sequence_dispatch()
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits_to_host_dispatch(logits_ptr, dst)
    }
    fn logits_ptr_is_fp32(&self, logits_ptr: DevicePtr) -> bool {
        self.logits_ptr_is_fp32_dispatch(logits_ptr)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr_dispatch()
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_on_device_dispatch(logits_ptr, _stream)
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        self.argmax_batch_dispatch(logits_ptr, n, _stream)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        self.hidden_after_norm_dispatch()
    }
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_dispatch(tokens, seq, stream)
    }
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        self.checkpoint_ssm_states_dispatch(seq)
    }
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        self.rollback_ssm_states_dispatch(seq, num_accepted)
    }
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        self.generate_speculative_dispatch(prompt_tokens, params, num_drafts)
    }
    fn has_proposer(&self) -> bool {
        self.has_proposer_dispatch()
    }
    fn has_self_speculative(&self) -> bool {
        self.has_self_speculative_dispatch()
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.decode_draft_dispatch(token, seq, stream)
    }
    fn cache_sequence(&self, seq: &SequenceState) {
        self.cache_sequence_dispatch(seq)
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.free_sequence_dispatch(seq)
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        self.decode_verify_graphed_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        self.decode_verify_graphed_k3_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        self.decode_verify_graphed_k4_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream)
    }
    fn save_hidden_for_mtp(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_hidden_for_mtp_dispatch(token_idx, _stream)
    }
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        self.run_mtp_propose_dispatch(token, position, seq, _stream)
    }
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi_dispatch(
            token,
            position,
            num_drafts,
            seq,
            _stream,
            grammar_bitmask,
        )
    }
    fn read_deferred_draft_token(&self) -> Result<u32> {
        self.read_deferred_draft_token_dispatch()
    }
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        self.trim_proposer_state_dispatch(seq, num_accepted, _stream)
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        self.compact_sequence_dispatch(seq, new_slot)
    }
    fn save_sequence_state(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        self.save_sequence_state_dispatch(seq, writer)
    }
    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.restore_sequence_state_dispatch(seq, num_blocks, reader)
    }
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks_dispatch()
    }
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.start_checkpoint_async_dispatch(seq)
    }
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        self.start_rollback_and_checkpoint_async_dispatch(seq, num_accepted)
    }
    fn sync_secondary(&self) -> Result<()> {
        self.sync_secondary_dispatch()
    }
    fn pre_verify_copy_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.pre_verify_copy_async_dispatch(seq)
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.commit_verify_state_async_dispatch(seq, num_accepted, k)
    }
    fn ep_worker_step(&self, seq: &mut SequenceState) -> Result<bool> {
        self.ep_worker_step_dispatch(seq)
    }
    fn is_ep(&self) -> bool {
        self.is_ep_dispatch()
    }
    fn is_mla(&self) -> bool {
        self.is_mla_dispatch()
    }
    fn decode_logits_fp32(&self) -> bool {
        self.decode_logits_fp32_dispatch()
    }
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.decode_logits_ptr_dispatch()
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd_dispatch(cmd)
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.ep_broadcast_tokens_dispatch(tokens)
    }
    fn default_stream(&self) -> u64 {
        self.default_stream_dispatch()
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_dispatch()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_dispatch()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_dispatch(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_dispatch(stream, event)
    }
}
