// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for NllbGpuModel`. The model owns all KV, returns bf16 logits
//! (the scheduler's default sampling path), and drives the encoder inside
//! `prefill` (seeding the decoder with `[decoder_start, forced_bos]` so the
//! forced target-language token stays out of the sampled stream). Speculative /
//! SSM / MTP / graph paths are not applicable and are stubbed.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::NllbGpuModel;
use super::kv::NllbSeqKv;
use crate::traits::{Model, SequenceState};

impl NllbGpuModel {
    /// Encoder pass + decoder seed, shared by `prefill` and `prefill_chunk`.
    fn prefill_impl(&self, tokens: &[u32], seq: &mut SequenceState) -> Result<DevicePtr> {
        let slot = seq.slot_idx;
        // Per-request LoRA: apply the adapter only when the request selected it.
        self.set_lora_active(seq.adapter_slot);
        // Per-request source/target language (0 → deployment default).
        let src_id = if seq.src_lang_id != 0 {
            seq.src_lang_id
        } else {
            self.lang.src_lang_id
        };
        let tgt_id = if seq.tgt_lang_id != 0 {
            seq.tgt_lang_id
        } else {
            self.lang.tgt_lang_id
        };
        let src = self.lang.encoder_input_with(src_id, tokens);
        {
            let mut map = self.kv.lock().unwrap();
            let kv = map
                .get_mut(&slot)
                .context("nllb prefill: no KV state for slot (alloc_sequence not called?)")?;
            self.run_encoder(&src, kv)?;
            // Seed the decoder: step 0 consumes `decoder_start` (logits ignored),
            // step 1 consumes `forced_bos = tgt_lang` and yields logits for the
            // FIRST real translation token — so the forced token never enters the
            // sampled stream and the scheduler samples normally from here. Each
            // concurrent prefill writes its OWN row (by slot) so they don't race.
            let row = self.prefill_logits_row(slot);
            self.forward_one(self.lang.decoder_start_id, kv, row)?;
            self.forward_one(tgt_id, kv, row)?;
        }
        self.sync()?;
        // Decoder-side bookkeeping (the scheduler appends generated tokens and
        // uses seq_len for its accounting; NLLB's own KV drives the real compute).
        seq.tokens = vec![self.lang.decoder_start_id, tgt_id];
        seq.seq_len = seq.tokens.len();
        seq.prompt_len = seq.tokens.len();
        Ok(self.prefill_logits_row(slot))
    }
}

impl Model for NllbGpuModel {
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.prefill_impl(tokens, seq)
    }

    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        if chunk_start != 0 || !is_last_chunk || chunk_len != tokens.len() {
            bail!("nllb: chunked prefill unsupported — the source must arrive in one chunk");
        }
        self.prefill_impl(tokens, seq)
    }

    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        let slot = seq.slot_idx;
        self.set_lora_active(seq.adapter_slot);
        let row = self.decode_logits_row(0);
        {
            let mut map = self.kv.lock().unwrap();
            let kv = map
                .get_mut(&slot)
                .context("nllb decode: no KV state for slot")?;
            self.forward_one(token, kv, row)?;
        }
        self.sync()?;
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(row)
    }

    /// Batched decode: one `forward_one` per sequence into CONTIGUOUS logit rows
    /// `0..n` (batch position `i` ↔ `seqs[i]`), the scheduler's row contract.
    /// Each sequence's own per-slot KV is looked up by `slot_idx`, so batch
    /// order is irrelevant. Sequences are processed serially on the default
    /// stream (shared decode scratch); the returned base pointer is `[n, vocab]`.
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if n == 0 || tokens.len() != n {
            bail!(
                "nllb decode_batch: tokens/seqs length mismatch ({}, {n})",
                tokens.len()
            );
        }
        if n > self.max_batch {
            bail!(
                "nllb decode_batch: n={n} exceeds max_batch={}",
                self.max_batch
            );
        }
        {
            let mut map = self.kv.lock().unwrap();
            for (i, seq) in seqs.iter().enumerate() {
                // Per-request LoRA gate, re-armed for each sequence in the batch.
                self.set_lora_active(seq.adapter_slot);
                let kv = map
                    .get_mut(&seq.slot_idx)
                    .context("nllb decode_batch: no KV state for slot")?;
                self.forward_one(tokens[i], kv, self.decode_logits_row(i))?;
            }
        }
        self.sync()?;
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[i]);
            seq.seq_len += 1;
        }
        Ok(self.decode_logits_row(0))
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn supports_beam(&self) -> bool {
        true
    }

    fn generate_beam_batch(&self, reqs: &[crate::traits::BeamReq]) -> Result<Vec<Vec<u32>>> {
        // Phase c: the C requests' Σ beams decode as ONE M=(Σ beams) batch per
        // step (cross-request co-dispatch). Single-adapter batches fuse; mixed
        // adapters fall back to serial inside `beam_batched_multi`.
        self.beam_batched_multi(reqs)
    }

    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    fn alloc_sequence(&self) -> Result<SequenceState> {
        let slot = self.slots.lock().unwrap().claim();
        let kv = NllbSeqKv::new(self.gpu.as_ref(), self.dec_layers, self.cache_rows, self.d)?;
        self.kv.lock().unwrap().insert(slot, kv);
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states: Vec::new(),
            proposer_state: None,
            slot_idx: slot,
            ssm_slot: None,
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: Vec::new(),
        })
    }

    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        let slot = seq.slot_idx;
        if slot == usize::MAX {
            return Ok(()); // migrated to a survivor by compact_sequence
        }
        if let Some(kv) = self.kv.lock().unwrap().remove(&slot) {
            kv.free(self.gpu.as_ref())?;
        }
        self.slots.lock().unwrap().release(slot);
        Ok(())
    }

    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        let old = seq.slot_idx;
        let mut map = self.kv.lock().unwrap();
        if let Some(kv) = map.remove(&old) {
            map.insert(new_slot, kv);
        }
        seq.slot_idx = new_slot;
        Ok(())
    }

    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        seq.slot_idx = usize::MAX;
    }

    fn cache_sequence(&self, _seq: &SequenceState) {
        // No prefix reuse for translation.
    }

    fn num_free_blocks(&self) -> usize {
        // The model owns its KV outside the paged block cache; report ample
        // headroom so the scheduler's swap/admission math never rejects.
        1 << 20
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.decode_logits_row(0)
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_of(logits_ptr)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        // `logits_ptr` is `[n, vocab]` bf16; argmax each contiguous row.
        (0..n)
            .map(|i| self.argmax_of(logits_ptr.offset(i * self.vocab * 2)))
            .collect()
    }

    // ── inapplicable paths: NLLB is non-speculative / non-SSM / non-MTP ──

    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr(0)
    }

    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        bail!("nllb: speculative verify not supported")
    }

    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }

    fn generate_speculative(
        &self,
        _tokens: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        bail!("nllb: speculative decoding not supported")
    }

    fn has_proposer(&self) -> bool {
        false
    }

    fn has_self_speculative(&self) -> bool {
        false
    }

    fn decode_draft(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        bail!("nllb: self-speculative draft not supported")
    }

    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        bail!("nllb: graphed verify not supported")
    }

    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        bail!("nllb: graphed verify not supported")
    }

    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        bail!("nllb: graphed verify not supported")
    }

    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn run_mtp_propose(
        &self,
        _token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn trim_proposer_state(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}
