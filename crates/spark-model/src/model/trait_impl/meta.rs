// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn vocab_size_dispatch(&self) -> usize {
        self.config.vocab_size
    }

    pub(super) fn high_speed_swap_dims_dispatch(&self) -> Option<spark_storage::ModelDims> {
        // Only attention models have a meaningful sense of K/V blocks; SSM-
        // only models would need a different orchestrator. We expose dims
        // unconditionally and let the scheduler decide whether to install,
        // gated by the user's --high-speed-swap CLI choice.
        //
        // KV paging identity (ATLAS_KV_PAGING): the SAME config-derived
        // fingerprint the SSM tier uses (quant identity + geometry + the
        // ATLAS_MODEL_ID salt), via the KV convention (blob_bytes = 0).
        // Underivable ⇒ None with a loud warn; the flag-ON connect then fails
        // fast with an actionable error unless ATLAS_KV_PAGING_NS is set. Every
        // other path ignores the field (default-off ⇒ unread).
        let model_fp = match crate::model::ssm_tier::ModelFingerprint::derive_kv(&self.config) {
            Ok(fp) => Some(fp.nonzero()),
            Err(e) => {
                tracing::warn!(
                    "KV paging fingerprint underivable ({e:#}); ATLAS_KV_PAGING=1 \
                     will fail fast unless ATLAS_KV_PAGING_NS is set"
                );
                None
            }
        };
        Some(spark_storage::ModelDims {
            num_layers: self.config.num_hidden_layers as u32,
            max_blocks_per_layer: self.max_blocks_per_seq,
            num_q_heads: self.config.num_attention_heads as u16,
            num_kv_heads: self.config.num_key_value_heads as u16,
            head_dim: self.config.head_dim as u16,
            block_size: self.kv_cache.lock().block_size() as u16,
            model_fp,
        })
    }

    /// Storage dtype of this sequence's SSM h-state (`ATLAS_SSM_H_FP16`).
    ///
    /// Read from the sequence's own first SSM layer state, which the decode
    /// mixer flips on its first decode step — so this is the invariant itself,
    /// not an assumption about it. Every layer of a sequence converts on the
    /// same step, so the first is representative.
    pub(super) fn seq_ssm_h_is_f16(&self, seq: &SequenceState) -> bool {
        seq.layer_states
            .iter()
            .find_map(|ls| ls.as_any().downcast_ref::<SsmLayerState>())
            .is_some_and(|s| s.h_is_f16)
    }

    /// Narrow this sequence's SSM h-state to FP16 (`ATLAS_SSM_H_FP16`), once.
    ///
    /// ★ MUST be called from the model's decode entry points, OUTSIDE the CUDA
    /// graph region. Launching it from inside the layer puts the conversion
    /// into the captured graph, which then re-converts the already-FP16 state
    /// on every replay — fluent-but-degenerate output that the host-side
    /// `h_is_f16` flag cannot prevent, because the flag correctly says
    /// "converted" while the graph replays the launch anyway.
    ///
    /// No-op when the flag is absent or the sequence is already converted, so
    /// it is safe (and cheap) to call on every decode step.
    /// ★ The stream is NOT a parameter. Every decode entry point reassigns its
    /// working stream to `default_stream()` before compute, and the staging
    /// buffer below is a SINGLE shared allocation — so issuing the conversion
    /// on a caller-supplied stream both (a) leaves it unordered against the
    /// decode kernels that read the state and (b) lets two sequences'
    /// conversions interleave through the same scratch. That produced NaN
    /// h-states on a concurrency-dependent subset of sequences (7/16 at C=16,
    /// 6/128 at C=128; clean at C<=8), surfacing as `"In!!!!!!"` completions
    /// cut at 49 tokens. Taking the stream from the backend makes the whole
    /// sequence of conversions self-serialising and ordered ahead of decode.
    pub(crate) fn ssm_h_to_f16_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        let stream = self.gpu.default_stream();
        if !crate::layers::qwen3_ssm::ssm_h_fp16_enabled() || self.ssm_pool.num_ssm_layers == 0 {
            return Ok(());
        }
        let h_bytes = self.ssm_pool.h_bytes;
        let f16_bytes = h_bytes / 2;
        let mut pending = false;
        for ls in seq.layer_states.iter() {
            if let Some(s) = ls.as_any().downcast_ref::<SsmLayerState>()
                && !s.h_is_f16
            {
                pending = true;
                break;
            }
        }
        if !pending {
            return Ok(());
        }
        if self.ssm_h_f32_to_f16_kernel.0 == 0 {
            bail!(
                "ATLAS_SSM_H_FP16: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve on this                  target — refusing to run the FP16 decode kernels over an FP32 pool"
            );
        }
        let scratch = match self.ssm_h_f16_scratch.get() {
            Some(p) => *p,
            None => {
                let p = self.gpu.alloc(f16_bytes)?;
                let _ = self.ssm_h_f16_scratch.set(p);
                p
            }
        };
        for ls in seq.layer_states.iter_mut() {
            let Some(s) = ls.as_any_mut().downcast_mut::<SsmLayerState>() else {
                continue;
            };
            if s.h_is_f16 {
                continue;
            }
            crate::layers::ops::ssm_h_state_f32_to_f16(
                self.gpu.as_ref(),
                self.ssm_h_f32_to_f16_kernel,
                s.h_state,
                scratch,
                (h_bytes / 4) as u64,
                stream,
            )?;
            self.gpu
                .copy_d2d_async(scratch, s.h_state, f16_bytes, stream)?;
            s.h_is_f16 = true;
        }
        if std::env::var("ATLAS_SSM_H_FP16_DEBUG").is_ok() {
            tracing::info!(
                "SSM_H_FP16_CONVERT slot={} seq_len={} prompt_len={} hptr={:#x}",
                seq.slot_idx,
                seq.seq_len,
                seq.prompt_len,
                seq.layer_states
                    .iter()
                    .find_map(|l| l.as_any().downcast_ref::<SsmLayerState>())
                    .map(|x| x.h_state.0)
                    .unwrap_or(0)
            );
        }
        Ok(())
    }

    pub(super) fn normalize_ssm_states_dispatch(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;

        let num_ssm = self.ssm_pool.num_ssm_layers;
        if num_ssm == 0 || self.ssm_state_norm_kernel.0 == 0 {
            return Ok(());
        }
        // Stage 1 leaves prefill FP32 end-to-end and this normalize only ever
        // runs on a PREFILLING sequence (its one decode-path caller is gated on
        // `mamba_num_heads > 0`, which is 0 for GDN), so the FP16 arm is not
        // reached today. It is selected from the real invariant rather than
        // hardcoded so stage 2 — native FP16 prefill writes — does not have to
        // reopen this dispatch.
        let norm_k = if self.seq_ssm_h_is_f16(seq) {
            if self.ssm_state_norm_f16_kernel.0 == 0 {
                anyhow::bail!(
                    "ATLAS_SSM_H_FP16: ssm_state_norm::ssm_state_clamp_norm_fused_f16 did not                      resolve, refusing to clamp an FP16 state through the FP32 kernel"
                );
            }
            self.ssm_state_norm_f16_kernel
        } else {
            self.ssm_state_norm_kernel
        };
        let slot = seq.slot_idx;

        // Build pointer table: [layer_0_h_state, layer_1_h_state, ...]
        let ptrs: Vec<u64> = (0..num_ssm)
            .map(|i| self.ssm_pool.h_state(i, slot).0)
            .collect();
        // SAFETY: the length is derived from `ptrs` itself —
        // `ptrs.len() * size_of::<u64>()` — over the `Vec<u64>` the `collect`
        // above just materialised (`len == num_ssm`, every element written by
        // the map, no `with_capacity` gap).
        let ptr_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
        self.gpu
            .copy_h2d_async(ptr_bytes, self.ssm_norm_ptrs_buf, stream)?;

        let (num_heads, k_dim, v_dim) = self.config.ssm_state_norm_dims();

        KernelLaunch::new(self.gpu.as_ref(), norm_k)
            .grid([num_heads as u32, num_ssm as u32, 1])
            .block([v_dim as u32, 1, 1])
            .arg_ptr(self.ssm_norm_ptrs_buf)
            .arg_u32(num_heads as u32)
            .arg_u32(k_dim as u32)
            .arg_u32(v_dim as u32)
            .launch(stream)?;

        Ok(())
    }

    pub(super) fn bind_gpu_to_thread_dispatch(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    pub(super) fn alloc_sequence_dispatch(&self, budget_tokens: usize) -> Result<SequenceState> {
        // Claim via the RAII guard so the slot is returned to the pool on EVERY
        // sequence-exit path (normal finish, abort/cancel, decode error,
        // swap-out failure, panic). The explicit `free_sequence`/
        // `compact_sequence` paths neutralize the guard so release is
        // exactly-once; `slot_idx` derives from the guard, which is SSOT for
        // the owned index until an explicit path takes it.
        let slot_guard = self.ssm_pool.claim_guarded()?;
        let slot = slot_guard
            .idx()
            .expect("claim_guarded returns a guard owning a slot");
        // Zero SSM state: stale h_state/conv_state from a prior sequence would
        // corrupt the recurrence during prefill.
        // CRITICAL: use Atlas's own stream (not stream 0) because Atlas's stream
        // is CU_STREAM_NON_BLOCKING and does NOT synchronize with stream 0.
        // Using stream 0 would race with the subsequent prefill kernel.
        let stream = self.gpu.default_stream();
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        // Ensure zero completes before any prefill kernels touch this slot.
        self.gpu.synchronize(stream)?;
        let has_mtp = self.proposer.is_some() || self.self_speculative;

        // ATLAS_MTP_DRAFTER_PREFILL: a fresh sequence invalidates the
        // whole-prompt hidden capture — without this, a warm-restored prefill
        // (no chunks computed) would pair the NEW prompt's tokens with the
        // PREVIOUS sequence's captured hiddens in the drafter prefill.
        self.mtp_prefill_capture_len
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // ATLAS_MTP_CARRY_DRAFTER: the position-indexed hidden interval is
        // per-sequence by construction. Resetting it here is what makes the
        // carry path immune to the latent cross-sequence stale-hidden bug that
        // the legacy `captured >= prompt_len` guard still has: a warm-turn
        // append can only ever read rows THIS sequence's prefill wrote.
        *self.mtp_store_range.lock() = (0, 0);

        // Build layer states: SSM layers point into the pool (fixed addresses),
        // attention layers use their own alloc_state (EmptyLayerState).
        // When MTP is available, pre-allocate checkpoint + K=2 intermediate
        // buffers so CUDA graph capture doesn't trigger lazy allocation.
        let mut ssm_layer_idx = 0usize;
        let mut layer_states: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                // Layer-independent (one FP32 staging blob per SLOT), so it is
                // the same pointer for every SSM layer of this sequence.
                let stage = self.ssm_pool.h_prefill_stage(slot);
                let mut ssm_state = SsmLayerState {
                    h_state: self.ssm_pool.h_state(ssm_layer_idx, slot),
                    conv_state: self.ssm_pool.conv_state(ssm_layer_idx, slot),
                    h_state_checkpoint: None,
                    conv_state_checkpoint: None,
                    h_state_intermediates: Vec::new(),
                    conv_state_intermediates: Vec::new(),
                    // A freshly allocated slot has just been zeroed, and zero
                    // is zero in both formats. Which format it then HOLDS is
                    // decided by the pool width, not by the phase: under the
                    // stage-3 f16-SIZED pool the slot is physically 2
                    // bytes/element and prefill stages its FP32 work
                    // elsewhere, so the slot is FP16 from here on and the
                    // decode mixer has nothing to do. Under an FP32-sized
                    // pool prefill writes FP32 in place and FP32 is the truth
                    // until the mixer converts.
                    h_is_f16: stage.is_some(),
                    h_prefill_stage: stage,
                };

                if has_mtp {
                    // Use pool-based fixed addresses (stable across sequence
                    // lifetimes → CUDA graph can replay without stale pointers).
                    ssm_state.h_state_checkpoint =
                        Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, slot));
                    ssm_state.conv_state_checkpoint =
                        Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, slot));

                    // Tiered pools: H count is per-SLOT (h_inter_count),
                    // conv count is uniform. The vec lengths are the
                    // capacity gates every verify arm checks before writing.
                    for t in 0..self.ssm_pool.h_inter_count(slot) {
                        ssm_state
                            .h_state_intermediates
                            .push(self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t));
                    }
                    for t in 0..self.ssm_pool.num_intermediates {
                        ssm_state
                            .conv_state_intermediates
                            .push(self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t));
                    }
                }

                layer_states.push(Box::new(ssm_state));
                ssm_layer_idx += 1;
            } else {
                layer_states.push(layer.alloc_state(self.gpu.as_ref())?);
            }
        }

        // Zero SSM states for the new sequence.
        // Synchronous reset: memset + stream sync ensures zero is visible
        // before any subsequent kernel reads the state.
        self.ssm_pool.reset_slot(slot, self.gpu.as_ref())?;
        // Double-check: explicit sync to guarantee zero is complete
        self.gpu.synchronize(self.gpu.default_stream())?;

        // Allocate MTP proposer state (owns its own KV cache block table)
        let proposer_state = match &self.proposer {
            Some(p) => Some(p.alloc_state_for(self.gpu.as_ref(), budget_tokens)?),
            None => None,
        };

        // No graph invalidation needed — pool addresses are stable across sequences.

        // Phase 6.1.d critical fix: pre-size disk_last_offloaded_per_layer
        // to the model's attention-layer count. The vector stays empty
        // when HSS isn't engaged (cache_blocks_per_seq is None) — the
        // helper short-circuits before reading from it. Sized here once
        // so the layer-0 offload helper doesn't need to grow a Vec on
        // every sequence's first decode step.
        let num_attn_layers = self.config.num_attention_layers();
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,          // default: defer to installed active adapter
            acquired_adapter_slot: -1, // Task #25: no ref held until prefill acquires
            src_lang_id: 0,            // NLLB-only per-request lang (0 = deployment default)
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states,
            proposer_state,
            slot_idx: slot,
            ssm_slot: Some(slot_guard),
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            mtp_capture_gen: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            cached_prefix_blocks: 0,
            prefix_ref_tokens: Vec::new(),
            prefix_lookup_applied: false,
            prefix_lookup_skip: false,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![0; num_attn_layers],
        })
    }

    pub(super) fn copy_logits_to_host_dispatch(
        &self,
        logits_ptr: DevicePtr,
        dst: &mut [u8],
    ) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    pub(super) fn logits_ptr_is_fp32_dispatch(&self, logits_ptr: DevicePtr) -> bool {
        self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0
    }

    pub(super) fn logits_buffer_ptr_dispatch(&self) -> DevicePtr {
        self.buffers.logits()
    }

    pub(super) fn argmax_on_device_dispatch(
        &self,
        logits_ptr: DevicePtr,
        _stream: u64,
    ) -> Result<u32> {
        // Use backend's default stream (same as decode) to avoid implicit
        // sync overhead from legacy default stream (handle 0).
        let stream = self.gpu.default_stream();
        // Use first 4 bytes of scratch buffer for the u32 output
        let out_ptr = self.buffers.scratch();
        // Dispatch by buffer dtype: when the logits pointer is the model's
        // FP32 scratch (single-token decode lm_head with use_fp32_logits),
        // run argmax_fp32; otherwise the buffer is BF16 (prefill /
        // batched-decode / non-Gemma-4 paths) and argmax_bf16 applies.
        // The kernel arg layout is identical (ptr, ptr, u32), so dispatch
        // is just a kernel-handle swap.
        let is_fp32 = self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0;
        let kernel = if is_fp32 {
            self.argmax_logits_kernel
        } else {
            self.argmax_kernel
        };
        ops::argmax_bf16(
            self.gpu.as_ref(),
            kernel,
            logits_ptr,
            out_ptr,
            self.config.vocab_size as u32,
            stream,
        )?;
        // D2H: copy 4 bytes (single u32) instead of vocab_size*2 = 304KB
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let gpu_token = u32::from_le_bytes(buf);

        Ok(gpu_token)
    }

    pub(super) fn argmax_batch_dispatch(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let out_ptr = self.buffers.scratch();
        // ONE launch, one block per row. The single-row `argmax_bf16` is a one-CTA
        // reduction (grid [1,1,1]), so n calls on the same stream serialise n
        // single-SM scans: measured 16 x 100.6 us = 1.6 ms per decode step at n=16.
        // The batched kernel runs the identical per-row body, so ties resolve the
        // same way — byte-identical. Falls back to the loop when the kernel set
        // lacks the batched entry.
        fn argmax_batch_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("ATLAS_NO_ARGMAX_BATCH").ok().as_deref() != Some("1"))
        }
        if self.argmax_batch_kernel.0 != 0 && argmax_batch_enabled() {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                logits_ptr,
                out_ptr,
                v as u32,
                n as u32,
                v as u32,
                stream,
            )?;
        } else {
            for i in 0..n {
                let logits_i = logits_ptr.offset(i * v * bf16);
                let out_i = out_ptr.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    v as u32,
                    stream,
                )?;
            }
        }
        let mut buf = vec![0u8; n * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(u32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ]));
        }
        Ok(results)
    }

    /// `norm_output()` holds the post-final-norm hidden state from the last decode.
    pub(super) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        self.buffers.norm_output()
    }
}
