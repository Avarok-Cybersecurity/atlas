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
        Some(spark_storage::ModelDims {
            num_layers: self.config.num_hidden_layers as u32,
            max_blocks_per_layer: self.max_blocks_per_seq,
            num_q_heads: self.config.num_attention_heads as u16,
            num_kv_heads: self.config.num_key_value_heads as u16,
            head_dim: self.config.head_dim as u16,
            block_size: self.kv_cache.lock().block_size() as u16,
        })
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
        let slot = seq.slot_idx;

        // Build pointer table: [layer_0_h_state, layer_1_h_state, ...]
        let ptrs: Vec<u64> = (0..num_ssm)
            .map(|i| self.ssm_pool.h_state(i, slot).0)
            .collect();
        let ptr_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
        self.gpu
            .copy_h2d_async(ptr_bytes, self.ssm_norm_ptrs_buf, stream)?;

        let (num_heads, k_dim, v_dim) = self.config.ssm_state_norm_dims();

        KernelLaunch::new(self.gpu.as_ref(), self.ssm_state_norm_kernel)
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

    pub(super) fn alloc_sequence_dispatch(&self) -> Result<SequenceState> {
        // Claim via the RAII guard so the slot is returned to the pool on EVERY
        // sequence-exit path (normal finish, abort/cancel, decode error,
        // swap-out failure, panic). The explicit `free_sequence`/
        // `compact_sequence` paths neutralize the guard so release is
        // exactly-once. `slot_idx` is derived from the guard (SSOT for the
        // owned index lives in the guard until an explicit path takes it).
        let slot_guard = self.ssm_pool.claim_guarded()?;
        let slot = slot_guard
            .idx()
            .expect("claim_guarded returns a guard owning a slot");
        // Zero SSM state to prevent stale h_state/conv_state from prior
        // sequences corrupting the recurrent computation during prefill.
        // CRITICAL: use Atlas's own stream (not stream 0) because Atlas's stream
        // is CU_STREAM_NON_BLOCKING and does NOT synchronize with stream 0.
        // Using stream 0 would race with the subsequent prefill kernel.
        let stream = self.gpu.default_stream();
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        // Ensure zero completes before any prefill kernels touch this slot.
        self.gpu.synchronize(stream)?;
        let has_mtp = self.proposer.is_some() || self.self_speculative;

        // Build layer states: SSM layers point into the pool (fixed addresses),
        // attention layers use their own alloc_state (EmptyLayerState).
        // When MTP is available, pre-allocate checkpoint + K=2 intermediate
        // buffers so CUDA graph capture doesn't trigger lazy allocation.
        let mut ssm_layer_idx = 0usize;
        let mut layer_states: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let mut ssm_state = SsmLayerState {
                    h_state: self.ssm_pool.h_state(ssm_layer_idx, slot),
                    conv_state: self.ssm_pool.conv_state(ssm_layer_idx, slot),
                    h_state_checkpoint: None,
                    conv_state_checkpoint: None,
                    h_state_intermediates: Vec::new(),
                    conv_state_intermediates: Vec::new(),
                };

                if has_mtp {
                    // Use pool-based fixed addresses (stable across sequence
                    // lifetimes → CUDA graph can replay without stale pointers).
                    ssm_state.h_state_checkpoint =
                        Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, slot));
                    ssm_state.conv_state_checkpoint =
                        Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, slot));

                    for t in 0..self.ssm_pool.num_intermediates {
                        ssm_state
                            .h_state_intermediates
                            .push(self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t));
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
            Some(p) => Some(p.alloc_state(self.gpu.as_ref())?),
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
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
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

    pub(super) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        // norm_output() holds the post-final-norm hidden state from the last decode
        self.buffers.norm_output()
    }
}
