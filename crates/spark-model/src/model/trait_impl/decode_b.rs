// SPDX-License-Identifier: AGPL-3.0-only
//! Decode phase B — batched multi-sequence decode.
//!
//! Same POD-array-to-byte-slice `unsafe` pattern as `verify_c.rs`; see
//! that file's module docs for the full safety contract.

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

mod build_states;

impl TransformerModel {
    pub(super) fn mixed_forward_dispatch(
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
        let n_decode = decode_tokens.len();
        let n_prefill = prefill_chunk_len;

        // Padded decode count for batched decode kernel compatibility
        let padded_n_guard = [2usize, 4, 8]
            .iter()
            .copied()
            .find(|&s| s >= n_decode)
            .unwrap_or(n_decode);

        // Guard: fall back to default (sequential) for EP, oversized, no decode,
        // or MLA. MLA models route the decode portion through `decode_batch`,
        // whose `decode_batch_dispatch` dispatches the batched MLA branch
        // (`ms_mla_decode`, issue #84). The fused `decode_multi_seq` body
        // inlined below is NOT used for MLA here — it shares a single layer
        // loop with the prefill chunk and that interleaving has not been
        // validated for the absorbed-MLA path — so MLA stays on the
        // dedicated `decode_batch` route.
        // Use padded_n (not n_decode) because padding slots consume hidden buffer space.
        if self.comm.is_some()
            || self.is_mla_dispatch()
            || (padded_n_guard + n_prefill) > self.buffers.max_batch_tokens()
            || n_decode == 0
        {
            let decode_logits = if !decode_tokens.is_empty() {
                self.decode_batch(decode_tokens, decode_seqs, stream)?
            } else {
                DevicePtr::NULL
            };
            let prefill_logits = self.prefill_chunk(
                prefill_tokens,
                prefill_seq,
                prefill_chunk_start,
                prefill_chunk_len,
                prefill_is_last,
                stream,
            )?;
            return Ok(crate::traits::MixedForwardResult {
                decode_logits,
                prefill_logits,
            });
        }

        // ── Fused mixed forward: single layer loop, weights loaded once per layer ──
        //
        // Layout in hidden/residual buffers (contiguous):
        //   [0 .. N*H*fp32)           = decode tokens (N sequences × 1 token each)
        //   [N*H*fp32 .. (N+M)*H*fp32) = prefill chunk tokens (1 sequence × M tokens)
        //
        // Per layer: decode_multi_seq on [0..N), then prefill on [N..N+M).
        // Both use non-overlapping hidden/residual regions. Intermediate scratch
        // buffers (norm_output, qkv_output, etc.) are overwritten by each sub-call,
        // safe because same CUDA stream guarantees sequential execution.

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad decode count to nearest [2, 4, 8] for batched decode kernel compat
        let padded_n = [2usize, 4, 8]
            .iter()
            .copied()
            .find(|&s| s >= n_decode)
            .unwrap_or(n_decode);

        // ── 1. Embed all tokens contiguously ──

        // 1a. Decode tokens → hidden[0..n_decode*H)
        for (i, &tok) in decode_tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(i * h * fp32), stream)?;
        }
        // 1b. Zero padding for decode [n_decode..padded_n)
        for i in n_decode..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }
        // 1c. Prefill chunk tokens → hidden[padded_n*H..(padded_n+M)*H)
        //     Use batched embed for efficiency (single kernel launch for M tokens)
        let prefill_hidden = hidden.offset(padded_n * h * fp32);
        let prefill_residual = residual.offset(padded_n * h * fp32);
        {
            let chunk_tokens =
                &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill];
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk_tokens.as_ptr() as *const u8, n_prefill * 4)
            };
            // Use norm_output as temporary staging for token IDs (overwritten by first layer)
            let token_ids_dev = self.buffers.norm_output();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                prefill_hidden,
                n_prefill as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(prefill_hidden, n_prefill, stream)?;
        }

        // ── 2. Lock KV cache once for both decode and prefill ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();

        // 2a. Allocate KV blocks for decode sequences
        for seq in decode_seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 2b. Allocate KV blocks for prefill sequence
        let prefill_end_pos = prefill_chunk_start + n_prefill;
        let prefill_blocks_needed = (prefill_end_pos - 1) / bs + 1;
        ensure_blocks_through_prefill(
            prefill_seq,
            prefill_blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        // ── 3. Upload decode metadata ──
        //
        // Place decode metadata in the logits buffer (not used until step 7).
        // This avoids conflicts with prefill MoE routing scratch at scratch[0..].
        // Decode metadata is small (padded_n ≤ 8, ~33KB max) and the logits buffer
        // is large (16 * vocab * 2 bytes ≈ 4.8MB). The logits are overwritten in
        // step 7 after the layer loop completes.
        //
        // BUG FIX 2026-05-10: offset by 64KB to avoid being overwritten by MoE
        // forward's `shared_gate_scratch` which also uses `logits` as scratch
        // (moe/forward.rs:211, forward_batched.rs:61, forward_k2.rs:91, etc.).
        // Without this offset, the first MoE call during the layer loop
        // overwrites decode_meta's positions/slots/seq_lens/block_table at
        // logits[0..16K], causing subsequent attention kernels to read
        // corrupted block_table → CUDA-700 illegal memory access. Reproducer:
        // Qwen3-Next-80B + 2 streams + chunked prefill, when one finishes
        // first and `mixed_forward` runs decode+prefill fused. 64KB offset
        // leaves room for the largest known shared-expert scratch
        // (shared_expert_intermediate_size × 2 ≤ 32KB observed for any
        // current Atlas model — 64KB is 2× safety margin).
        let decode_meta_base = self.buffers.logits().offset(65536);

        let decode_metadata = self.upload_batch_metadata_at(
            decode_seqs,
            padded_n,
            &mut kv_cache,
            decode_meta_base,
            stream,
        )?;

        // ── 4. Upload prefill metadata ──
        //
        // Prefill metadata at scratch[moe_scratch..], same layout as prefill_chunk.
        let proc_start = prefill_chunk_start;
        let proc_count = n_prefill;
        let effective_seq_len_start = prefill_chunk_start;
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let prefill_meta_base = self.buffers.scratch().offset(meta_offset);
        let slot_offset = (proc_count * 4 + 7) & !7;
        let needs_paged = effective_seq_len_start > 0;

        {
            // SAFETY: Single-threaded scheduler access.
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            let pinned = stg.ptr;
            let mut cursor = proc_count * 4;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.positions.as_ptr() as *const u8,
                    pinned,
                    proc_count * 4,
                );
            }

            if !needs_paged {
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = prefill_seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
                cursor = slot_offset;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stg.slots.as_ptr() as *const u8,
                        pinned.add(cursor),
                        proc_count * 8,
                    );
                }
                cursor += proc_count * 8;
            }

            assert!(
                cursor <= stg.bytes,
                "mixed_forward prefill metadata overflow: {cursor} > {}",
                stg.bytes
            );
            let pinned_slice = unsafe { std::slice::from_raw_parts(pinned, cursor) };
            self.gpu
                .copy_h2d_async(pinned_slice, prefill_meta_base, stream)?;
        }

        if needs_paged {
            let current_blocks = prefill_seq.block_table.len();
            let upload_start = self
                .ensure_chunked_prefill_meta(prefill_seq, prefill_tokens.len(), bs)?
                .uploaded_blocks;
            // Phase 6.3: skip upload in HSS mode (orchestrator bypasses kernel).
            if upload_start < current_blocks && prefill_seq.hss_window_start() == 0 {
                let new_blocks = &prefill_seq.block_table[upload_start..];
                let bt_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(new_blocks),
                    )
                };
                let block_table_base = prefill_seq
                    .chunked_prefill_meta
                    .as_ref()
                    .unwrap()
                    .block_table;
                self.gpu.copy_h2d_async(
                    bt_bytes,
                    block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                    stream,
                )?;
                prefill_seq
                    .chunked_prefill_meta
                    .as_mut()
                    .unwrap()
                    .uploaded_blocks = current_blocks;
            }

            let seq_len_val = (proc_start + proc_count) as u32;
            let seq_len_bytes = unsafe {
                std::slice::from_raw_parts(
                    &seq_len_val as *const u32 as *const u8,
                    std::mem::size_of::<u32>(),
                )
            };
            let seq_len_base = prefill_seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
            self.gpu
                .copy_h2d_async(seq_len_bytes, seq_len_base, stream)?;

            let block_table_base = prefill_seq
                .chunked_prefill_meta
                .as_ref()
                .unwrap()
                .block_table;
            ops::fill_slots_from_block_table(
                self.gpu.as_ref(),
                self.fill_slots_kernel,
                prefill_meta_base.offset(slot_offset),
                block_table_base,
                proc_start as u32,
                proc_count as u32,
                bs as u32,
                stream,
            )?;
        }

        // Force H2D metadata copies to complete before layer forward.
        self.gpu.synchronize(stream)?;

        let (prefill_bt_dev, prefill_sl_dev) = if needs_paged {
            let page_meta = prefill_seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        // Request-scoped LoRA routing for the fused (SLAI) prefill portion. The
        // decode portion already routes via `upload_batch_metadata_at` (its own
        // +128 gap); the prefilling sequence uses the dedicated `lora_seq_slot`
        // arena buffer (`proc_count` uniform slots), so the two never collide.
        // Without this, a prefilling request co-scheduled with decodes would
        // still contaminate its prompt KV with the global active adapter.
        let prefill_seq_slot = self.upload_seq_slot_uniform(
            prefill_seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;
        let prefill_metadata = AttnMetadataDev {
            positions: prefill_meta_base,
            positions_h: prefill_meta_base,
            positions_w: prefill_meta_base,
            slot: prefill_meta_base.offset(slot_offset),
            seq_len: prefill_sl_dev,
            block_table: prefill_bt_dev,
            max_blocks_per_seq: prefill_seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: prefill_seq_slot,
        };

        // ── 5. Build decode layer states ──
        let (seq_lens, block_tables, mut all_layer_states) =
            self.mixed_build_decode_layer_states(decode_seqs, padded_n, n_decode)?;

        // ── 6. Fused layer loop ──
        //
        // For each layer: process decode portion then prefill portion.
        // Weights are loaded once by the first sub-call and remain in L2
        // cache for the second sub-call. This halves memory bandwidth vs
        // the sequential decode_batch + prefill_chunk approach.
        let decode_ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(decode_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            // #30: decode never routes prefill — installed-pair/bgmv path only.
            routed_lora_layers: None,
        };

        let prefill_ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(prefill_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            // #30: the fused (SLAI) prefill portion routes by the prefilling
            // seq's slot (None unless it routes to a non-active slot).
            routed_lora_layers: self.routed_slot_layers(prefill_seq.adapter_slot),
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 6a. Decode: N sequences × 1 token each on hidden[0..padded_n*H)
            let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
            layer.decode_multi_seq(
                hidden,
                residual,
                padded_n,
                &mut layer_state_refs,
                &mut kv_cache,
                &seq_lens,
                &block_tables,
                &decode_ctx,
                stream,
            )?;

            // 6b. Prefill: 1 sequence × M tokens on hidden[padded_n*H..]
            layer.prefill(
                prefill_hidden,
                prefill_residual,
                proc_count,
                prefill_seq.layer_states[layer_idx].as_mut(),
                &mut kv_cache,
                effective_seq_len_start,
                &mut prefill_seq.block_table,
                &mut prefill_seq.disk_block_ids,
                &mut prefill_seq.disk_last_offloaded_per_layer,
                0, // kv_write_start: no prefix cache skip in fused path
                &prefill_ctx,
                stream,
            )?;
        }

        // ── Step 0 (spec blocker B1): per-chunk SSM state normalize ──
        //
        // Normalize the prefill seq's h_state on the SAME `stream`
        // (= default_stream, reassigned near the top of this fn) that the
        // GDN recurrence just wrote it on — in-order, no event, no race.
        // This MUST cover EVERY mixed chunk INCLUDING the last: mixed_forward
        // runs the GDN write on default_stream, so the terminal normalize
        // also belongs here. Leaving the is_last normalize in run_standard.rs
        // on prefill_stream (as the original Step 0 did) does NOT order these
        // default_stream writes → the final chunk reads a stale state →
        // nondeterministic corruption (the residual B1 race that failed
        // token-for-token validation, 0/12). The standard prefill_chunk path
        // keeps its own same-stream (prefill_stream) every-chunk normalize.
        self.normalize_ssm_states_dispatch(prefill_seq, stream)?;

        // Restore decode layer_states to sequences
        for (seq, ls) in decode_seqs
            .iter_mut()
            .zip(all_layer_states.drain(..n_decode))
        {
            seq.layer_states = ls;
        }

        // ── 7. Final norm + LM head ──
        let head_out = self.mixed_final_norm_lm_head(
            hidden,
            prefill_hidden,
            padded_n,
            proc_count,
            prefill_is_last,
            h,
            bf16,
            fp32,
            stream,
        )?;
        let decode_logits = head_out.decode_logits;
        let prefill_logits = head_out.prefill_logits;

        // ── 8. Update sequence states (after all computation) ──
        for (i, seq) in decode_seqs.iter_mut().enumerate() {
            seq.tokens.push(decode_tokens[i]);
            seq.seq_len += 1;
        }
        prefill_seq.tokens.extend_from_slice(
            &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill],
        );
        prefill_seq.seq_len = prefill_chunk_start + n_prefill;

        Ok(crate::traits::MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }
}
