// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase A — non-chunked single-pass path.
//!
//! Same `unsafe { from_raw_parts(...) }` pattern as the verify_*.rs
//! files: stack arrays / `Vec`s of POD integers reinterpreted as byte
//! slices for synchronous-enqueue H2D upload. See `verify_c.rs` module
//! docs for the full safety contract.

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

mod vision;

impl TransformerModel {
    pub(super) fn prefill_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        if n <= 1 {
            // Single token: use decode path (CUDA graph optimized)
            for &token in tokens {
                self.decode(token, seq, stream)?;
            }
            return Ok(self.decode_logits_ptr());
        }

        // Guard: prompt must not exceed buffer arena capacity.
        let arena_cap = self.buffers.max_batch_tokens();
        if n > arena_cap {
            anyhow::bail!(
                "Prompt ({n} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Use chunked prefill (--max-prefill-tokens) or reduce prompt length."
            );
        }

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let _bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Zero ALL buffers (EP=1 and EP=2) — the NCCL all-reduce path reads
        // buffers that may carry stale data from prior requests with different
        // token counts. The EP=2 CUDA 700 was from the 4MB recv buffer overflow
        // (fixed in 1ae4883); zero_all kept everywhere as defense-in-depth.
        self.buffers.zero_all(self.gpu.as_ref(), stream)?;

        let mut kv_cache = self.kv_cache.lock();

        // ── 1. Prefix cache lookup (BEFORE embedding — Marconi may skip tokens) ──
        let bs = kv_cache.block_size();
        let prefix_match = if self.tokens_have_vision_pad(tokens) {
            spark_runtime::prefix_cache::PrefixMatch::empty()
        } else {
            self.prefix_cache
                .lookup(tokens, bs, seq.session_hash, seq.adapter_id)
        };
        let mut kv_write_start = prefix_match.matched_tokens;
        seq.cached_prefix_tokens = prefix_match.matched_tokens;
        // Record the original prompt length — cache_sequence() uses it later
        // to avoid double-bumping ref_counts on the prompt portion.
        seq.prompt_len = n;

        // Reuse cached blocks (inc_ref for shared ownership).
        for &block_idx in &prefix_match.matched_blocks {
            kv_cache.inc_ref(block_idx);
            seq.block_table.push(block_idx);
        }
        reuse_prefix_match_disk_ids(
            &prefix_match.matched_disk_block_ids,
            &mut seq.disk_block_ids,
        );

        // Allocate new blocks for the remaining (uncached) tokens.
        let blocks_needed = (n - 1) / bs + 1;
        // Phase 6.3: single-shot prefill cannot stream long prompts because
        // the K/V for ALL prompt tokens must be HBM-resident before the
        // single Flash Attention pass runs (no per-chunk offload window).
        // Bail with a clear message directing to chunked prefill.
        if let Some(cap) = kv_cache.config().cache_blocks_per_seq
            && blocks_needed > cap as usize
        {
            anyhow::bail!(
                "high-speed-swap: prompt of {} blocks exceeds \
                     --high-speed-swap-cache-blocks-per-seq={}; this single-shot \
                     prefill path requires the whole prompt fit in HBM. Use \
                     chunked prefill (set --max-prefill-tokens ≤ {} × block_size) \
                     to stream long prompts to disk.",
                blocks_needed,
                cap,
                cap
            );
        }
        ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        // ── Marconi: try to restore SSM state and skip cached prefix ──
        // With intermediate checkpoints, ssm_snapshot_tokens may be less than
        // matched_tokens. Use ssm_snapshot_tokens as the skip point.
        // Session isolation: only restore snapshots belonging to this session.
        // Phase 1b spill-tier fault-in (#6): fold a resident hit with a
        // faulted-back spilled anchor; see `ssm_fault_in::eff_ssm_snapshot`.
        let (eff_snapshot, eff_snapshot_tokens) =
            self.eff_ssm_snapshot(&prefix_match, seq.session_hash, stream);
        let marconi_skip = if let Some(snap_id) = eff_snapshot {
            let snap_tok = eff_snapshot_tokens;
            if snap_tok > 0
                && kv_write_start <= n
                && self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash)
            {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                if snap_tok < kv_write_start {
                    tracing::info!(
                        "Marconi intermediate hit: restored from checkpoint at token {} \
                         (skipping {} tokens, recomputing {} SSM tokens to match point {})",
                        snap_tok,
                        snap_tok,
                        kv_write_start - snap_tok,
                        kv_write_start,
                    );
                } else {
                    tracing::info!(
                        "Marconi SSM cache hit: {} tokens skipped ({} blocks), snapshot {}",
                        kv_write_start,
                        prefix_match.matched_blocks.len(),
                        snap_id,
                    );
                }
                // All tokens matched AND the snapshot covers the full match →
                // skip the whole prompt (process only the last token). But an
                // *intermediate* checkpoint at full match (snap_tok < n — e.g. a
                // faulted-in anchor whose leaf was evicted) restored state at
                // `snap_tok`, not `n`; skipping to `n` would desync SSM state
                // from KV/positions → garbage. Then skip only to `snap_tok` so
                // the suffix recomputes SSM over [snap_tok, n). (Mirrors the
                // prefill_b/prefill_c warm-hit fix.)
                kv_write_start = if kv_write_start >= n && snap_tok >= kv_write_start {
                    n
                } else {
                    snap_tok
                };
                true
            } else {
                if kv_write_start > 0 {
                    tracing::info!(
                        "Prefix cache hit: {} tokens ({} blocks) reused (KV only)",
                        kv_write_start,
                        prefix_match.matched_blocks.len(),
                    );
                }
                false
            }
        } else {
            let has_ssm_layers = self.config.num_ssm_layers() > 0;
            if kv_write_start > 0 && has_ssm_layers {
                // SSM models: can't reuse KV without SSM snapshot — the SSM state
                // is recomputed from scratch, producing different hidden states than
                // what originally populated the cached KV blocks. Force full KV rewrite.
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — recomputing all KV",
                    kv_write_start,
                    prefix_match.matched_blocks.len(),
                );
                kv_write_start = 0;
                false
            } else if kv_write_start > 0 && kv_write_start < n {
                // Pure attention (MLA/GQA) — no SSM state needed, KV cache is self-contained.
                // Skip cached tokens entirely: only embed + forward uncached suffix.
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) reused, processing {} new tokens (no SSM in this model)",
                    kv_write_start,
                    prefix_match.matched_blocks.len(),
                    n - kv_write_start,
                );
                true
            } else {
                false
            }
        };

        // Determine tokens to actually process
        let (proc_tokens, proc_count, seq_len_start) = if marconi_skip && kv_write_start >= n {
            // Exact match: entire prompt cached with SSM snapshot.
            // Process only the last token through decode path to produce logits.
            (&tokens[n - 1..], 1, n - 1)
        } else if marconi_skip {
            // Partial match: skip cached prefix, process uncached suffix.
            (
                &tokens[kv_write_start..],
                n - kv_write_start,
                kv_write_start,
            )
        } else {
            // Original path: process all tokens
            (tokens, n, 0usize)
        };

        // ── 2. Embed tokens → [proc_count, H] contiguous ──
        {
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(proc_tokens.as_ptr() as *const u8, proc_count * 4)
            };
            let token_ids_dev = self.buffers.scratch();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            // Also stage token IDs into the STABLE token_ids buffer (scratch is
            // reused for MoE routing during the layer loop). DeepSeek-V4 hash-MoE
            // layers read `tid2eid[token_id]` per token, in this same order.
            self.gpu
                .copy_h2d_async(token_ids_bytes, self.buffers.token_ids(), stream)?;
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                proc_count as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, proc_count, stream)?;
        }

        // ── 3. Upload attention metadata via pinned staging (one H2D copy) ──
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);

        let slot_offset = (proc_count * 4 + 7) & !7;

        // Lock staging, build metadata, pack, single H2D copy
        let (block_table_dev, seq_len_dev) = {
            // SAFETY: Single-threaded scheduler access (see TransformerModel Send/Sync docs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(seq_len_start as u32..(seq_len_start + proc_count) as u32);
            stg.slots.clear();
            stg.slots
                .extend((seq_len_start..seq_len_start + proc_count).map(|i| {
                    let block_idx = seq
                        .physical_block_for(i / bs)
                        .unwrap_or(self.dummy_kv_block);
                    (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                }));

            let pinned = stg.ptr;
            let mut cursor = 0usize;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.positions.as_ptr() as *const u8,
                    pinned.add(cursor),
                    proc_count * 4,
                );
            }
            cursor = slot_offset;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.slots.as_ptr() as *const u8,
                    pinned.add(cursor),
                    proc_count * 8,
                );
            }
            cursor += proc_count * 8;

            let devs = if marconi_skip {
                let bt_start = (cursor + 3) & !3;
                let bt_len = seq.block_table.len() * 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        seq.block_table.as_ptr() as *const u8,
                        pinned.add(bt_start),
                        bt_len,
                    );
                }
                let sl_start = (bt_start + bt_len + 3) & !3;
                let seq_len_val = n as u32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &seq_len_val as *const u32 as *const u8,
                        pinned.add(sl_start),
                        4,
                    );
                }
                cursor = sl_start + 4;
                (meta_base.offset(bt_start), meta_base.offset(sl_start))
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };

            assert!(cursor <= stg.bytes, "prefill metadata overflow");
            let pinned_slice = unsafe { std::slice::from_raw_parts(pinned, cursor) };
            self.gpu.copy_h2d_async(pinned_slice, meta_base, stream)?;
            devs
        };

        // ── M2 request-scoped LoRA routing (prefill). Every one of the
        // `proc_count` prompt tokens carries THIS request's adapter — the
        // headline fix (prefill previously always applied the global active
        // adapter, contaminating a routed request's prompt KV). A dedicated
        // arena buffer (`lora_seq_slot`, sized max_batch_tokens) holds the
        // m-element slot array; the packed meta gap is unsafe here because
        // positions span `proc_count*4` bytes from meta_base+0. Prefill is
        // eager (graph_capture:false) + this H2D precedes the layer loop, so
        // it rides the existing metadata phasing. `DevicePtr(0)` (no pool) →
        // the K/V/O apply sites take the byte-identical installed-pair path.
        // `seq.adapter_slot == -1` (no `adapter` field) resolves to active.
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            // Marconi warm hit: GDN layers replay from a restored SSM state
            // and must use the bit-faithful WY4 recurrence (see layer.rs).
            gdn_exact_replay: marconi_skip,
            // Hash-MoE: token IDs for the `proc_count` tokens processed this
            // pass, in MoE-loop order (uploaded above to the stable buffer).
            token_ids: Some(self.buffers.token_ids()),
            // #30: request slot pairs (None unless routing to a non-active slot).
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
        };

        // ── 4. Forward through all layers ──
        // When Marconi skip is active, seq_len_start > 0 triggers paged attention
        // in attention layers. SSM layers process only proc_count tokens
        // using restored h_state + conv_state. On a Marconi intermediate hit
        // the first (matched - snap_tok) processed tokens replay positions
        // already in shared prefix-cache blocks — write-floor them so
        // attention can't rewrite cached K/V with non-bit-exact recompute
        // (see prefill_b/forward_layers.rs). Leaf hit → floor 0 (all new).
        let layer_kv_write_start = if marconi_skip {
            seq.cached_prefix_tokens
                .saturating_sub(seq_len_start)
                .min(proc_count)
        } else {
            kv_write_start
        };
        let diag_prefill = self.profile && proc_count > 1; // Only with --profile
        for (i, layer) in self.layers.iter().enumerate() {
            layer
                .prefill(
                    hidden,
                    residual,
                    proc_count,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq_len_start,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    layer_kv_write_start,
                    &ctx,
                    stream,
                )
                .map_err(|e| anyhow::anyhow!("Prefill layer {i} failed: {e}"))?;
            // DFlash prefill capture: writes layer i's hidden output for
            // all `proc_count` tokens into the seq's accumulator at slots
            // [layer_kv_write_start .. layer_kv_write_start + proc_count].
            // No-op when DFlash is disabled.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                layer_kv_write_start,
                proc_count,
                stream,
            )?;

            // MLA diagnostic: dump per-layer hidden state norm (once per session)
            static DIAG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if self.profile
                && self.config.model_type == "mistral"
                && !DIAG_DONE.load(std::sync::atomic::Ordering::Relaxed)
            {
                self.gpu.synchronize(stream)?;
                // Read last token's hidden state (what goes to LM head)
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, h_sz * 2)
                };
                if self.gpu.copy_d2h(hidden.offset(last_offset), bytes).is_ok() {
                    let vals: Vec<f32> = buf
                        .iter()
                        .map(|&b| f32::from_bits((b as u32) << 16))
                        .collect();
                    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!("LAYER_NORM L{i}: hidden_norm={norm:.4}");
                    if i == self.layers.len() - 1 {
                        DIAG_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }

            // Diagnostic: check last token's hidden state norm at every layer.
            // This is what goes to the LM head — divergence here causes bad logits.
            if diag_prefill {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (last_vals, last_norm) =
                    self.readback_bf16(hidden.offset(last_start * fp32), h.min(64))?;
                let last_nan = last_vals.iter().filter(|v| v.is_nan()).count();
                let last_inf = last_vals.iter().filter(|v| v.is_infinite()).count();
                let lt = self.config.layer_type(i);
                // Print every 4th layer + first/last to keep output manageable
                if i % 4 == 0 || i == self.layers.len() - 1 || last_nan > 0 || last_inf > 0 {
                    tracing::warn!(
                        "DIAG L{i} ({lt:?}) last_tok: norm={last_norm:.4} nan={last_nan} inf={last_inf} first4={:.4?}",
                        &last_vals[..4.min(last_vals.len())]
                    );
                }
            }
        }

        // ── 5. Final norm on LAST token only ──
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            last_hidden,
            &self.final_norm,
            normed,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // ── 6. LM head on last token → logits ──
        self.lm_head(normed, stream)?;

        // ── 7. Update sequence state ──
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = n;
        // #155: prime the decode-checkpoint cadence gate so the first decode
        // checkpoint never fires on a block boundary the prompt already
        // crossed (would snapshot 1-2 tokens past the prompt edge).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // ── 8. Insert into prefix cache + save SSM snapshot for Marconi ──
        self.prefill_save_snapshot_with_vision_gate(tokens, seq, &mut kv_cache, bs, stream);

        // DFlash: advance the seq's `ctx_len` to span all just-prefilled
        // positions so the next propose() can read them.
        self.update_dflash_ctx_len_after_prefill(seq, layer_kv_write_start, proc_count)?;

        Ok(self.decode_logits_ptr())
    }
}
