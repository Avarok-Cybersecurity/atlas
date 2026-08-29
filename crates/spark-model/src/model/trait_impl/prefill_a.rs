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
        seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
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
            self.levers.kv_poison,
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
            // Below `marconi_min_tokens()` the snapshot restore costs more in lost
            // drafter acceptance than the skipped prefill saves — see the helper.
            if snap_tok >= crate::model::mtp_carry::marconi_min_tokens()
                && snap_tok > 0
                && kv_write_start <= n
                && self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash)
                // Aux-carrying models (PLE/QSA) decline aux-less slots — a
                // mid-chunk tail capture, or a snapshot from before this
                // feature — rather than restore a stale lexical state.
                && (!self.requires_aux_state() || self.ssm_snapshots.aux(snap_id).is_some())
            {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                if let Some(aux) = self.ssm_snapshots.aux(snap_id) {
                    self.apply_aux_states(seq, &aux, stream)?;
                }
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
            // SAFETY: `proc_count` is not an independent count. Each of the
            // three arms of the `(proc_tokens, proc_count, seq_len_start)`
            // binding above pairs a subslice of `tokens` with that subslice's
            // OWN length — `&tokens[n-1..]`/1, `&tokens[kv_write_start..]`/
            // `n - kv_write_start`, `tokens`/`n` — so
            // `proc_count == proc_tokens.len()` on every path and the byte
            // length is exactly `proc_tokens.len() * size_of::<u32>()`. The
            // bytes are initialised: `tokens` is a live `&[u32]`.
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
            // `proc_tokens` is always a SUFFIX of `tokens` (all three arms of
            // the binding above slice from the tail), so the tokens preceding
            // it are exactly `tokens[..start]` — which is what the n-gram hash
            // needs to read backwards into. `ngram_lookbehind()` is 0 for
            // models without one, making `ctx` just the processed tokens.
            let start = tokens.len() - proc_count;
            let ctx_start = start.saturating_sub(self.ngram_lookbehind());
            self.embed_tokens_fused(
                &tokens[ctx_start..start + proc_count],
                proc_count,
                hidden,
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

            // Each field is bounded BEFORE it is written — the packer refuses
            // rather than overrunning and asserting on the wreckage afterwards.
            // The field that can actually get too big is `seq.block_table`,
            // whose length tracks the sequence's KV blocks, i.e. the context
            // length.
            //
            // Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after the
            // positions array that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs). `slot_offset` and
            // `proc_count*8` are both multiples of 8, so `bt_start` and
            // `sl_start` round to their inputs and open no further gap.
            let mut pack = stg.packer_for(self.buffers.scratch_bytes().saturating_sub(meta_offset));
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;

            let devs = if marconi_skip {
                let bt_start = (pack.high_water() + 3) & !3;
                pack.put_at("block_table", bt_start, &seq.block_table)?;
                let sl_start = (pack.high_water() + 3) & !3;
                pack.put_at("seq_len", sl_start, &[n as u32])?;
                (meta_base.offset(bt_start), meta_base.offset(sl_start))
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };

            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
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
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
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
            host_token_ids: None,
            // #30: request slot pairs (None unless routing to a non-active slot).
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(seq.adapter_slot),
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

            // MLA diagnostic: dump per-layer hidden state norm (once per model).
            // Per-model latch (see `ModelStats::dumped`) rather than a static: an
            // operator who sets the flag and then swaps models must still get the
            // dump, instead of it being swallowed by the previous model's shot.
            if self.profile
                && self.config.model_type == "mistral"
                && self.stats.dumped.keyed("mla_prefill_norms")
            {
                self.gpu.synchronize(stream)?;
                // Read last token's hidden state (what goes to LM head)
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                // SAFETY: `buf` is `vec![0u16; h_sz]` on the line above, so it
                // owns exactly `h_sz * size_of::<u16>()` initialised bytes and
                // the length matches its capacity. `bytes` is the only live
                // reference to that allocation for its whole lifetime — it is
                // last used on the `copy_d2h` line below, and `buf` is not read
                // again until after that.
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
                    if i == self.layers.len() - 1 {}
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

        // ATLAS_MTP_DRAFTER_PREFILL: capture the processed rows' final-layer
        // hiddens for the whole-prompt drafter prefill. No-op when disabled.
        self.try_mtp_prefill_capture(seq, seq_len_start, proc_count, stream)?;

        // ── 5. Final norm on LAST token only ──
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(last_hidden, normed, 1, h as u32, eps, stream)?;

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
