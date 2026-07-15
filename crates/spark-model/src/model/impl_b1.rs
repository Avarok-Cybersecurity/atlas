// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// Upload batch metadata with fixed stride for CUDA graph compatibility.
    ///
    /// Uses `self.max_blocks_per_seq` as a constant block_table stride (captured
    /// in the graph). Pads to `padded_n` with dummy entries for unused slots.
    pub(super) fn upload_batch_metadata_fixed(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let n = seqs.len();
        let block_size = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        let mut positions = Vec::with_capacity(padded_n);
        let mut slots = Vec::with_capacity(padded_n);
        let mut seq_lens_host = Vec::with_capacity(padded_n);
        // Default-fill with `dummy_kv_block` so any kernel out-of-bounds read
        // lands on the always-zeroed dummy block instead of physical block 0
        // (which is dummy_kv_block, also zero — but the explicit sentinel
        // mirrors vLLM's pad_slot_id pattern (PR #6214 / #32118) and makes
        // the intent obvious to future readers).
        let mut block_table_flat: Vec<i32> =
            vec![self.dummy_kv_block as i32; padded_n * max_blocks as usize];

        // Active sequences
        for (i, seq) in seqs.iter().enumerate() {
            let pos = seq.seq_len as u32;
            positions.push(pos);

            let block_idx = pos as usize / block_size;
            let block_offset = pos as usize % block_size;
            let physical_block = seq
                .physical_block_for(block_idx)
                .unwrap_or(self.dummy_kv_block);
            let slot = (physical_block as i64) * (block_size as i64) + (block_offset as i64);
            slots.push(slot);

            seq_lens_host.push((seq.seq_len + 1) as i32);

            // CONCURRENT-DECODE INVARIANT: a real seq's block_table must cover
            // its (seq_len + 1) tokens. If shorter, paged attention OOB-reads
            // dummy_kv_block (now safe via sentinel above) but SSM state has
            // already been advanced — corruption follows. Catch in dev builds.
            debug_assert!(
                seq.block_table.len() > (seq.seq_len / block_size),
                "seq slot={} seq_len={} block_table.len={} (need >= {})",
                seq.slot_idx,
                seq.seq_len,
                seq.block_table.len(),
                (seq.seq_len / block_size) + 1,
            );

            for (j, &block) in seq.block_table.iter().take(max_blocks as usize).enumerate() {
                block_table_flat[i * max_blocks as usize + j] = block as i32;
            }
        }

        // Padding slots: write to dummy KV block, seq_len=1 (position 0)
        let dummy_slot = (self.dummy_kv_block as i64) * (block_size as i64);
        for i in n..padded_n {
            positions.push(0);
            slots.push(dummy_slot);
            seq_lens_host.push(1);
            block_table_flat[i * max_blocks as usize] = self.dummy_kv_block as i32;
        }

        let meta_base = self.buffers.scratch().offset(32768);
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let sl_bytes: Vec<u8> = seq_lens_host.iter().flat_map(|s| s.to_le_bytes()).collect();
        let bt_bytes: Vec<u8> = block_table_flat
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();

        self.gpu.copy_h2d_async(&pos_bytes, meta_base, stream)?;
        self.gpu
            .copy_h2d_async(&slot_bytes, meta_base.offset(256), stream)?;
        self.gpu
            .copy_h2d_async(&sl_bytes, meta_base.offset(512), stream)?;
        self.gpu
            .copy_h2d_async(&bt_bytes, meta_base.offset(768), stream)?;

        // M2 per-request LoRA routing: upload the per-seq adapter-slot buffer to
        // the unused metadata gap at meta_base+128 (positions occupy +0..+32 at
        // padded_n<=8; slots begin at +256 — so +128 never overlaps). Fixed
        // address, per-step contents → graph-safe. `DevicePtr(0)` when no
        // adapter is resident (the bgmv apply sites then no-op).
        let seq_slot = self.upload_seq_slots(seqs, padded_n, meta_base.offset(128), stream)?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: padded_n as u32,
            seq_slot,
        })
    }

    /// Build + upload the `[padded_n]` i32 adapter-slot buffer for per-request
    /// LoRA routing, at `dst`. Returns `dst` when an adapter pool is resident
    /// (so the batched bgmv reads it), or `DevicePtr(0)` when there is no LoRA
    /// (apply sites skip). Resolution + pad handling live in the pure
    /// [`crate::lora::build_seq_slot_host`] (unit-tested).
    fn upload_seq_slots(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let active = match self.lora.as_ref() {
            Some(lw) => lw.active as i32,
            None => return Ok(DevicePtr(0)),
        };
        let adapter_slots: Vec<i32> = seqs.iter().map(|s| s.adapter_slot).collect();
        let host = crate::lora::build_seq_slot_host(&adapter_slots, padded_n, active);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }

    /// Upload a UNIFORM `[count]` i32 adapter-slot buffer where every row =
    /// `resolve(adapter_slot, active)` (`-1` → active). Used by the
    /// single-request paths (single-seq decode, one-request prefill, and
    /// spec-verify of one sequence): those all carry a single `adapter_slot`,
    /// applied to `count` rows (`count == 1` for decode/verify, `count == m`
    /// for prefill). Returns `dst` when an adapter pool is resident (so the
    /// routed bgmv reads it) or `DevicePtr(0)` when there is no LoRA (apply
    /// sites then take the byte-identical installed-pair fallback). Resolution
    /// \+ `count`-fill go through the unit-tested
    /// [`crate::lora::build_seq_slot_host`].
    pub(crate) fn upload_seq_slot_uniform(
        &self,
        adapter_slot: i32,
        count: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let active = match self.lora.as_ref() {
            Some(lw) => lw.active as i32,
            None => return Ok(DevicePtr(0)),
        };
        // Byte-identity guard: a request whose EFFECTIVE adapter is the active one
        // (no per-request `adapter` field -> adapter_slot=-1 -> active, OR it named
        // the active adapter) must keep the INSTALLED-pair path (apply_lora_delta:
        // dense_gemm_tc for prefill, gemv for m=1 decode) — NOT the bgmv, whose
        // per-row gemv would perturb prefill numerics vs today. Return the null
        // buffer so the apply site is untouched. ONLY a request routing to a
        // DIFFERENT (non-active) adapter uploads a slot buffer and takes the bgmv
        // (a NEW routed path — no prior byte-identity baseline to preserve).
        let resolved = if adapter_slot >= 0 {
            adapter_slot
        } else {
            active
        };
        if resolved == active {
            return Ok(DevicePtr(0));
        }
        let slots = vec![adapter_slot; count];
        let host = crate::lora::build_seq_slot_host(&slots, count, active);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }

    /// #30 (routed-prefill precision): the request slot's GLOBAL-layer-indexed
    /// LoRA pairs for a prefill's `ForwardContext.routed_lora_layers`, borrowed
    /// from the pool. `Some` ONLY when `adapter_slot` routes to a NON-active,
    /// in-range slot (see [`crate::lora::LoraWeights::routed_prefill_slot`], the
    /// same predicate `upload_seq_slot_uniform` uses to decide bgmv-vs-installed);
    /// `None` for active/base requests and no-LoRA runs (installed-pair path,
    /// byte-identical). A shared `&self.lora` borrow living exactly as long as the
    /// prefill `ForwardContext`.
    pub(crate) fn routed_slot_layers(
        &self,
        adapter_slot: i32,
    ) -> Option<&[Option<crate::lora::LoraLayerWeights>]> {
        let lw = self.lora.as_ref()?;
        let resolved = lw.routed_prefill_slot(adapter_slot)?;
        Some(lw.slots[resolved].layers.as_slice())
    }

    /// Upload batch metadata to a caller-specified device address.
    ///
    /// Same layout as `upload_batch_metadata_fixed` (positions at +0, slots
    /// at +256, seq_lens at +512, block_table at +768) but writes to
    /// `meta_base` instead of the hardcoded `scratch+32768`. Used by the
    /// fused `mixed_forward` to place decode metadata at a non-conflicting
    /// offset within the scratch buffer.
    pub(super) fn upload_batch_metadata_at(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        kv_cache: &mut PagedKvCache,
        meta_base: DevicePtr,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let n = seqs.len();
        let block_size = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        let mut positions = Vec::with_capacity(padded_n);
        let mut slots = Vec::with_capacity(padded_n);
        let mut seq_lens_host = Vec::with_capacity(padded_n);
        // Sentinel default: see upload_batch_metadata_fixed for rationale.
        let mut block_table_flat: Vec<i32> =
            vec![self.dummy_kv_block as i32; padded_n * max_blocks as usize];

        for seq in seqs.iter() {
            let pos = seq.seq_len as u32;
            positions.push(pos);

            let block_idx = pos as usize / block_size;
            let block_offset = pos as usize % block_size;
            let physical_block = seq
                .physical_block_for(block_idx)
                .unwrap_or(self.dummy_kv_block);
            let slot = (physical_block as i64) * (block_size as i64) + (block_offset as i64);
            slots.push(slot);

            seq_lens_host.push((seq.seq_len + 1) as i32);
        }

        for (i, seq) in seqs.iter().enumerate() {
            for (j, &block) in seq.block_table.iter().take(max_blocks as usize).enumerate() {
                block_table_flat[i * max_blocks as usize + j] = block as i32;
            }
        }

        // Padding slots
        let dummy_slot = (self.dummy_kv_block as i64) * (block_size as i64);
        for i in n..padded_n {
            positions.push(0);
            slots.push(dummy_slot);
            seq_lens_host.push(1);
            block_table_flat[i * max_blocks as usize] = self.dummy_kv_block as i32;
        }

        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let sl_bytes: Vec<u8> = seq_lens_host.iter().flat_map(|s| s.to_le_bytes()).collect();
        let bt_bytes: Vec<u8> = block_table_flat
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();

        self.gpu.copy_h2d_async(&pos_bytes, meta_base, stream)?;
        self.gpu
            .copy_h2d_async(&slot_bytes, meta_base.offset(256), stream)?;
        self.gpu
            .copy_h2d_async(&sl_bytes, meta_base.offset(512), stream)?;
        self.gpu
            .copy_h2d_async(&bt_bytes, meta_base.offset(768), stream)?;

        // Per-request routing slots at the +128 gap (see upload_batch_metadata_fixed).
        let seq_slot = self.upload_seq_slots(seqs, padded_n, meta_base.offset(128), stream)?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: padded_n as u32,
            seq_slot,
        })
    }

    /// Read back first `n` BF16 values from device and return as f32 + L2 norm.
    pub(super) fn readback_bf16(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 2;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }

    /// Read FP32 values from GPU memory (diagnostics).
    pub(super) fn readback_f32(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 4;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }

    /// Profile mode: run each layer with sync+timing, no CUDA graph.
    pub(super) fn decode_profiled(
        &self,
        token: u32,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        use std::time::Instant;

        let num_attn = self.config.num_attention_layers();
        let mut attn_us = 0u64;
        let mut ssm_us = 0u64;
        // Detailed per-operation profiling:
        // - First 2 decode tokens: always (for diagnostics)
        // - MLA models: always (per-op GPU sync prevents buffer aliasing corruption
        //   in the absorbed attention path — Q_absorbed, Q_rope, V_extracted share buffers)
        let is_mla = ctx.config.kv_lora_rank > 0;
        let detail = is_mla || seq.seq_len < seq.tokens.len() + 2;
        let inner_ctx = if detail {
            ctx
        } else {
            // Suppress per-op profiling by creating a non-profile context
            &ForwardContext {
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                attn_metadata: ctx.attn_metadata,
                profile: false,
                comm: ctx.comm,
                graph_capture: ctx.graph_capture,
                gdn_exact_replay: false,
                token_ids: None,
                // #30: forward the parent's routing (None on this decode-profiling
                // path, but never silently drop it if a prefill ever re-wraps).
                routed_lora_layers: ctx.routed_lora_layers,
            }
        };

        // Diagnostic: dump hidden state for first 2 decode tokens after prefill
        let diag = seq.seq_len < seq.tokens.len() + 2;
        if diag {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_f32(hidden, 8)?;
            tracing::info!(
                "DIAG tok={} after_embed (FP32): norm={:.4} vals={:.4?}",
                seq.seq_len,
                norm,
                &vals[..4]
            );
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let t0 = Instant::now();
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                inner_ctx,
                stream,
            )?;
            self.gpu.synchronize(stream)?;
            let elapsed = t0.elapsed().as_micros() as u64;
            if self.config.layer_type(i) == atlas_core::config::LayerType::FullAttention {
                attn_us += elapsed;
            } else {
                ssm_us += elapsed;
            }

            // Diagnostic: after each layer for first token
            if diag {
                let (vals, norm) = self.readback_f32(hidden, 8)?;
                let lt = self.config.layer_type(i);
                tracing::info!(
                    "DIAG tok={} after_L{} ({:?}) [FP32]: norm={:.4} vals={:.4?}",
                    seq.seq_len,
                    i,
                    lt,
                    norm,
                    &vals[..4]
                );
            }
        }

        // Final norm + LM head
        let t0 = Instant::now();
        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;
        self.lm_head(normed, stream)?;
        self.gpu.synchronize(stream)?;
        let head_us = t0.elapsed().as_micros() as u64;

        // Diagnostic: dump top-5 logits
        if diag {
            let logits_ptr = self.buffers.logits();
            let v = self.config.vocab_size;
            let mut logit_buf = vec![0u8; v * 2];
            self.gpu.copy_d2h(logits_ptr, &mut logit_buf)?;
            let logits: Vec<f32> = logit_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            tracing::info!("DIAG tok={} top5_logits: {:?}", seq.seq_len, &indexed[..5]);
        }

        let total_us = attn_us + ssm_us + head_us;
        tracing::info!(
            "PROFILE tok={}: total={:.1}ms attn={:.1}ms({}) ssm={:.1}ms({}) head={:.1}ms",
            seq.seq_len,
            total_us as f64 / 1000.0,
            attn_us as f64 / 1000.0,
            num_attn,
            ssm_us as f64 / 1000.0,
            self.layers.len() - num_attn,
            head_us as f64 / 1000.0,
        );

        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.decode_logits_ptr())
    }

    /// Eager decode skipping SSM layers. Used by self-speculative drafting.
    /// KV cache entries are appended (will be overwritten by verify).
    /// SSM state is NOT updated (SSM layers are skipped entirely).
    pub(super) fn decode_draft(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 1. Embedding lookup
        self.embed(token, hidden, stream)?;

        // 2. Pre-allocate KV cache blocks + upload attention metadata
        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
        ensure_blocks_through_decode(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = seq.block_table.len() as u32;

        let pos_val = seq.seq_len as u32;
        self.gpu
            .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        self.gpu
            .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;

        let actual_seq_len = (seq.seq_len + 1) as i32;
        self.gpu
            .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        // Request-scoped LoRA routing for the draft pass (same 1-elem +128-gap
        // layout as decode_a). Without it, self-speculative drafts would be
        // proposed with the global active adapter and mostly rejected by a
        // correctly-routed verify — a pure acceptance-rate loss, not a
        // correctness one, but cheap to avoid.
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, 1, meta_base.offset(128), stream)?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false, // Eager mode — no CUDA graph
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None, // #30: offline single-seq decode; no prefill route.
        };

        // Eager layer loop: skip SSM layers, run attention layers only
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                continue; // Skip SSM layers
            }
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &ctx,
                stream,
            )?;
        }

        // Final norm + LM head
        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;
        self.lm_head(normed, stream)?;

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }
}
