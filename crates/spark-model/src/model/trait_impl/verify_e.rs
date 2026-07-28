// SPDX-License-Identifier: AGPL-3.0-only

//! Batched K=4 verify: n sequences × 4 rows in ONE eager forward.
//!
//! Generalizes verify_c2's single-sequence K=4 body to `R = n*4` seq-major
//! rows (`r = i*4 + j`) so the n weight-reading verify forwards collapse into
//! one — the structural fix for the measured MTP serialization at C>1
//! (cap=4 at C=4: 25.8 vs 48.5 tok/s; see BATCHED_MTP_SPEC.md).
//!
//! EAGER ONLY: the slot-keyed `verify4_graph` cache is meaningless at n>1
//! (a graph would bake one slot-vector's state pointers); slot-vector-keyed
//! capture is a measured follow-up. Everything per-sequence (GDN conv+WY4
//! body, block tables, rollback intermediates) reuses existing machinery
//! verbatim — only base addresses move.
//!
//! Same `unsafe { from_raw_parts(...) }` pattern as verify_c.rs; see that
//! file's module docs for the full safety contract.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// Whether the batched K=4 verify can run for `n` sequences.
    ///
    /// Self-gates to the envelope verify_e was built and audited for:
    /// non-EP, non-HSS, non-DFlash, no LoRA (the uniform seq_slot upload
    /// carries ONE adapter slot), MTP proposer present (stash allocated),
    /// n in 2..=4 (R ≤ 16 = the proven decode_a2 C=16 metadata + buffer
    /// envelope). Everything outside falls back to the per-seq loop.
    pub(super) fn can_batch_verify_k4_dispatch(&self, n: usize) -> bool {
        (2..=4).contains(&n)
            && self.comm.is_none()
            && self.lora.is_none()
            && self.dflash_hidden_save.is_none()
            && !self.verify_hidden_stash.is_null()
            // HSS: the paged-decode kernel reads HBM only, missing on-disk
            // history (see verify_c2's HSS fallback) — batched path unsupported.
            && self
                .kv_cache
                .lock()
                .config()
                .cache_blocks_per_seq
                .is_none()
    }

    /// Batched K=4 verify for `n = seqs.len()` sequences (R = n*4 rows).
    ///
    /// Row r = i*4 + j is sequence i's token j (`tokens[i] = [last_verified,
    /// d0, d1, d2]`). Weight-bearing ops (QKVZ/out_proj/FFN/lm_head) batch
    /// across all R rows via the existing M-generic arms; attention runs
    /// through `decode_multi_seq` with per-row block tables / seq lens; the
    /// GDN conv+WY4 body runs per-sequence via `decode_verify_multi`
    /// (byte-identical per-seq math, row-offset bases).
    ///
    /// On success: per seq `tokens` += 4 drafts, `seq_len` += 4 (verdict
    /// rewind is the caller's arithmetic, as on the per-seq path). On Err no
    /// sequence state has been advanced. Logits rows stay live for row-based
    /// pipeline picks until the next forward — callers must consume them
    /// (and stash hiddens) BEFORE any propose.
    pub(super) fn decode_verify_batched_k4_dispatch(
        &self,
        tokens: &[[u32; 4]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 4]>> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let k = 4usize;
        let n = seqs.len();
        ensure!(n >= 2 && tokens.len() == n, "batched verify: n={n} tokens={}", tokens.len());
        let r_total = n * k;
        // R ≤ 16: the proven decode_a2 C=16 metadata envelope. The meta gaps
        // below (positions ≤128 B at +0, seq_slot at +128, slots ≤256 B at
        // +256, seq_lens at +512, bt staging sized for 32 rows in sizes.rs)
        // and the 32-row logits cap all hold at R=16 with 2x margin.
        ensure!(r_total <= 16, "batched verify: R={r_total} exceeds the audited 16-row envelope");

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: embed R tokens + allocate KV blocks ──
        for (i, toks) in tokens.iter().enumerate() {
            for (j, &t) in toks.iter().enumerate() {
                self.embed(t, hidden.offset((i * k + j) * h * bf16), stream)?;
            }
        }

        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            let last_pos = seq.seq_len + k - 1;
            ensure_blocks_through_decode(
                seq,
                last_pos / bs,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // ── Phase 2: R-row attention metadata (verify_c2 layout, R rows) ──
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;

        let mut positions = [0u32; 16];
        let mut slots = [0i64; 16];
        let mut seq_lens = [0i32; 16];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..k {
                let r = i * k + j;
                let pos = seq.seq_len + j;
                positions[r] = pos as u32;
                let physical_block = seq.physical_block_for(pos / bs).unwrap_or(0);
                slots[r] = (physical_block as i64) * (bs as i64) + ((pos % bs) as i64);
                // Per-row causal clamp: row r attends through its own position.
                seq_lens[r] = (pos + 1) as i32;
            }
        }
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, r_total * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;
        let slot_bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, r_total * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;
        let sl_bytes =
            unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, r_total * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        // Block tables: row r = seq i's table (bt staging sized for 32 rows).
        let needed = r_total * mb;
        let mut bt_buf = vec![0i32; needed];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..k {
                let row = i * k + j;
                for (bi, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + bi] = block as i32;
                }
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // No-LoRA gate in can_batch: uniform upload returns DevicePtr(0)
        // (installed-pair path) — kept for structural parity with verify_c2.
        debug_assert!(r_total <= 32, "verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot = self.upload_seq_slot_uniform(
            seqs[0].adapter_slot,
            r_total,
            meta_base.offset(128),
            stream,
        )?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: r_total as u32,
            seq_slot,
        };

        // ATLAS_K4_DIAG=1: stream-sync checkpoint after every layer so an
        // illegal access is attributed to the exact layer (same hatch as
        // verify_c2 — this path is ALWAYS eager, so it is always legal).
        let k4_diag = std::env::var("ATLAS_K4_DIAG").ok().as_deref() == Some("1");

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };

        // Host-side per-row attention args (verify_c2 pattern, R rows).
        let mut seq_lens_vec: Vec<usize> = Vec::with_capacity(r_total);
        let mut block_tables_vec: Vec<Vec<u32>> = Vec::with_capacity(r_total);
        for seq in seqs.iter() {
            for j in 0..k {
                seq_lens_vec.push(seq.seq_len + j);
                block_tables_vec.push(seq.block_table.clone());
            }
        }

        // ── Phase 3: layer loop (one R-row weight sweep) ──
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            if layer_type == LayerType::FullAttention {
                let mut dummy_states: Vec<Box<dyn LayerState>> = (0..r_total)
                    .map(|_| layer.alloc_state(self.gpu.as_ref()))
                    .collect::<Result<_>>()?;
                let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                    dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    r_total,
                    &mut refs,
                    &mut kv_cache,
                    &seq_lens_vec,
                    &block_tables_vec,
                    &ctx,
                    stream,
                )?;
            } else {
                let mut state_refs: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|s| s.layer_states[layer_idx].as_mut())
                    .collect();
                layer.decode_verify_multi(
                    hidden,
                    residual,
                    n,
                    k,
                    &mut state_refs,
                    &mut kv_cache,
                    &ctx,
                    stream,
                )?;
            }

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!(
                    "K4_DIAG(batched): CUDA error after layer {layer_idx} ({layer_type:?}): {e:#}"
                );
            }
        }

        // ── Phase 4: final norm [R, H] + lm_head + per-row argmax ──
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            r_total as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;

        if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
            anyhow::bail!("K4_DIAG(batched): CUDA error after final norm: {e:#}");
        }

        // R ≤ 16 < the 32-row logits buffer cap (sizes.rs).
        self.lm_head_batched(normed, r_total as u32, self.buffers.logits(), stream)?;

        if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
            anyhow::bail!("K4_DIAG(batched): CUDA error after lm_head_batched: {e:#}");
        }

        let vocab = self.config.vocab_size;
        let argmax_out = self.buffers.scratch();
        for r in 0..r_total {
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                self.buffers.logits().offset(r * vocab * bf16),
                argmax_out.offset(r * 4),
                vocab as u32,
                stream,
            )?;
        }

        // ── Phase 5: D2H + host bookkeeping ──
        let mut buf = vec![0u8; r_total * 4];
        self.gpu.copy_d2h(argmax_out, &mut buf)?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = [0u32; 4];
            for j in 0..k {
                let o = (i * k + j) * 4;
                v[j] = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            }
            out.push(v);
        }

        for (i, seq) in seqs.iter_mut().enumerate() {
            for &t in &tokens[i] {
                seq.tokens.push(t);
            }
            seq.seq_len += k;
        }

        Ok(out)
    }
}
