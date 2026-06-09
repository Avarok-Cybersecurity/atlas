// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 4: forward through every transformer layer (decode-path on
//! single-token chunks, prefill-path otherwise) plus DFlash capture
//! and per-layer profiling/diagnostics.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn prefill_b_forward_layers(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_write_start: usize,
        marconi_skip: bool,
        meta_base: DevicePtr,
        slot_offset: usize,
        pos_stream_bytes: usize,
        use_mrope: bool,
        needs_paged: bool,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let (block_table_dev, seq_len_dev) = if needs_paged {
            let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let (positions_h_dev, positions_w_dev) = if use_mrope {
            (
                meta_base.offset(pos_stream_bytes),
                meta_base.offset(pos_stream_bytes * 2),
            )
        } else {
            (meta_base, meta_base)
        };
        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: positions_h_dev,
            positions_w: positions_w_dev,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
        };

        // Consume the one-shot ATLAS_PROFILE_FIRST flag (additive).
        let profile_now = self.profile
            || self
                .profile_first_pending
                .swap(false, std::sync::atomic::Ordering::Relaxed);

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: profile_now,
            comm: self.comm_ref(),
            graph_capture: false,
        };

        // When proc_count == 1 (warm prefix cache hit), use the decode layer path
        // instead of the prefill path. Decode uses GEMV kernels optimized for M=1
        // and the decode MoE path, which is ~7x faster per layer than the prefill
        // GEMM path for a single token (0.7ms/layer vs 5ms/layer).
        let use_decode_path = proc_count == 1 && effective_seq_len_start > 0;
        let layer_kv_write_start = if marconi_skip { 0 } else { kv_write_start };
        let prefill_t0 = if profile_now {
            self.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut layer_times: Vec<u128> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let lt0 = if profile_now {
                self.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if use_decode_path {
                layer
                    .decode(
                        hidden,
                        residual,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill-as-decode layer {i} failed: {e}"))?;
            } else {
                layer
                    .prefill(
                        hidden,
                        residual,
                        proc_count,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill chunk layer {i} failed: {e}"))?;
            }
            // DFlash chunked-prefill capture.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                layer_kv_write_start,
                proc_count,
                stream,
            )?;
            if let Some(lt0) = lt0 {
                self.gpu.synchronize(stream)?;
                layer_times.push(lt0.elapsed().as_micros());
            }
            // MLA diagnostic: per-layer hidden norm for Mistral (once per session)
            static CHUNK_DIAG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if profile_now
                && self.config.model_type == "mistral"
                && !CHUNK_DIAG_DONE.load(std::sync::atomic::Ordering::Relaxed)
            {
                self.gpu.synchronize(stream)?;
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
                    tracing::info!(
                        "LAYER_NORM L{i}/{}: hidden_norm={norm:.4}",
                        self.layers.len()
                    );
                    if i == self.layers.len() - 1 {
                        CHUNK_DIAG_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
            // Diagnostic: dump hidden state norm after first 4 and last 4 layers
            if profile_now && (i < 4 || i >= self.layers.len() - 4) {
                self.gpu.synchronize(stream)?;
                let (_, norm) = self.readback_bf16(hidden, self.config.hidden_size.min(64))?;
                tracing::info!("L{i} hidden[0] norm={norm:.4}");
            }
            // Per-layer numerical-divergence dump (env-gated, zero overhead when
            // unset). `ATLAS_NEMO_DUMP=<dir>` writes the LAST token's full
            // post-layer residual-stream hidden vector for every layer as
            // headerless little-endian f32: `<dir>/atlas_L{i}.bin`. Overwrites
            // on every call so the final chunk's last token wins (methodology
            // §3 gotcha #5). Compared 1:1 against the HF CPU/GPU oracle.
            if is_last_chunk
                && let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
                && !dir.is_empty()
            {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (vals, _) = if self.config.use_fp32_residual {
                    self.readback_f32(hidden.offset(last_start * fp32), h)?
                } else {
                    self.readback_bf16(hidden.offset(last_start * fp32), h)?
                };
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                let path = std::path::Path::new(&dir).join(format!("atlas_L{i}.bin"));
                std::fs::write(&path, &bytes).ok();
                if i == self.layers.len() - 1 {
                    tracing::info!(
                        "ATLAS_NEMO_DUMP: wrote {} per-layer hidden \
                         vectors ({h} f32 each) to {dir}",
                        self.layers.len()
                    );
                }
            }
            // Last-chunk diagnostic: log LAST token's hidden norm at every layer.
            if profile_now && is_last_chunk && proc_count > 1 && (chunk_start + chunk_len) > 16384 {
                self.gpu.synchronize(stream)?;
                let last_start = (proc_count - 1) * h;
                let (vals, norm) =
                    self.readback_bf16(hidden.offset(last_start * fp32), h.min(16))?;
                let lt = self.config.layer_type(i);
                tracing::warn!(
                    "DIAG L{i} ({lt:?}) last_tok_norm={norm:.4} first2={:.4?}",
                    &vals[..2.min(vals.len())]
                );
            }
        }
        if let Some(t0) = prefill_t0 {
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            let mut indexed: Vec<(usize, u128)> = layer_times.iter().copied().enumerate().collect();
            indexed.sort_by_key(|x| std::cmp::Reverse(x.1));
            let top5: Vec<String> = indexed
                .iter()
                .take(5)
                .map(|(i, us)| format!("L{}={:.2}ms", i, *us as f64 / 1000.0))
                .collect();
            let path_label = if use_decode_path { "decode" } else { "prefill" };
            tracing::info!(
                "Prefill chunk {} tok (proc {}, {}): {:.1}ms total, top5: {}",
                chunk_len,
                proc_count,
                path_label,
                total_us as f64 / 1000.0,
                top5.join(", "),
            );
        }
        Ok(())
    }
}
