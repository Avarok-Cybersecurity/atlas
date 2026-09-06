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
        midcap: Option<&super::midchunk_capture::MidCapturePlan>,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        // BF16 residual is the shipping config (2 bytes/element).
        let elem_bytes = 2usize;
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

        // Request-scoped LoRA routing (chunked prefill) — dedicated arena buffer
        // holding `proc_count` uniform slots (see prefill_a.rs). Covers both the
        // paged-prefill layer path and the warm-prefix `use_decode_path` fork
        // (proc_count==1): the single-seq decode apply reads slot[0], correct
        // for the uniform buffer. `DevicePtr(0)` (no pool) → installed-pair path.
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;
        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: positions_h_dev,
            positions_w: positions_w_dev,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };

        // Consume the one-shot ATLAS_PROFILE_FIRST flag (additive).
        let profile_now = self.profile
            || self
                .profile_first_pending
                .swap(false, std::sync::atomic::Ordering::Relaxed);

        // Mid-chunk tail capture (opt-in): fresh per-pass SSM-layer ordinal
        // counter; each SSM layer's prefill increments it once, in model order,
        // to index the plan's per-layer snapshot destinations.
        let midcap_counter = std::sync::atomic::AtomicUsize::new(0);
        let midchunk_capture = midcap.map(|p| crate::layer::MidchunkCapture {
            cap_local: p.cap_local,
            h_dsts: &p.h_dsts,
            conv_dsts: &p.conv_dsts,
            h_bytes: p.h_bytes,
            conv_bytes: p.conv_bytes,
            ssm_layer_counter: &midcap_counter,
            cap_local_early: p.cap_local_early,
            h_dsts_early: &p.h_dsts_early,
            conv_dsts_early: &p.conv_dsts_early,
        });

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
            profile: profile_now,
            comm: self.comm_ref(),
            graph_capture: false,
            // Marconi warm hit: GDN layers replay from a restored SSM state
            // and must use the bit-faithful WY4 recurrence (see layer.rs).
            gdn_exact_replay: marconi_skip,
            // Hash-MoE: this chunk's token IDs (uploaded in prefill_b_embed_chunk
            // to the stable buffer, in chunk order matching the MoE loop).
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            // #30: request slot pairs (None unless routing to a non-active slot).
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
            midchunk_capture,
            moe_lora_route: self.moe_lora_route(seq.adapter_slot),
        };

        // When proc_count == 1 (warm prefix cache hit), use the decode layer path
        // instead of the prefill path. Decode uses GEMV kernels optimized for M=1
        // and the decode MoE path, which is ~7x faster per layer than the prefill
        // GEMM path for a single token (0.7ms/layer vs 5ms/layer).
        let use_decode_path = proc_count == 1 && effective_seq_len_start > 0;
        // Marconi warm hit: this pass replays SSM state over [snap_tok,
        // matched) — positions whose K/V already live in shared prefix-cache
        // blocks. Pass the per-chunk count of those replay tokens as the
        // layer write floor so attention layers do NOT rewrite them with
        // non-bit-exact recomputed values (drift would poison the shared
        // blocks and ratchet across turns). `seq.cached_prefix_tokens` is
        // the radix-tree match point; tokens at or past it are new and are
        // written normally.
        let layer_kv_write_start = if marconi_skip {
            seq.cached_prefix_tokens
                .saturating_sub(effective_seq_len_start)
                .min(proc_count)
        } else {
            kv_write_start
        };
        let prefill_t0 = if profile_now {
            self.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut layer_times: Vec<u128> = Vec::new();
        // HOST-TIME instrumentation (ATLAS_PREFILL_HOST_TIMING=1). Distinct
        // from `profile_now`: that path synchronizes per layer, which
        // serialises host and device and hides the host-side cost this is
        // meant to expose. Here NOTHING synchronizes — these are pure host
        // wall-clock spans, to be compared against the GPU-busy time an nsys
        // trace reports for the same request.
        let host_timing = std::env::var("ATLAS_PREFILL_HOST_TIMING").as_deref() == Ok("1");
        let t_loop = host_timing.then(std::time::Instant::now);
        let mut t_in_prefill = std::time::Duration::ZERO;
        let mut t_dflash = std::time::Duration::ZERO;
        for (i, layer) in self.layers.iter().enumerate() {
            let t_pf = host_timing.then(std::time::Instant::now);
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
            if let Some(t) = t_pf {
                t_in_prefill += t.elapsed();
            }
            let t_df = host_timing.then(std::time::Instant::now);
            // DFlash chunked-prefill capture. The destination base is the
            // chunk's PROCESSING start — the absolute position of buffer row
            // 0 — NOT `layer_kv_write_start`. The two coincide on a cold
            // prompt, which is how the KV write floor ended up here, but on a
            // prefix-cache hit the forward recomputes from position 0 while
            // KV writes start at the matched boundary: passing the floor
            // shifted EVERY captured row up by the cached-prefix length, left
            // ctx rows [0, matched) holding the pool slot's previous
            // occupant, and advanced ctx_len past the prompt. At C=1 the
            // stale rows were the identical previous request's (accidentally
            // correct); under concurrency they were another sequence's, the
            // drafts went to garbage, and acceptance pinned at ~0 — the
            // cache-hit half of the 2026-08-21 derailment matrix.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                effective_seq_len_start,
                proc_count,
                stream,
            )?;
            if let Some(t) = t_df {
                t_dflash += t.elapsed();
            }
            if let Some(lt0) = lt0 {
                self.gpu.synchronize(stream)?;
                layer_times.push(lt0.elapsed().as_micros());
            }
            // Hyper-stream RMS trail (`ATLAS_DUMP_HYPER_RMS=1`): after each
            // layer, RMS over the FP32 mHC highway [proc_count, hc_mult, H]
            // — directly comparable to the reference golden's `layer_rms`
            // (bench/qwen4_exp/forward_ref.py) since both sides keep the
            // highway in f32. Localizes which layer the engine's logit
            // dilution (KL 0.3-1.6 nats/token vs reference) first appears
            // in. Debug-only: syncs + D2H per layer.
            if std::env::var("ATLAS_DUMP_HYPER_RMS").as_deref() == Ok("1")
                && self.config.hc_mult > 0
            {
                let n = proc_count * self.config.hc_mult * self.config.hidden_size;
                let mut buf = vec![0u8; n * 4];
                self.gpu
                    .copy_d2h_on_stream(ctx.buffers.hc_streams(), &mut buf, stream)?;
                let vals: &[f32] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) };
                let ssq: f64 = vals.iter().map(|&v| (v as f64) * (v as f64)).sum();
                tracing::info!(
                    "HYPER_RMS layer {i} tokens {proc_count} rms {:.6}",
                    (ssq / n as f64).sqrt()
                );
            }
            // MLA diagnostic: per-layer hidden norm for Mistral (once per model).
            // Per-model latch (see `ModelStats::dumped`) rather than a static: an
            // operator who sets the flag and then swaps models must still get the
            // dump, instead of it being swallowed by the previous model's shot.
            if profile_now
                && self.config.model_type == "mistral"
                && self.stats.dumped.keyed("mla_chunk_norms")
            {
                self.gpu.synchronize(stream)?;
                let last_offset = (proc_count - 1) * self.config.hidden_size * 4;
                let h_sz = self.config.hidden_size;
                let mut buf = vec![0u16; h_sz];
                // SAFETY: `buf` is `vec![0u16; h_sz]` on the line above, so it
                // owns exactly `h_sz * size_of::<u16>()` initialised bytes and
                // its length equals its capacity. `bytes` is the only live
                // reference to that allocation for its whole lifetime — last
                // used on the `copy_d2h` line below, and `buf` is not read again
                // until after that.
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
                    if i == self.layers.len() - 1 {}
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
                let (vals, _) = self.readback_bf16(hidden.offset(last_start * elem_bytes), h)?;
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
                    self.readback_bf16(hidden.offset(last_start * elem_bytes), h.min(16))?;
                let lt = self.config.layer_type(i);
                tracing::warn!(
                    "DIAG L{i} ({lt:?}) last_tok_norm={norm:.4} first2={:.4?}",
                    &vals[..2.min(vals.len())]
                );
            }
        }
        if let Some(t) = t_loop {
            let wall = t.elapsed();
            let ffn_us = crate::layers::qwen3_attention::take_ffn_host_us();
            let ph = crate::layers::qwen3_attention::take_attn_phase_us();
            // loop_wall is the host's own elapsed time across the whole layer
            // dispatch. `in_prefill` is time inside layer.prefill() itself,
            // `dflash` is the per-layer DFlash capture (expected ~0 for models
            // with DFlash disabled), and the remainder is other per-layer
            // bookkeeping. Compare loop_wall against nsys GPU-busy time: the
            // difference is host work not overlapped with the device.
            tracing::info!(
                "PREFILL HOST TIMING layers={} tokens={}: loop_wall={:.1}ms in_prefill={:.1}ms ffn={:.1}ms attn_rest={:.1}ms dflash={:.1}ms | qkv={:.1}ms mid={:.1}ms attn_kernel={:.1}ms",
                self.layers.len(),
                proc_count,
                wall.as_secs_f64() * 1e3,
                t_in_prefill.as_secs_f64() * 1e3,
                ffn_us as f64 / 1e3,
                (t_in_prefill.as_micros() as f64 - ffn_us as f64) / 1e3,
                t_dflash.as_secs_f64() * 1e3,
                ph[0] as f64 / 1e3,
                ph[1] as f64 / 1e3,
                ph[2] as f64 / 1e3,
            );
        }

        // ATLAS_MTP_DRAFTER_PREFILL: capture this chunk's final-layer hidden
        // rows for the whole-prompt drafter prefill. No-op when disabled.
        self.try_mtp_prefill_capture(seq, effective_seq_len_start, proc_count, stream)?;
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
            // Aggregate the same per-layer samples by layer type so the profile
            // attributes cost to mamba / moe / attention instead of bare indices.
            let mut by_type: std::collections::BTreeMap<String, (u128, usize)> =
                std::collections::BTreeMap::new();
            for (i, us) in layer_times.iter().copied().enumerate() {
                let e = by_type
                    .entry(format!("{:?}", self.config.layer_type(i)))
                    .or_insert((0, 0));
                e.0 += us;
                e.1 += 1;
            }
            let per_type: Vec<String> = by_type
                .iter()
                .map(|(k, (us, n))| {
                    format!(
                        "{}x{}={:.0}ms(avg {:.1})",
                        n,
                        k,
                        *us as f64 / 1000.0,
                        *us as f64 / 1000.0 / *n as f64
                    )
                })
                .collect();
            tracing::info!(
                "Prefill chunk {} tok (proc {}, {}): {:.1}ms total, by_type: {}, top5: {}",
                chunk_len,
                proc_count,
                path_label,
                total_us as f64 / 1000.0,
                per_type.join(", "),
                top5.join(", "),
            );
        }
        Ok(())
    }
}
