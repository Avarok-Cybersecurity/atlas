// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers (`u32`, `i32`, `i64`, `usize`) as byte
//! slices for H2D upload. See `verify_c.rs` module docs for the full
//! safety contract — same pattern, same invariants here.

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
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize; // hidden states = BF16; matches verify_c.rs

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed K tokens
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1c. Upload attention metadata to scratch (offset 32768).
        // Two layouts depending on whether prefill-mode attention is active:
        //   decode:   K-row format (num_seqs=K, K copies of block_table)
        //   prefill:  1-row format (num_seqs=1, single block_table row)
        // Positions and slots are identical in both layouts.
        let use_prefill_attn = std::env::var("ATLAS_VERIFY_PREFILL_ATTN")
            .ok()
            .as_deref()
            == Some("1");

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // positions[K×4] — same for both layouts
        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        // slots[K×8] — same for both layouts
        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let metadata = if use_prefill_attn {
            // Prefill layout: single seq_len value, single block_table row.
            // prefill_attention_paged computes kv_len from seq_len_start + num_tokens
            // and reads block_table as a flat 1D array — no seq_idx stride.
            let seq_len_val = [(seq.seq_len + k) as u32];
            let sl_bytes = unsafe {
                std::slice::from_raw_parts(seq_len_val.as_ptr() as *const u8, 4)
            };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let mut bt_buf = vec![0i32; mb];
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[j] = block as i32;
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, mb * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
            }
        } else {
            // Decode layout: K rows of seq_lens and block_table.
            let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
            let sl_bytes =
                unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let needed = k * mb;
            let mut bt_buf = vec![0i32; needed];
            for row in 0..k {
                for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + j] = block as i32;
                }
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: max_blocks,
                num_seqs: k as u32,
            }
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1: legacy debug flag (still honored).
        let debug_no_graph =
            std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1");
        // PROTECTIVE (pre-existing graphed-K=γ corruption): the K=γ verify graph
        // emits degenerate/corrupt output — a capture bug downstream of the SSM
        // dual-buffer commit (confirmed: graphed ≠ eager at T=0, reproduces with
        // EAGLE on AND off). Force EAGER by default so the default path is
        // CORRECT. Eager is also faster here (SSM-decode-bound: 13 vs 11.5 tok/s).
        // Re-enable graphs ONLY to debug the underlying bug.
        let allow_kgamma_graph =
            std::env::var("ATLAS_DFLASH_UNSAFE_KGAMMA_GRAPH").ok().as_deref() == Some("1");
        if !allow_kgamma_graph {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "DFlash K=γ verify: forcing EAGER execution — the graphed K=γ \
                     path produces corrupt output (pre-existing capture bug \
                     downstream of the SSM dual-buffer commit). Eager is also \
                     faster here. Set ATLAS_DFLASH_UNSAFE_KGAMMA_GRAPH=1 to force \
                     graphs (debugging only — KNOWN to corrupt output)."
                );
            }
        }
        let force_eager = debug_no_graph || !allow_kgamma_graph;
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            // ── ATLAS_DFLASH_TIMING=1: per-phase GPU timing breakdown. Brackets
            // each phase with gpu.synchronize() (serializes — only under the flag).
            // Requires eager (syncs are illegal under graph capture); K=γ is forced
            // eager by the guard, so this is a no-op when graphs are somehow on.
            let timing = (std::env::var("ATLAS_DFLASH_TIMING").ok().as_deref() == Some("1"))
                && !use_graphs;
            if std::env::var("ATLAS_DFLASH_TIMING").ok().as_deref() == Some("1") && use_graphs {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "ATLAS_DFLASH_TIMING ignored: needs eager K=γ (incompatible with graph capture)."
                    );
                }
            }
            let mut us_attn: u128 = 0; // attention prefill (sum over attn layers)
            let mut us_p1: u128 = 0; // SSM phase-1 per-token conv
            let mut us_p2: u128 = 0; // SSM phase-2 GDN (wy16 or wy4+replay)
            let mut us_p3: u128 = 0; // SSM phase-3 norm+proj+FFN
            let mut us_head: u128 = 0; // final norm + lm_head + argmax

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // SSM three-phase parallel verify (ATLAS_VERIFY_PREFILL_SSM=1).
            // When enabled, SSM layers use prefill_phase1/gdn_full/phase3
            // instead of sequential decode_batched(k), reducing GDN work from
            // K serial h_state updates to one WY4 batch recurrence.
            let use_prefill_ssm = std::env::var("ATLAS_VERIFY_PREFILL_SSM")
                .ok()
                .as_deref()
                == Some("1");
            // EAGLE-fix (K=γ): capture ALL k verify rows so the scheduler can
            // append rows 0..=num_accepted to ctx (fixes ctx-undercount + EAGLE
            // shift). Flag off → legacy single row-0 capture.
            let eagle_fix =
                std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
            // Precompute SSM buffer dimensions (same as prefill_c.rs / phase1_inner).
            let nk = self.config.linear_num_key_heads;
            let kd = self.config.linear_key_head_dim;
            let nv = self.config.linear_num_value_heads;
            let vd = self.config.linear_value_head_dim;
            let ssm_key_dim = nk * kd;
            let ssm_value_dim = nv * vd;
            let ssm_conv_dim = ssm_key_dim * 2 + ssm_value_dim;
            let ssm_gate_stride = nv * 2; // elements (FP32)
            let gdn_bufs = GdnPrefillBuffers {
                qkv:       self.gdn_buf_qkv,
                gate_beta: self.gdn_buf_gate_beta,
                output:    self.gdn_buf_out,
                z:         self.gdn_buf_z,
                total_len: k,
            };

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else if use_prefill_attn {
                        debug_assert!(seq.seq_len > 0, "prefill-verify requires non-empty context");
                        let _ts = if timing {
                            self.gpu.synchronize(stream)?;
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
                        layer.prefill(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            0,
                            &ctx,
                            stream,
                        )?;
                        if let Some(t) = _ts {
                            self.gpu.synchronize(stream)?;
                            us_attn += t.elapsed().as_micros();
                        }
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        layer.decode_multi_seq(
                            hidden,
                            residual,
                            k,
                            &mut refs,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else if layer.is_ssm_layer() && use_prefill_ssm {
                    // Three-phase parallel SSM verify (see SSM_VERIFY_PLAN.md).
                    //
                    // Phase 1: K serial single-token calls fill gdn_bufs at
                    // token_offset=t and slide conv_state one step each, so
                    // conv_state_intermediates[t] can be saved after each call.
                    let _ts1 = if timing {
                        self.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    for t in 0..k {
                        layer.prefill_phase1(
                            hidden.offset(t * h * bf16),
                            residual.offset(t * h * bf16),
                            1,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len + t,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            0,
                            &gdn_bufs,
                            t,
                            &ctx,
                            stream,
                        )?;
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        if t < s.conv_state_intermediates.len() {
                            self.gpu.copy_d2d_async(
                                s.conv_state,
                                s.conv_state_intermediates[t],
                                self.ssm_pool.conv_bytes,
                                stream,
                            )?;
                        }
                    }
                    if let Some(t) = _ts1 {
                        self.gpu.synchronize(stream)?;
                        us_p1 += t.elapsed().as_micros();
                    }
                    // Phase 2: single fused WY16 pass (K=16) writes final h_state
                    // AND Hi_0..Hi_14 intermediates inline — eliminating the WY4
                    // batch + the per-token replay loop. Falls back to WY4+replay
                    // when wy16 is unavailable (other K, or kernel NULL).
                    let _ts2 = if timing {
                        self.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let used_wy16 = layer.prefill_gdn_wy16(
                        seq.layer_states[layer_idx].as_mut(),
                        &gdn_bufs,
                        &ctx,
                        stream,
                    )?;
                    if !used_wy16 {
                    // Phase 2: one WY4 batch GDN recurrence over all K tokens.
                    layer.prefill_gdn_full(
                        seq.layer_states[layer_idx].as_mut(),
                        &gdn_bufs,
                        &ctx,
                        stream,
                    )?;
                    // Fill h_state_intermediates via per-token replay from checkpoint (R7).
                    // Extract DevicePtr copies before the replay loop to avoid borrow
                    // conflicts with the trait method calls inside the loop.
                    let (h_state_ptr, h_ckpt_opt, h_ints_vec) = {
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        (
                            s.h_state,
                            s.h_state_checkpoint,
                            s.h_state_intermediates.clone(),
                        )
                    };
                    let h_bytes = self.ssm_pool.h_bytes;
                    // Save WY4 final h_state to scratch
                    self.gpu.copy_d2d_async(
                        h_state_ptr,
                        self.ssm_verify_h_tmp,
                        h_bytes,
                        stream,
                    )?;
                    // Restore pre-verify checkpoint so replay starts from correct state
                    if let Some(ckpt) = h_ckpt_opt {
                        self.gpu.copy_d2d_async(ckpt, h_state_ptr, h_bytes, stream)?;
                    }
                    // K serial single-token GDN steps; after each step h_state_ptr
                    // device memory holds h_state after tokens 0..=t.
                    for t in 0..k.min(h_ints_vec.len()) {
                        let tok_gdn = GdnPrefillBuffers {
                            qkv:      gdn_bufs.qkv.offset(t * ssm_conv_dim * bf16),
                            gate_beta: gdn_bufs.gate_beta.offset(t * ssm_gate_stride * 4),
                            output:   gdn_bufs.output.offset(t * ssm_value_dim * bf16),
                            z:        gdn_bufs.z.offset(t * ssm_value_dim * bf16),
                            total_len: 1,
                        };
                        layer.prefill_gdn_full(
                            seq.layer_states[layer_idx].as_mut(),
                            &tok_gdn,
                            &ctx,
                            stream,
                        )?;
                        self.gpu.copy_d2d_async(
                            h_state_ptr,
                            h_ints_vec[t],
                            h_bytes,
                            stream,
                        )?;
                    }
                    // Restore WY4 final h_state
                    self.gpu.copy_d2d_async(
                        self.ssm_verify_h_tmp,
                        h_state_ptr,
                        h_bytes,
                        stream,
                    )?;
                    } // end !used_wy16 (WY4 batch + per-token replay fallback)
                    if let Some(t) = _ts2 {
                        self.gpu.synchronize(stream)?;
                        us_p2 += t.elapsed().as_micros();
                    }
                    // Phase 3: gated RMS norm + output projection + FFN
                    let _ts3 = if timing {
                        self.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    layer.prefill_phase3(
                        hidden,
                        residual,
                        k,
                        &gdn_bufs,
                        0,
                        &ctx,
                        stream,
                    )?;
                    if let Some(t) = _ts3 {
                        self.gpu.synchronize(stream)?;
                        us_p3 += t.elapsed().as_micros();
                    }
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }

                // DFlash per-layer hidden capture into dflash_hidden_save.
                // EAGLE-fix: capture all k verify rows (row-major) so the
                // scheduler can append one ctx slot per accepted position.
                // Legacy: capture only row 0 (last_token), appended via propose.
                if eagle_fix {
                    self.try_dflash_capture_all(layer_idx, k, stream)?;
                } else {
                    self.try_dflash_capture(layer_idx, 0, stream)?;
                }
            }

            // Final norm [K, H]
            let _tsh = if timing {
                self.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                k as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head for K tokens
            self.lm_head_batched(normed, k as u32, stream)?;

            // Argmax inside graph (fixed scratch addresses — graph-safe)
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            for t in 0..k {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }
            if let Some(t) = _tsh {
                self.gpu.synchronize(stream)?;
                us_head += t.elapsed().as_micros();
            }

            // ── ATLAS_DFLASH_TIMING: aggregate per-phase costs, log every 50 steps ──
            if timing {
                use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
                static A: AtomicU64 = AtomicU64::new(0);
                static P1: AtomicU64 = AtomicU64::new(0);
                static P2: AtomicU64 = AtomicU64::new(0);
                static P3: AtomicU64 = AtomicU64::new(0);
                static HD: AtomicU64 = AtomicU64::new(0);
                static N: AtomicU64 = AtomicU64::new(0);
                A.fetch_add(us_attn as u64, Relaxed);
                P1.fetch_add(us_p1 as u64, Relaxed);
                P2.fetch_add(us_p2 as u64, Relaxed);
                P3.fetch_add(us_p3 as u64, Relaxed);
                HD.fetch_add(us_head as u64, Relaxed);
                let n = N.fetch_add(1, Relaxed) + 1;
                if n % 50 == 0 {
                    let ms = |x: &AtomicU64| x.load(Relaxed) as f64 / n as f64 / 1000.0;
                    let (a, p1, p2, p3, hd) = (ms(&A), ms(&P1), ms(&P2), ms(&P3), ms(&HD));
                    let sum = (a + p1 + p2 + p3 + hd).max(1e-6);
                    tracing::info!(
                        "DFLASH TIMING avg/{n} steps: attn={a:.1} p1_conv={p1:.1} p2_gdn={p2:.1} p3={p3:.1} head={hd:.1} | sum={sum:.1}ms (attn {:.0}% p1 {:.0}% p2 {:.0}% p3 {:.0}% head {:.0}%)",
                        100.0 * a / sum,
                        100.0 * p1 / sum,
                        100.0 * p2 / sum,
                        100.0 * p3 / sum,
                        100.0 * hd / sum,
                    );
                }
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
