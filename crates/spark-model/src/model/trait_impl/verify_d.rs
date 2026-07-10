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
        // See `build_kgamma_attn_metadata` for the decode-vs-prefill layout.
        let use_prefill_attn =
            std::env::var("ATLAS_VERIFY_PREFILL_ATTN").ok().as_deref() == Some("1");
        let metadata = self.build_kgamma_attn_metadata(k, seq, bs, use_prefill_attn, stream)?;

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
        let allow_kgamma_graph = std::env::var("ATLAS_DFLASH_UNSAFE_KGAMMA_GRAPH")
            .ok()
            .as_deref()
            == Some("1");
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
            let timing =
                (std::env::var("ATLAS_DFLASH_TIMING").ok().as_deref() == Some("1")) && !use_graphs;
            if std::env::var("ATLAS_DFLASH_TIMING").ok().as_deref() == Some("1") && use_graphs {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "ATLAS_DFLASH_TIMING ignored: needs eager K=γ (incompatible with graph capture)."
                    );
                }
            }
            let mut us_head: u128 = 0; // final norm + lm_head + argmax

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // SSM three-phase parallel verify (ATLAS_VERIFY_PREFILL_SSM=1).
            // When enabled, SSM layers use prefill_phase1/gdn_full/phase3
            // instead of sequential decode_batched(k), reducing GDN work from
            // K serial h_state updates to one WY4 batch recurrence.
            let use_prefill_ssm =
                std::env::var("ATLAS_VERIFY_PREFILL_SSM").ok().as_deref() == Some("1");
            // EAGLE-fix (K=γ): capture ALL k verify rows so the scheduler can
            // append rows 0..=num_accepted to ctx (fixes ctx-undercount + EAGLE
            // shift). Flag off → legacy single row-0 capture.
            let eagle_fix = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
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
                qkv: self.gdn_buf_qkv,
                gate_beta: self.gdn_buf_gate_beta,
                output: self.gdn_buf_out,
                output_f32: self.gdn_buf_out_f32,
                output_f32_written: std::cell::Cell::new(false),
                z: self.gdn_buf_z,
                total_len: k,
            };

            let timing_res = self.run_kgamma_verify_layers(
                seq,
                &mut kv_cache,
                hidden,
                residual,
                k,
                h,
                bf16,
                hss_engaged,
                use_prefill_attn,
                use_prefill_ssm,
                eagle_fix,
                timing,
                &seq_lens_vec,
                &block_tables_vec,
                &gdn_bufs,
                ssm_conv_dim,
                ssm_gate_stride,
                ssm_value_dim,
                &ctx,
                stream,
            )?;
            let us_attn = timing_res.us_attn;
            let us_p1 = timing_res.us_p1;
            let us_p2 = timing_res.us_p2;
            let us_p3 = timing_res.us_p3;

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
            self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

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
            // Iteration-4 KL-gate diagnostic prototype
            // (`ATLAS_DFLASH_KL_DIAG=1`, default off, exploratory only —
            // NOT a production gate). Self-confidence proxy computed from
            // the SAME batched-verify logits already produced above (no
            // second serial-decode forward pass): per verified position,
            // top-1 vs top-2 margin in logit space (`= ln(p1/p2)` exactly,
            // softmax-normalization-invariant) and top-1 probability via a
            // full-vocab logsumexp. Requires eager (D2H + host loop is
            // illegal under CUDA graph capture); no-ops when `use_graphs`.
            // Only viable with `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` in the env
            // (forces `use_graphs=false` via `force_eager` above).
            if !use_graphs && std::env::var("ATLAS_DFLASH_KL_DIAG").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let mut row = vec![0u8; vocab * bf16];
                for t in 0..k {
                    let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                    self.gpu.copy_d2h(logits_t, &mut row)?;

                    let mut top1_v = f32::NEG_INFINITY;
                    let mut top1_i = 0u32;
                    let mut top2_v = f32::NEG_INFINITY;
                    for vi in 0..vocab {
                        let val = kl_diag_bf16_to_f32(row[vi * 2], row[vi * 2 + 1]);
                        if val > top1_v {
                            top2_v = top1_v;
                            top1_v = val;
                            top1_i = vi as u32;
                        } else if val > top2_v {
                            top2_v = val;
                        }
                    }
                    let mut sum_exp = 0.0f64;
                    for vi in 0..vocab {
                        let val = kl_diag_bf16_to_f32(row[vi * 2], row[vi * 2 + 1]);
                        sum_exp += ((val - top1_v) as f64).exp();
                    }
                    let lse = top1_v as f64 + sum_exp.ln();
                    let p_top1 = ((top1_v as f64) - lse).exp();
                    let margin = top1_v - top2_v;

                    let pos = seq.seq_len + t;
                    let expected = if t + 1 < k { Some(tokens[t + 1]) } else { None };
                    let matched = expected == Some(top1_i);
                    tracing::info!(
                        "KL_DIAG pos={pos} margin={margin:.4} p_top1={p_top1:.4} \
                         argmax={top1_i} expected={expected:?} matched={matched}"
                    );
                }
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
                if n.is_multiple_of(50) {
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

        // Host-side glue — `copy_d2h` internally issues cuMemcpyDtoHAsync_v2
        // on the default stream followed by a blocking cuStreamSynchronize
        // (see cuda_backend/gpu_impl.rs). This is the ONLY unconditional
        // sync/D2H pair on the shipping K=γ verify path (all other
        // sync/copy_d2h call sites in this file are gated behind
        // ATLAS_DFLASH_TIMING/ATLAS_DFLASH_KL_DIAG, both off by default) —
        // reads back the K argmax token ids so the scheduler can run
        // partial-accept comparison on the host.
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

/// KL-gate diagnostic helper (`ATLAS_DFLASH_KL_DIAG=1`).
fn kl_diag_bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits16 = (lo as u16) | ((hi as u16) << 8);
    f32::from_bits((bits16 as u32) << 16)
}
