// SPDX-License-Identifier: AGPL-3.0-only

//! Batched K-row verify: n sequences × `ks[i]` rows in ONE eager forward.
//!
//! Generalizes verify_c2's single-sequence K=4 body to `R = Σ ks` seq-major
//! rows (sequence i at rows `[off_i, off_i + ks[i])`, `ks[i]` = its drafts+1 in
//! 2..=4, chosen per step by the K-vs-batch ladder — `speculative::ladder` —
//! and made RAGGED per sequence by D-Cut) so the n weight-reading verify
//! forwards collapse into one — the structural fix for the measured MTP
//! serialization at C>1 (cap=4 at C=4: 25.8 vs 48.5 tok/s; see
//! BATCHED_MTP_SPEC.md). R is capped at 96 = the exact capacity of the
//! logits rows / meta gaps / bt staging (sizes.rs), reached at n=32 × k=3
//! rows (the 32:2 depth-at-width shape, wave 11; previously 64 at n=32 ×
//! k=2, 32 at n=16 × k=2).
//!
//! CUDA GRAPHS (slot-VECTOR-keyed): the forward span (layer loop + final
//! norm + lm_head + argmax) is captured per distinct ssm-slot vector — the
//! slot-keyed `verify4_graph` cache is meaningless at n>1 because a graph
//! bakes EVERY sequence's state pointers, which are a function of the whole
//! vector. Pre-graph each step (embed, KV-block ensure, metadata/bt/WY-table
//! H2D into fixed addresses) and the argmax D2H stay eager — exactly the
//! decode_a2 padded_n-graph pattern. Kill switch `ATLAS_NO_MTP_VERIFY_GRAPHS`
//! (PRESENCE). Everything per-sequence (GDN conv+WY4 body, block tables,
//! rollback intermediates) reuses existing machinery verbatim — only base
//! addresses move — with an optional cross-sequence batched conv+WY fast
//! path in the GDN Multi arm (see trait_decode_batched_conv_gdn_multi.rs).
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
    /// Whether the batched verify can run for `ks.len()` sequences at `ks[i]`
    /// rows each (ragged since D-Cut).
    ///
    /// Self-gates to the envelope verify_e was built and audited for:
    /// non-EP, non-HSS, non-DFlash, no LoRA (the uniform seq_slot upload
    /// carries ONE adapter slot), MTP proposer present (stash allocated,
    /// `VERIFY_WY_TABLE_SEQS` = 32 slots ⇒ n ≤ 32), EVERY `ks[i]` in 2..=4
    /// (the MTP ladder range; intermediates pools are sized for the
    /// configured max K), and R = Σ ks ≤ `VERIFY_ROW_CAP` = 96 (the exact
    /// logits-rows / meta-gap / bt-staging capacity — sizes.rs). Everything
    /// outside falls back to the per-seq loop.
    pub(super) fn can_batch_verify_dispatch(&self, ks: &[usize]) -> bool {
        let n = ks.len();
        (2..=crate::layer::VERIFY_WY_TABLE_SEQS).contains(&n)
            && ks.iter().all(|k| (2..=4).contains(k))
            && ks.iter().sum::<usize>() <= super::verify_e2::VERIFY_ROW_CAP
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

    /// Batched K-row verify for `n = seqs.len()` sequences (R = Σ ks rows,
    /// each `ks[i]` = that sequence's drafts+1 in 2..=4, ragged since D-Cut).
    ///
    /// Row `off_i + j` is sequence i's token j (its slice of `tokens` is
    /// `[last_verified, d0, .., d_{ks[i]-2}]`, flat seq-major). Weight-bearing
    /// ops (QKVZ/out_proj/FFN/lm_head) batch across all R rows via the
    /// existing M-generic arms; attention runs through `decode_multi_seq`
    /// with per-row block tables / seq lens; the GDN conv+WY body runs
    /// through `decode_verify_multi`, whose two-launch cross-sequence fast
    /// path now engages at EVERY ladder width k in 2..=4
    /// (trait_decode_batched_conv_gdn_multi.rs), falling back to the
    /// byte-identical per-sequence loop only when the slot layout is
    /// fragmented.
    ///
    /// On success: per seq `tokens` += ks[i], `seq_len` += ks[i] (rewind is
    /// the caller's arithmetic, as on the per-seq path). On Err no sequence
    /// state has been advanced. Logits rows stay live for row-based
    /// pipeline picks until the next forward — callers must consume them
    /// (and stash hiddens) BEFORE any propose.
    pub(super) fn decode_verify_batched_dispatch(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let t_launch = std::time::Instant::now();
        let mapped_argmax = mapped_argmax_host_dev(self.gpu.as_ref());
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n = seqs.len();
        // Row offsets: sequence i owns rows [off[i], off[i+1]). RAGGED since
        // D-Cut; `off[i] = i*k` is the uniform special case.
        let mut off: Vec<usize> = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        for &k in ks {
            off.push(acc);
            acc += k;
        }
        off.push(acc);
        let r_total = acc;
        let k_max = ks.iter().copied().max().unwrap_or(0);
        ensure!(
            n >= 2
                && ks.len() == n
                && ks.iter().all(|k| (2..=4).contains(k))
                && tokens.len() == r_total,
            "batched verify: n={n} ks={ks:?} tokens={}",
            tokens.len()
        );
        // R ≤ VERIFY_ROW_CAP (96): the exact capacity of the meta gaps below
        // (positions 384 B at +0, seq_slot at +384, slots 768 B at +768,
        // seq_lens 384 B at +1536, bt at +2048 staged for 96 rows in
        // sizes.rs) and the 96-row logits cap. Reached at n=32 × k=3 rows
        // (the 32:2 depth-at-width shape).
        ensure!(
            r_total <= super::verify_e2::VERIFY_ROW_CAP,
            "batched verify: R={r_total} exceeds the {}-row buffer capacity",
            super::verify_e2::VERIFY_ROW_CAP
        );

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: embed R tokens + allocate KV blocks ──
        for (r, &t) in tokens.iter().enumerate() {
            self.embed(t, hidden.offset(r * h * bf16), stream)?;
        }

        let bs = kv_cache.block_size();
        for (i, seq) in seqs.iter_mut().enumerate() {
            let last_pos = seq.seq_len + ks[i] - 1;
            ensure_blocks_through_decode(
                seq,
                last_pos / bs,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
        }

        // ATLAS_K4_DIAG=1: stream-sync checkpoint after every layer so an
        // illegal access is attributed to the exact layer (same hatch as
        // verify_c2). Forces EAGER — per-layer syncs are illegal under
        // capture (verify_c2's gate pattern).
        let k4_diag = super::verify_e2::k4_diag_enabled();

        // Pre-graph: stage the per-GDN-layer WY pointer tables into the
        // fixed staging buffer (contents refreshed BEFORE any replay, like
        // the attention metadata below). NULL → per-seq WY loop. Staged for
        // EVERY ladder width now that wy2/wy3 carry the same `state_is_table`
        // pointer-table form as wy4: at k<4 the fast path used to decline
        // into the per-seq conv/WY loop (n launches per layer instead of 2),
        // which is exactly the k<4 verify-step cost the n=16 matrix measured.
        // Staged at the batch's DEEPEST width: a sequence pruned to fewer rows
        // simply leaves its tail slabs unread (the WY launch for its depth
        // reads `k-1` intermediate tables), and the strides are k-independent.
        let wy_tables_base = self.upload_verify_wy_tables(&*seqs, k_max, &[], stream)?;

        // ── Graph decision: exact slot-vector hit, or drain-tail borrow ──
        // Keyed by the ssm-slot VECTOR (verify_e2.rs): every baked SSM
        // pointer is a function of it; meta/embeds live at fixed addresses
        // refreshed below. can_batch already excludes EP/HSS/LoRA/DFlash.
        // On an exact miss, `graph_borrow.rs` may pick a WIDER captured key
        // whose (slot, k) pairs start with this batch's — its active rows
        // are then exactly the scheduler's `off[i]` rows, and the baked tail
        // pairs become GHOST rows this step must feed (pad metadata, pad
        // embeds, synthesized WY entries). Tail safety: each ghost slot is
        // currently free (pad writes land on unowned pool state, zeroed
        // again at the next claim) and its tiered intermediate pool covers
        // the baked depth.
        let graphs_on = super::verify_e2::verify_graphs_enabled() && !k4_diag;
        let graph_key = if graphs_on {
            self.verify_batched_graph_key(&*seqs, ks, wy_tables_base.is_null())
        } else {
            None
        };
        let mut graphs = graph_key
            .as_ref()
            .map(|_| self.verify_batched_graphs.lock());
        // LRU touch on hit: bump the tick so eviction always removes the
        // least-recently-replayed slot vector.
        let mut replay: Option<spark_runtime::gpu::GraphHandle> = None;
        let mut ghosts: Vec<(u32, u32)> = Vec::new();
        // Graph outcome for the periodic ATLAS_MTP_ACCEPT_DEBUG summary
        // (verify_e2). `Eager` until something claims otherwise — that is
        // also the honest value when graphs are off or the batch is
        // unkeyable.
        let mut outcome = super::verify_e2::VerifyGraphOutcome::Eager;
        if let (Some(g), Some(key)) = (&mut graphs, &graph_key) {
            g.1 += 1;
            let tick = g.1;
            if let Some(e) = g.0.get_mut(key) {
                e.1 = tick;
                replay = Some(e.0);
                outcome = super::verify_e2::VerifyGraphOutcome::Replay;
            } else if super::graph_borrow::graph_borrow_enabled() {
                let wy_present = !wy_tables_base.is_null();
                let borrowed =
                    super::graph_borrow::find_borrowable_verify_key(key, g.0.keys(), |s, k| {
                        self.ssm_pool.slot_is_free(s as usize)
                            && (!wy_present
                                || self.ssm_pool.h_inter_count(s as usize) + 1 >= k as usize)
                    });
                // Every cached key was captured under `ensure!(R <= 96)`, so
                // the borrowed total row count fits the fixed 96-row meta
                // arrays and logits cap by construction — but that bound
                // guards the `unsafe` upload lengths below, so it is
                // re-checked as a hard borrow veto, never assumed.
                if let Some(b) = borrowed
                    && r_total + b.ghosts.iter().map(|&(_, k)| k as usize).sum::<usize>()
                        <= super::verify_e2::VERIFY_ROW_CAP
                {
                    let e =
                        g.0.get_mut(&b.key)
                            .expect("borrowed key comes from this cache");
                    e.1 = tick;
                    replay = Some(e.0);
                    ghosts = b.ghosts;
                    outcome = super::verify_e2::VerifyGraphOutcome::Borrow;
                    // INFO once per transition (same cardinality as the
                    // captures this replaces); repeats of the same pair
                    // stay silent. Provable engagement: grep "graph borrow".
                    if super::graph_borrow::VERIFY_BORROW_LOG.should_log(key, &b.key) {
                        tracing::info!(
                            "verify graph borrow: n={n} R={r_total} -> replaying captured \
                             {}-seq key with {} ghost pairs",
                            (b.key.len() - 1) / 2,
                            ghosts.len()
                        );
                    }
                }
            }
        }
        // Rows the DISPATCH must prepare: active rows plus any ghost tail.
        // `r_up <= VERIFY_ROW_CAP` (96) holds on every path: the no-ghost
        // case by the `ensure!` above, the borrow case by the veto at accept.
        let r_ghost: usize = ghosts.iter().map(|&(_, k)| k as usize).sum();
        let r_up = r_total + r_ghost;
        if !ghosts.is_empty() {
            // Ghost embeds: a real token's embedding (0) keeps pad lanes on
            // finite values; their outputs are never read.
            for r in r_total..r_up {
                self.embed(0, hidden.offset(r * h * bf16), stream)?;
            }
            // Re-stage the WY tables with the ghost entries appended —
            // synthesized from the pool, since a captured entry is a pure
            // function of (layer, slot).
            let k_ghost = ghosts.iter().map(|&(_, k)| k as usize).max().unwrap_or(0);
            let restaged =
                self.upload_verify_wy_tables(&*seqs, k_max.max(k_ghost), &ghosts, stream)?;
            // Fail fast, never silently: a presence flip would replay the
            // graph against tables missing its baked ghost entries.
            ensure!(
                restaged.is_null() == wy_tables_base.is_null(),
                "verify graph borrow: ghost WY restage flipped table presence"
            );
        }

        // ── Phase 2: R-row attention metadata (verify_c2 layout SHAPE at
        // WIDER gaps — 96 rows: positions [0,384) | seq_slot [384,768) |
        // slots i64 [768,1536) | seq_lens [1536,1920) | bt at +2048. This
        // path's own layout only: every metadata consumer receives absolute
        // pointers via `AttnMetadataDev`, and each step (and each graph's
        // replay) re-uploads its own layout pre-dispatch, so verify_c2 /
        // decode_a2 keeping the narrow 32-row gaps is not a conflict. ──
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;

        let mut positions = [0u32; 96];
        let mut slots = [0i64; 96];
        let mut seq_lens = [0i32; 96];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..ks[i] {
                let r = off[i] + j;
                let pos = seq.seq_len + j;
                positions[r] = pos as u32;
                let physical_block = seq.physical_block_for(pos / bs).unwrap_or(0);
                slots[r] = (physical_block as i64) * (bs as i64) + ((pos % bs) as i64);
                // Per-row causal clamp: row r attends through its own position.
                seq_lens[r] = (pos + 1) as i32;
            }
        }
        // Ghost tail rows (borrow replay only): pad metadata, exactly the
        // decode_a2 padding shape — position 0, the always-safe dummy KV
        // block, causal clamp 1. Their SSM lanes are handled by the baked
        // pool addresses + the restaged WY tables; nothing here may point at
        // a live sequence.
        let dummy_kv = (self.dummy_kv_block as i64) * (bs as i64);
        for r in r_total..r_up {
            positions[r] = 0;
            slots[r] = dummy_kv;
            seq_lens[r] = 1;
        }
        // SAFETY: `positions` is the fixed `[0u32; 96]` above, so its size is
        // 96 * 4 = 384 B; the `ensure!(r_total <= VERIFY_ROW_CAP)` guard plus
        // the `debug_assert!(r_up <= VERIFY_ROW_CAP)` (r_up rows were
        // captured under the same cap) make `r_up * 4 <= 384`.
        // The array is zero-init at declaration and rows `0..r_up` are all
        // written by the fill loops (`off` is the prefix sum of `ks`, so
        // `off[i]+j` covers `0..r_total` exactly; the ghost loop covers
        // `r_total..r_up`). `u32` is POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, r_up * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;
        // SAFETY: `slots` is the fixed `[0i64; 96]` above (768 B); the same
        // bounds argument gives `r_up * 8 <= 768`. Zero-init at declaration,
        // rows `0..r_up` written by the fill loops; `i64` is POD.
        let slot_bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, r_up * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(768), stream)?;
        // SAFETY: `seq_lens` is the fixed `[0i32; 96]` above (384 B); the same
        // bounds argument gives `r_up * 4 <= 384`. Zero-init at declaration,
        // rows `0..r_up` written by the fill loops; `i32` is POD.
        let sl_bytes =
            unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, r_up * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(1536), stream)?;

        // Block tables: row r = seq i's table (bt staging sized for 96 rows,
        // sizes.rs `bt_rows`). Ghost rows read only entry 0 (causal clamp 1)
        // — point it at the dummy KV block, matching decode_a2's pad rows.
        let needed = r_up * mb;
        let mut bt_buf = vec![0i32; needed];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..ks[i] {
                let row = off[i] + j;
                for (bi, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + bi] = block as i32;
                }
            }
        }
        for row in r_total..r_up {
            bt_buf[row * mb] = self.dummy_kv_block as i32;
        }
        // SAFETY: `bt_buf` is `vec![0i32; needed]` on the line above, so its
        // LEN is `needed` and `needed * 4 == size_of_val(&bt_buf[..])` — the
        // read stops at `len`, never in the `Vec`'s spare capacity. Zero-init
        // at construction covers the rows/columns the fill loop skips when
        // `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(2048), stream)?;

        // No-LoRA gate in can_batch: uniform upload returns DevicePtr(0)
        // (installed-pair path) — kept for structural parity with verify_c2.
        debug_assert!(
            r_up <= super::verify_e2::VERIFY_ROW_CAP,
            "verify seq_slot [384,768) gap holds R ≤ 96"
        );
        let seq_slot = self.upload_seq_slot_uniform(
            seqs[0].adapter_slot,
            r_up,
            meta_base.offset(384),
            stream,
        )?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(768),
            seq_len: meta_base.offset(1536),
            block_table: meta_base.offset(2048),
            max_blocks_per_seq: max_blocks,
            num_seqs: r_up as u32,
            seq_slot,
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };

        // ── Phase 3: CUDA graph replay, or capture/eager forward ──
        // The graph decision (exact hit / drain-tail borrow) ran above,
        // before the metadata fill, so ghost rows were prepared with the
        // rest of this step's fixed-address refresh.
        if let Some(graph) = replay {
            // Replay: kernels read this step's metadata + WY tables from the
            // fixed addresses refreshed above; the ~4-5k launches of the
            // layer loop + head + argmax dispatch as one graph.
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
        } else {
            // First step for this (slot vector, k) key (or graphs off): run
            // the body, capturing. A full cache no longer disables capture —
            // the LRU entry is destroyed at insert time (see below), so
            // slot-vector churn can never push the path permanently eager.
            let capture = graphs.is_some();

            let ctx = ForwardContext {
                buffers: &self.buffers,
                hc_row_offset: 0,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                dispatch: &self.dispatch,
                // Route-aware v0: base (Skip) proceeds free; an active adapter is
                // rejected before the fold on these multi-seq/speculative paths
                // (reject_decode_lora), so Fold is inert here.
                moe_lora_route: self.decode_moe_route(),
                derived: &self.derived,
                levers: &self.levers,
                stats: &self.stats,
                attn_metadata: Some(metadata),
                profile: false,
                comm: self.comm_ref(),
                graph_capture: capture,
                gdn_exact_replay: false,
                token_ids: None,
                host_token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
            };

            // Host-side per-row attention args (verify_c2 pattern, R rows).
            let mut seq_lens_vec: Vec<usize> = Vec::with_capacity(r_total);
            let mut block_tables_vec: Vec<Vec<u32>> = Vec::with_capacity(r_total);
            for (i, seq) in seqs.iter().enumerate() {
                for j in 0..ks[i] {
                    seq_lens_vec.push(seq.seq_len + j);
                    block_tables_vec.push(seq.block_table.clone());
                }
            }

            // Dummy attention states are stateless (multi_seq attention
            // ignores them) — allocated OUTSIDE the capture window.
            let mut attn_dummy_states: Vec<Vec<Box<dyn LayerState>>> = Vec::new();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(layer_idx) == LayerType::FullAttention {
                    attn_dummy_states.push(
                        (0..r_total)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?,
                    );
                }
            }

            if capture {
                self.gpu.begin_capture(stream)?;
            }

            let mut attn_idx = 0usize;
            let mut ssm_idx = 0usize;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    let mut refs: Vec<&mut (dyn LayerState + 'static)> = attn_dummy_states
                        [attn_idx]
                        .iter_mut()
                        .map(|s| s.as_mut())
                        .collect();
                    attn_idx += 1;
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
                    let mut wy_slice = DevicePtr::NULL;
                    if layer_type == LayerType::LinearAttention {
                        if !wy_tables_base.is_null() {
                            wy_slice = wy_tables_base
                                .offset(ssm_idx * crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES);
                        }
                        ssm_idx += 1;
                    }
                    let mut state_refs: Vec<&mut (dyn LayerState + 'static)> = seqs
                        .iter_mut()
                        .map(|s| s.layer_states[layer_idx].as_mut())
                        .collect();
                    layer.decode_verify_multi(
                        hidden,
                        residual,
                        n,
                        ks,
                        &mut state_refs,
                        &mut kv_cache,
                        wy_slice,
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
            self.final_norm_apply(
                hidden,
                normed,
                r_total as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!("K4_DIAG(batched): CUDA error after final norm: {e:#}");
            }

            // R ≤ VERIFY_ROW_CAP = the 96-row logits buffer cap (sizes.rs).
            self.lm_head_batched(normed, r_total as u32, self.buffers.logits(), stream)?;

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!("K4_DIAG(batched): CUDA error after lm_head_batched: {e:#}");
            }

            let vocab = self.config.vocab_size;
            // MAPPED-ARGMAX (2026-07-30, the 130 ms stall fix that finally
            // held): the argmax kernel writes its 4 B/row results DIRECTLY to
            // page-locked host-mapped memory (UMA device alias), so the step
            // ends with a kernel, not a copy-engine op. Traced root cause
            // (PROGRESS_LOG 6.16): a 128-320 B tail-of-queue DtoH sat ~137 ms
            // on a FREE copy engine before being picked up (1.8 us to execute
            // once it ran) — invariant to pageable/pinned/stream/spin/graph/
            // keep-awake arms, all buried by experiment. No copy op, no
            // pickup. Kill switch ATLAS_NO_MAPPED_ARGMAX=1 restores scratch +
            // on-stream copy. The mapped blob is allocated once (before the
            // first graph capture, so replays bake the same fixed address).
            let argmax_out = match mapped_argmax {
                Some((_, dev)) => dev,
                None => self.buffers.scratch(),
            };
            // ONE launch, one block per row (single-row argmax is a one-CTA
            // scan; R serial calls = R single-SM scans, ~100 us each at this
            // vocab). Byte-identical per-row body; loop fallback when the
            // kernel is absent.
            if self.argmax_batch_kernel.0 != 0 {
                ops::argmax_bf16_batch(
                    self.gpu.as_ref(),
                    self.argmax_batch_kernel,
                    self.buffers.logits(),
                    argmax_out,
                    vocab as u32,
                    r_total as u32,
                    vocab as u32,
                    stream,
                )?;
            } else {
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
            }

            if capture {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for batched verify ks={ks:?} (n={n}, key={:?})",
                        graph_key
                    );
                    if let (Some(ref mut g), Some(key)) = (graphs.as_mut(), graph_key) {
                        if g.0.len() >= super::verify_e2::VERIFY_BATCHED_GRAPH_CAP {
                            // Evict the least-recently-used graph. Safe to
                            // destroy: every batched verify step ends with a
                            // blocking argmax D2H on this stream, so any
                            // earlier step's replay has already completed.
                            if let Some(evict) =
                                g.0.iter()
                                    .min_by_key(|(_, entry)| entry.1)
                                    .map(|(key, _)| key.clone())
                                && let Some((old, _)) = g.0.remove(&evict)
                                && let Err(e) = self.gpu.destroy_graph(old)
                            {
                                tracing::warn!("batched-verify graph evict: {e:#}");
                            }
                        }
                        g.1 += 1;
                        let tick = g.1;
                        g.0.insert(key, (graph, tick));
                        outcome = super::verify_e2::VerifyGraphOutcome::Capture;
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }
        // Live key count read while the guard is still held — it is the
        // other half of the capture-rate signal (churn against the 32-entry
        // LRU is what turns a miss into a re-capture).
        let live_keys = graphs.as_ref().map(|g| g.0.len()).unwrap_or(0);
        drop(graphs);
        super::verify_e2::record_verify_graph_outcome(n, live_keys, outcome);

        // ── Phase 5: D2H + host bookkeeping ──
        // Argmax landed at scratch row 0 (graph replay and eager both write
        // the same fixed address). Blocking D2H = the step's one host sync.
        // ATLAS_MTP_TIMING attribution (2026-07-30): everything above this
        // line is the LAUNCH region (host-side dispatch + graph replay,
        // recorded as Argmax); the copy below is the wait-for-GPU + copy
        // (recorded as D2h). Splits the ~127 ms fwd cost between "host
        // launching work" and "host waiting on the stream".
        let t_d2h = std::time::Instant::now();
        let launch_us = t_d2h.duration_since(t_launch).as_micros() as u64;
        let mut buf = vec![0u8; r_total * 4];
        let mut filled = false;
        if let Some((host, _)) = mapped_argmax {
            // Kernel-written host-mapped results: one stream sync (kernels
            // only — no copy op in the queue), then read host memory.
            self.gpu.synchronize(stream)?;
            // SAFETY: `host` is the 65_536 B `alloc_host_pinned` blob from
            // `mapped_argmax_host_dev`, live for the process lifetime; the
            // `ensure!(r_total <= VERIFY_ROW_CAP)` guard (cap 96) bounds
            // `r_total * 4 <= 384`. Those exact bytes were INITIALISED by the
            // argmax dispatch, which wrote one 4 B row each for `0..r_total`
            // through this blob's UMA device alias, and the `synchronize`
            // above ordered those writes before this read.
            let src = unsafe { std::slice::from_raw_parts(host, r_total * 4) };
            buf.copy_from_slice(src);
            filled = true;
        }
        // PINNED on-stream D2H (2026-07-30, the 130 ms stall fix). Traced
        // (PROGRESS_LOG 6.15/6.16): with a PAGEABLE destination the
        // cuMemcpyDtoHAsync call itself blocked ~137 ms and the ALREADY
        // LAUNCHED verify graph did not begin executing until the call
        // returned — GPU frozen for the duration, then graph + queued D2Ds
        // ripped through in one dense burst. A page-locked destination takes
        // the true async path: enqueue returns immediately, the stream runs
        // the ~11 ms graph, and the sync waits only for that. The buffer is
        // a lazily-allocated 64 KB pinned blob (r_total <= 32 rows x 4 B
        // needs 128 B; headroom for future wider verifies), reused for the
        // process lifetime — the scheduler thread is the only caller.
        // Kill switches: ATLAS_NO_PINNED_VERIFY_D2H=1 -> pageable on-stream;
        // ATLAS_VERIFY_D2H_DEFAULT_STREAM=1 -> the original default-stream arm.
        if filled {
            // mapped path already read the results — no copy arm runs.
        } else if super::verify_e2::verify_d2h_default_stream() {
            self.gpu.copy_d2h(self.buffers.scratch(), &mut buf)?;
        } else if super::verify_e2::verify_d2h_no_pinned() {
            self.gpu
                .copy_d2h_on_stream(self.buffers.scratch(), &mut buf, stream)?;
        } else {
            use std::sync::atomic::{AtomicPtr, Ordering};
            const PINNED_CAP: usize = 65_536;
            static PINNED: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
            let mut p = PINNED.load(Ordering::Acquire);
            if p.is_null() {
                p = self.gpu.alloc_host_pinned(PINNED_CAP)?;
                // Single-threaded caller (scheduler); store unconditionally.
                PINNED.store(p, Ordering::Release);
            }
            if buf.len() <= PINNED_CAP {
                // SAFETY: PINNED points at a live cuMemAllocHost blob of
                // PINNED_CAP bytes and the enclosing `if buf.len() <=
                // PINNED_CAP` is the length bound, so the `&mut [u8]` stays
                // inside the allocation. `u8` has no validity invariant, so
                // aliasing uninitialised pinned bytes as `&mut [u8]` is sound
                // to WRITE; the `copy_d2h_on_stream` below fills all
                // `buf.len()` bytes before `copy_from_slice` reads them. The
                // scheduler thread is the sole user, and the slice does not
                // outlive this block.
                let dst = unsafe { std::slice::from_raw_parts_mut(p, buf.len()) };
                self.gpu
                    .copy_d2h_on_stream(self.buffers.scratch(), dst, stream)?;
                buf.copy_from_slice(dst);
            } else {
                self.gpu
                    .copy_d2h_on_stream(self.buffers.scratch(), &mut buf, stream)?;
            }
        }
        {
            // Local fwd-split telemetry (ATLAS_MTP_TIMING=1): launch region vs
            // the blocking argmax D2H. Lives here because mtp_timing is a
            // spark-server module. One INFO line per 100 batched verifies.
            use std::sync::atomic::{AtomicU64, Ordering};
            static LAUNCH_US: AtomicU64 = AtomicU64::new(0);
            static D2H_US: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *ON.get_or_init(|| std::env::var("ATLAS_MTP_TIMING").as_deref() == Ok("1")) {
                let d2h_us = t_d2h.elapsed().as_micros() as u64;
                LAUNCH_US.fetch_add(launch_us, Ordering::Relaxed);
                D2H_US.fetch_add(d2h_us, Ordering::Relaxed);
                let n_done = N.fetch_add(1, Ordering::Relaxed) + 1;
                if n_done.is_multiple_of(100) {
                    let l = LAUNCH_US.swap(0, Ordering::Relaxed);
                    let d = D2H_US.swap(0, Ordering::Relaxed);
                    tracing::info!(
                        "batched-verify fwd split [100 calls]: launch={:.2}ms d2h_wait={:.2}ms",
                        l as f64 / 100_000.0,
                        d as f64 / 100_000.0,
                    );
                }
            }
        }

        let mut out = Vec::with_capacity(r_total);
        for r in 0..r_total {
            let o = r * 4;
            out.push(u32::from_le_bytes([
                buf[o],
                buf[o + 1],
                buf[o + 2],
                buf[o + 3],
            ]));
        }

        for (i, seq) in seqs.iter_mut().enumerate() {
            for &t in &tokens[off[i]..off[i + 1]] {
                seq.tokens.push(t);
            }
            seq.seq_len += ks[i];
        }

        Ok(out)
    }
}

/// Page-locked host blob + its UMA device alias for the mapped-argmax path.
/// Allocated once per process (before the first verify graph capture, so
/// captured replays bake the same fixed device address). Returns `None` when
/// the backend cannot map (non-UMA / stub backends) or the kill switch
/// `ATLAS_NO_MAPPED_ARGMAX=1` is set — callers then use the scratch + copy
/// path unchanged.
fn mapped_argmax_host_dev(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Option<(*mut u8, spark_runtime::gpu::DevicePtr)> {
    use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
    static HOST: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static DEV: AtomicU64 = AtomicU64::new(0);
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("ATLAS_NO_MAPPED_ARGMAX").as_deref() == Ok("1")) {
        return None;
    }
    let mut h = HOST.load(Ordering::Acquire);
    if h.is_null() {
        // Single-threaded caller (the scheduler); failures latch OFF via the
        // null host + 0 dev pair staying unset each call (cheap re-probe is
        // fine — alloc failures here are permanent config facts, not races).
        h = gpu.alloc_host_pinned(65_536).ok()?;
        let d = gpu.host_ptr_to_device(h).ok()?;
        DEV.store(d.0, Ordering::Release);
        HOST.store(h, Ordering::Release);
    }
    let d = DEV.load(Ordering::Acquire);
    if d == 0 {
        return None;
    }
    Some((h, spark_runtime::gpu::DevicePtr(d)))
}
