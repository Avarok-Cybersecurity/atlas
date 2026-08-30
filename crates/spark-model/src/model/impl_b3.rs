// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
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
    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: emit the full token sequence
        // ONCE so a Python reference can run the SAME tokens through HF
        // transformers and dump matching hidden-state captures.
        // Per-model latch: a static would let the previous model swallow this
        // one's dump. Env first, so a disabled dump never burns the shot.
        if std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
            .ok()
            .as_deref()
            == Some("1")
            && self.stats.dumped.keyed("dump:dflash_tokens")
        {
            let tokens_json = serde_json::json!({
                "prompt_len": position - seq.tokens.len() + seq.tokens.len(),
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/atlas_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/atlas_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // MTP loads ALL experts on every rank (no EP filtering), so its MoE
        // output is already complete — no all_reduce needed. Passing comm: None
        // prevents MoeLayer::forward() from doubling the output via SUM.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None, // #30: MTP/draft decode never routes prefill.
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(), // route-aware: base(Skip) skips fold, adapter folds (single-seq reject lifted)
        };
        // Give the drafter its prompt context on the first propose of this
        // sequence: whole-prompt prefill on a COLD turn, carried rows + a
        // short append on a WARM one. See `ensure_drafter_context`.
        self.ensure_drafter_context(proposer, seq, &ctx, stream);
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        // ATLAS_MTP_CATCHUP: before proposing, feed pairs the drafter missed
        // during a serial-decode stretch. Coordinates (measured 2026-07-20 on
        // the 27B rig): at propose entry `position == seq.tokens.len()` and
        // the imminent forward_one writes the pair for sequence key
        // `position - 1`; pair key k = (embed(tokens[k+1]), hidden_k), RoPE
        // k+1. The serial-decode ring stores, under label n, the hidden of
        // the step that COMMITTED token n — i.e. hidden_{n-1} — so pair key k
        // reads ring label k+1. Drafter KV slots are compacted (append-only)
        // while RoPE stays sequence-space, so RoPE gaps are already the norm:
        // partial feeds (clipped to ring coverage) are safe, and wrong feeds
        // cannot corrupt output (verify rejects bad drafts).
        if crate::speculative::mtp_catchup_enabled() && !self.mtp_catchup_ring.is_null() {
            let rows = proposer.drafter_rows(prop_state.as_mut());
            let last_key = proposer.last_pair_key(prop_state.as_mut());
            let (start, count) = *self.mtp_catchup_meta.lock();
            // ATLAS_MTP_REFEED_DEBUG: the ring round-trip check. The pair key
            // this propose is ABOUT to write is `position - 1`, and it reads
            // its hidden from `mtp_hidden_save`; under the label convention
            // that same hidden is ring label `position`. So
            // `fp(ring[position % rows]) == fp(mtp_hidden_save)` iff the
            // ring's write-side slot arithmetic, the D2D plumbing and the
            // read-side slot arithmetic all agree. It does NOT prove the
            // label convention itself (see `mtp_refeed_shift`).
            if crate::speculative::mtp_refeed_debug() {
                let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                let h = self.config.hidden_size;
                let fp_save = crate::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_hidden_save,
                    h,
                );
                let fp_ring = crate::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_catchup_ring.offset((position % ring_rows) * h * 2),
                    h,
                );
                let covered = count > 0 && position >= start && position < start + count;
                tracing::info!(
                    "REFEED_DBG propose position={position} rows={rows} \
                     last_key={last_key:?} ring=[{start},+{count}) \
                     fp_save={fp_save:016x} fp_ring[{position}]={fp_ring:016x} \
                     covered={covered} roundtrip_ok={}",
                    covered && fp_save == fp_ring,
                );
            }
            if let Some(last) = last_key
                && rows > 0
                && count > 0
            {
                // Missing pair keys: (last .. position-1); the propose itself
                // covers position-1. Clip to ring coverage [start, start+count)
                // in label space (label = key + 1).
                let mut k0 = (last + 1).max(start.saturating_sub(1));
                let k1 = (position.saturating_sub(2)).min((start + count).saturating_sub(2));
                let want = (position.saturating_sub(1)).saturating_sub(last + 1);
                if k0 <= k1 && want > 0 {
                    let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                    let h = self.config.hidden_size;
                    let bf16 = 2usize;
                    let fed_from = k0;
                    while k0 <= k1 {
                        // Ring-contiguous segment: labels k0+1 .. until wrap.
                        let slot = (k0 + 1) % ring_rows;
                        let seg_last = k1.min(k0 + (ring_rows - slot) - 1);
                        let n_rows = seg_last - k0 + 1;
                        // Row r feeds pair key k0+r = embed(tokens[k0+r+1]):
                        // the impl reads prompt_tokens[r+1], so pass the
                        // window starting at index k0 (n_rows + 1 tokens).
                        let toks = &seq.tokens[k0..=seg_last + 1];
                        let hid = self.mtp_catchup_ring.offset(slot * h * bf16);
                        if crate::speculative::mtp_refeed_debug() {
                            for r in 0..n_rows {
                                let fp = crate::speculative::hidden_fingerprint(
                                    self.gpu.as_ref(),
                                    hid.offset(r * h * bf16),
                                    h,
                                );
                                tracing::info!(
                                    "REFEED_DBG feed key={} label={} tok={} rope={} fp={fp:016x}",
                                    k0 + r,
                                    k0 + r + 1,
                                    toks[r + 1],
                                    k0 + r + 1,
                                );
                            }
                        }
                        let row_base = proposer.drafter_rows(prop_state.as_mut());
                        match proposer.catchup_drafter(
                            toks,
                            hid,
                            row_base,
                            k0 + 1,
                            prop_state.as_mut(),
                            &ctx,
                            stream,
                        ) {
                            Ok(w) if w == n_rows => k0 = seg_last + 1,
                            Ok(w) => {
                                tracing::debug!(
                                    "MTP catch-up: short feed ({w}/{n_rows} rows) — degrading"
                                );
                                break;
                            }
                            Err(e) => {
                                tracing::debug!("MTP catch-up: feed failed ({e:#}) — degrading");
                                break;
                            }
                        }
                    }
                    if k0 > k1 {
                        tracing::debug!(
                            "MTP catch-up: fed pair keys {fed_from}..={k1} \
                             (missed {want}, position {position})"
                        );
                    }
                } else if want > 0 {
                    tracing::debug!(
                        "MTP catch-up: gap of {want} pairs outside ring coverage \
                         (last_key={last} position={position} ring=[{start},+{count}))"
                    );
                }
            }
        }
        let drafts = proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            prop_state.as_mut(),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            self.dflash_hidden_save,
        )?;
        // Confidence clamp (ATLAS_MTP_DRAFT_CONF, staged off by default):
        // when the drafter's chain confidence is below tau, discard the
        // drafts — the next step decodes serially instead of paying a
        // verify that would most likely reject (break-even acceptance at
        // K=1 on the 35B MoE is ~0.66). The drafter KV rows written by
        // this propose MUST be trimmed exactly as a full rejection would
        // (after_verify(0)), or the drafter desyncs from the target.
        let tau = crate::speculative::draft_conf_tau();
        if tau > 0.0
            && !drafts.is_empty()
            && let Some(conf) = proposer.last_confidence()
            && conf < tau
        {
            tracing::debug!(
                "MTP draft skipped: chain confidence {conf:.3} < tau {tau:.3}                  (pos {position}, {} drafts trimmed)",
                drafts.len(),
            );
            proposer.after_verify(0, prop_state.as_mut(), stream)?;
            return Ok(Vec::new());
        }
        Ok(drafts)
    }

    // Proposer-wiring accessors live in impl_b3_accessors.rs (500-LoC cap).

    /// DFlash prefill capture: copy `proc_count` tokens × hidden_size BF16
    /// from `self.buffers.hidden_states()` (filled by the just-completed
    /// prefill layer) into the per-sequence DFlash accumulator. Called
    /// inside the prefill layer loop after each layer. No-op when:
    ///   - DFlash is disabled (capture_layers empty)
    ///   - `layer_idx` is not in `dflash_capture_layers`
    ///   - The seq has no `DflashProposerState`
    ///   - Rank > 0 under EP/TP (drafter is rank-0 only)
    ///
    /// Layout: writes `hidden[t]` BF16 into
    /// `acc[(chunk_start + t) * 5 * h + slot_idx * h]` for each t.
    /// Per-layer call performs `proc_count` strided d2d_async copies —
    /// at typical prefill of 128–4096 tokens × 5 capture layers, total
    /// 640–20480 launches per prefill. Acceptable launch overhead for
    /// first land; replace with a strided-scatter kernel if profiling
    /// shows it's a bottleneck.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let dstate = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let max_ctx = dstate.max_ctx_len;
        let src_base = self.buffers.hidden_states();
        for t in 0..proc_count {
            let abs_pos = chunk_start + t;
            if abs_pos >= max_ctx {
                break; // accumulator full; drop later positions
            }
            let src = src_base.offset(t * h * bf16);
            let dst_offset = abs_pos * n_capture * h * bf16 + slot_idx * h * bf16;
            self.gpu
                .copy_d2d_async(src, acc_base.offset(dst_offset), h * bf16, stream)?;
        }
        Ok(())
    }

    /// After prefill completes, advance the seq's DFlash `ctx_len` to
    /// `chunk_start + proc_count` so the drafter sees all captured prompt
    /// positions on the first propose() call.
    /// Warm-turn adopt of the previous turn's whole DFlash proposer state
    /// (`model::dflash_carry`). Call at prefill chunk 0, AFTER the
    /// prefix/Marconi decision fixed `proc_start` (the first position this
    /// prefill will process — the first position the 5-layer capture will
    /// cover) and BEFORE any capture write, so a successful adopt lets this
    /// chunk's captures land in the CARRIED accumulator and extend real
    /// context instead of a zeroed fresh one.
    ///
    /// Every bail path frees the carried state (never puts it back): a
    /// mismatched request self-clears the slot, returning the drafter pool
    /// blocks before this request's own drafter allocation needs them.
    /// Failure is always a degrade to today's behavior, never an error.
    pub(super) fn try_adopt_dflash_carry(
        &self,
        seq: &mut crate::traits::SequenceState,
        tokens: &[u32],
        proc_start: usize,
    ) -> Result<()> {
        use crate::layers::DflashProposerState;
        if self.dflash_capture_layers.is_empty()
            || !crate::model::dflash_carry::dflash_carry_enabled()
            || !self.levers.drafter.carry
        {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let Some(ref proposer) = self.proposer else {
            return Ok(());
        };
        let Some(mut entry) = self.dflash_carry.lock().take() else {
            return Ok(());
        };
        let common = entry.common_prefix_len(tokens);
        // The fresh state's capacity was sized for THIS request's reach; the
        // carried accumulator was sized for the PREVIOUS one. Adopting a
        // smaller accumulator would fail-fast on ctx append late in decode,
        // so require at least the fresh capacity.
        let fresh_cap = seq
            .proposer_state
            .as_mut()
            .and_then(|ps| {
                ps.as_any_mut()
                    .downcast_mut::<DflashProposerState>()
                    .map(|d| d.max_ctx_len)
            })
            .unwrap_or(usize::MAX);
        let refuse: Option<&'static str> = (|| {
            let Some(d) = entry
                .state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
            else {
                return Some("carried state is not a DFlash state");
            };
            // The trim equates ctx SLOT index with TOKEN position, which only
            // holds over the accumulator's leading prompt-aligned run
            // (positions == identity). A mid-turn window slide (serial-append
            // watermark) or a reordered append breaks the equality from some
            // slot on — trimming past that point would keep slots holding
            // LATER positions under an earlier index. Use the identity run as
            // the trim ceiling: a slid state's run is 0 and falls to the
            // floor refusal below.
            if d.ctx_positions.len() != d.ctx_len {
                return Some("ctx positions out of sync with ctx_len");
            }
            let aligned = d
                .ctx_positions
                .iter()
                .enumerate()
                .take_while(|(i, p)| **p == *i as i32)
                .count();
            let trim = common.min(aligned);
            if trim < crate::model::dflash_carry::dflash_carry_min_ctx() {
                return Some("common prefix below the carry floor");
            }
            if trim < proc_start {
                // Rows [trim..proc_start) were neither carried nor will they
                // be captured — adopting would re-create the zero-row hole.
                return Some("coverage gap between carried ctx and this prefill");
            }
            // The carried accumulator only has to fit the PROMPT (plus a
            // small margin for the first verify block): decode-time ctx
            // appends clamp at capacity (dflash_decode_append & friends) and
            // the serial path slides its window, so a conversation that
            // outgrows the carried allocation degrades drafter context late
            // in the turn instead of overflowing. Requiring the fresh
            // state's full reach here refused EVERY growing conversation —
            // the exact traffic the carry exists for.
            if d.max_ctx_len < tokens.len() + 32 {
                return Some("carried accumulator smaller than this prompt");
            }
            let _ = fresh_cap;
            // Trim to the verified common prefix. KV slots at or beyond
            // `ctx_committed` are recomputed from the accumulator before any
            // read (`precompute_ctx_kv` covers [ctx_committed..ctx_len)), and
            // per-slot positions below the trim were stamped at insert and
            // stay valid.
            d.ctx_len = trim;
            d.ctx_positions.truncate(trim);
            d.ctx_committed = d.ctx_committed.min(trim);
            d.ctx_count_drafter = d.ctx_count_drafter.min(trim);
            d.seq_len = d.seq_len.min(trim);
            d.last_num_accepted = 0;
            d.last_num_drafted = 0;
            d.skip_next_decode_append = false;
            d.prefill_done = false;
            None
        })();
        if let Some(why) = refuse {
            let carried_ctx = entry
                .state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .map(|d| (d.ctx_len, d.max_ctx_len))
                .unwrap_or((0, 0));
            tracing::debug!(
                "DFLASH_CARRY adopt refused ({why}): common={common} proc_start={proc_start} \
                 prompt={} carried_ctx={} carried_cap={}",
                tokens.len(),
                carried_ctx.0,
                carried_ctx.1,
            );
            proposer.free_state(self.gpu.as_ref(), entry.state.as_mut())?;
            return Ok(());
        }
        // ── Accumulator growth (the growing-conversation case) ──
        // The carried accumulator was sized for the PREVIOUS request's
        // reach. Adopting it undersized made the serial-append watermark
        // SLIDE mid-turn once ctx hit its cap — losing the prompt-aligned
        // prefix and refusing the NEXT turn's adopt (measured: adopt
        // alternated turns). The fresh state this sequence just allocated
        // holds a right-sized, zeroed accumulator: copy the trimmed rows
        // into it and exchange accumulators, so the adopted state has this
        // request's capacity and the fresh state (freed below) takes the
        // small one down with it. The carried PAGED ctx KV is dropped on
        // growth — its device block table is capacity-sized from the old
        // max, so keeping it would overflow on growth; first propose
        // re-allocates at the new size and re-precomputes over the trimmed
        // REAL rows (chunked by ctx_window), which costs what today's warm
        // turn already pays — the accept-quality win stays.
        if let Some(mut fresh) = seq.proposer_state.take() {
            let grew = {
                let (Some(c), Some(f)) = (
                    entry
                        .state
                        .as_any_mut()
                        .downcast_mut::<DflashProposerState>(),
                    fresh.as_any_mut().downcast_mut::<DflashProposerState>(),
                ) else {
                    // Non-DFlash fresh state cannot happen on a DFlash serve;
                    // restore and bail rather than risk mismatched frees.
                    seq.proposer_state = Some(fresh);
                    proposer.free_state(self.gpu.as_ref(), entry.state.as_mut())?;
                    return Ok(());
                };
                if c.max_ctx_len < f.max_ctx_len
                    && f.ctx_hidden_acc.0 != 0
                    && c.ctx_hidden_acc.0 != 0
                {
                    let stream = self.gpu.default_stream();
                    self.gpu.copy_d2d_async(
                        c.ctx_hidden_acc,
                        f.ctx_hidden_acc,
                        c.ctx_len * c.ctx_slot_bytes,
                        stream,
                    )?;
                    // The copy READS the small accumulator, which the
                    // `free_state(fresh)` below frees — drain the stream so
                    // the free cannot race the read. Once per adopt.
                    self.gpu.synchronize(stream)?;
                    std::mem::swap(&mut c.ctx_hidden_acc, &mut f.ctx_hidden_acc);
                    std::mem::swap(&mut c.max_ctx_len, &mut f.max_ctx_len);
                    // KV-PRESERVING growth: the paged ctx KV itself lives in
                    // pool blocks that never move — only the DEVICE block
                    // table was capacity-sized from the old max_ctx_len
                    // (propose derives `max_blocks` from the CURRENT max, so
                    // a grown max would overflow the old table on the next
                    // incremental grab). Re-allocate the table at the new
                    // capacity and re-upload the host-side table (the source
                    // of truth, a few hundred bytes); the committed
                    // watermark and every precomputed K/V slot survive, so a
                    // growth-turn adopt keeps the same ~0.17 s precompute
                    // saving as a non-growth one. `+17` covers the widest
                    // gamma the pools ever size for (`gamma + 1 <= 17`,
                    // matching propose's `max_ctx_len + gamma + 1` bound;
                    // the head's own gamma is not reachable from here).
                    // An allocation failure falls back to dropping the KV —
                    // the accumulator rows above are still the carry.
                    if let Some(old_bt) = c.block_table_dev.take() {
                        const BLOCK_SIZE: usize = 16;
                        let new_max_blocks = (c.max_ctx_len + 17).div_ceil(BLOCK_SIZE);
                        match self.gpu.alloc(new_max_blocks * std::mem::size_of::<u32>()) {
                            Ok(new_bt) => {
                                if !c.block_table.is_empty() {
                                    let bytes: Vec<u8> = c
                                        .block_table
                                        .iter()
                                        .flat_map(|b| b.to_le_bytes())
                                        .collect();
                                    self.gpu.copy_h2d(&bytes, new_bt)?;
                                }
                                c.block_table_dev = Some(new_bt);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "DFLASH_CARRY: block-table realloc failed ({e:#}); \
                                     dropping carried paged KV (accumulator rows kept)"
                                );
                                let blocks = std::mem::take(&mut c.block_table);
                                proposer.free_drafter_kv(&blocks);
                                c.max_ctx_count_drafter = 0;
                                c.ctx_count_drafter = 0;
                                c.ctx_committed = 0;
                                c.seq_len = 0;
                            }
                        }
                        self.gpu.free(old_bt)?;
                    }
                    true
                } else {
                    false
                }
            };
            proposer.free_state(self.gpu.as_ref(), fresh.as_mut())?;
            if grew {
                tracing::debug!("DFLASH_CARRY grew accumulator to this request's reach");
            }
        }
        let adopted_ctx = entry
            .state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .map(|d| (d.ctx_len, d.ctx_committed))
            .unwrap_or((0, 0));
        seq.proposer_state = Some(entry.state);
        tracing::info!(
            "DFLASH_CARRY adopted: ctx={} (committed {}) of prompt={} — capture resumes at {}",
            adopted_ctx.0,
            adopted_ctx.1,
            tokens.len(),
            proc_start,
        );
        Ok(())
    }

    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            let new_len = (chunk_start + proc_count).min(dstate.max_ctx_len);
            dstate.ctx_len = new_len;
            // Phase I (v2): seed per-slot fixed positions for the prompt
            // captures. Prefill slot i holds prompt position i, so the
            // fixed rope position is simply its index. Keep parallel to
            // ctx_len. Re-seed idempotently across prefill chunks.
            dstate.ctx_positions = (0..new_len).map(|i| i as i32).collect();
        }
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Captures only the latest-decoded-token's hidden, matching the
    /// `save_hidden_for_mtp` semantics. The `token_idx` argument selects
    /// which row of `self.buffers.hidden_states()` to read — pass 0 for the
    /// single-token decode path.
    ///
    /// Under EP/TP world > 1: only rank 0 owns the drafter (replicated, not
    /// sharded — same pattern as MTP under EP — see model.rs:7232 comment),
    /// so non-rank-0 ranks skip the capture. The captured hiddens are
    /// post-TP-allreduce so semantically correct on rank 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors save_hidden_for_mtp's effective behavior).
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        // The residual stream is always BF16, so DFlash hidden capture
        // copies BF16 bytes directly with no downcast.
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst_slot = dst.offset(slot * h * bf16);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        Ok(())
    }

    /// Capture `hidden_states[token_idx]` for every DFlash capture layer into
    /// `dflash_hidden_save`. Called from `verify_dflash_step` after the Phase 3
    /// D2H sync, so `token_idx` is the confirmed bonus position. Runs outside
    /// the CUDA graph so the correct accept-prefix position can be used.
    pub(super) fn save_dflash_hidden_dispatch(&self, token_idx: usize, stream: u64) -> Result<()> {
        for &layer_idx in &self.dflash_capture_layers {
            self.try_dflash_capture(layer_idx, token_idx, stream)?;
        }
        Ok(())
    }

    /// K=gamma EAGLE capture: copy the per-layer hidden of ALL `k` verify rows into
    /// the row-major `dflash_hidden_save` ([row0 | row1 | ... ], each row =
    /// n_capture * hidden_size * bf16). Called once per capture layer inside the
    /// verify graph (k is fixed per captured graph). After verify, the scheduler
    /// appends rows 0..=num_accepted to ctx so every committed position gets its
    /// target hidden (fixes the ctx-undercount) and the bonus generator (row
    /// num_accepted) is the freshest slot (EAGLE). No-op unless DFlash is on,
    /// this layer is a capture layer, and rank 0.
    pub(super) fn try_dflash_capture_all(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        debug_assert!(
            k <= kmax,
            "try_dflash_capture_all: k={k} exceeds dflash_hidden_save_rows={kmax}"
        );
        let k_capped = k.min(kmax);
        for t in 0..k_capped {
            let src = self.buffers.hidden_states().offset(t * h * bf16);
            let dst_slot = dst.offset(t * ctx_slot_bytes + slot * h * bf16);
            self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        }
        Ok(())
    }
}
