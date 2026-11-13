// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-head state of MTP head v2 (L11) and the hook
//! methods `forward_one` / `forward_batch_position` call.
//!
//! One field on [`super::MtpHead`] (`v2`) holds everything the sprint adds to
//! the drafter: the ranked vocabulary map (`mtp_vocab_rank = "freq"`), the
//! device-resident propose chain (`mtp_propose_graph`), the bounded attention
//! window (`set_kv_window`, for L36) and the out-of-head accounting. Every hook
//! is a no-op returning the "not handled" answer when its feature is off, so
//! with both rows off the drafter issues exactly the launches and copies it
//! issued before the sprint.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::vocab_rank::{self, RankedHead};
use crate::layers::ops;

/// Longest chain the device-resident propose carries (drafts per round).
pub const MAX_CHAIN_DRAFTS: usize = 16;
/// Widest batched propose the chain carries (`batch_caps::PROPOSE_META_SEQS`).
pub const MAX_CHAIN_WIDTH: usize = super::batch_caps::PROPOSE_META_SEQS;

/// The device-resident propose chain (`mtp_propose_graph`).
///
/// While a round is active, position j of the chain embeds its input tokens on
/// the device from position j-1's ids (`batched_embed`, the same rows
/// `forward_one` copies from a host id), and every position's ids (and, for
/// the batched propose, top-1 log-probabilities) are collected into
/// `ids` / `lp` instead of being read back. The round then reads them back in
/// ONE copy. Nothing in the chain waits on the host, which is what makes it a
/// graph-safe sequence (`ForwardContext::graph_capture` is set on the batched
/// path); the per-position metadata uploads stay stream-ordered async copies.
pub(super) struct ChainState {
    enabled: bool,
    gather_k: KernelHandle,
    /// `[MAX_CHAIN_DRAFTS, MAX_CHAIN_WIDTH]` u32, position-major.
    ids: DevicePtr,
    /// `[MAX_CHAIN_DRAFTS, MAX_CHAIN_WIDTH]` f32, position-major.
    lp: DevicePtr,
    active: AtomicBool,
    pos: AtomicU32,
    width: AtomicU32,
}

/// Ends a chain round on every exit path, including an error mid-chain.
pub(super) struct ChainRound<'a>(&'a ChainState);

impl Drop for ChainRound<'_> {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}

impl ChainState {
    fn off() -> Self {
        Self {
            enabled: false,
            gather_k: KernelHandle(0),
            ids: DevicePtr::NULL,
            lp: DevicePtr::NULL,
            active: AtomicBool::new(false),
            pos: AtomicU32::new(0),
            width: AtomicU32::new(0),
        }
    }

    /// Whether the chain can carry a round of `num_drafts` positions over
    /// `width` sequences.
    pub(super) fn fits(&self, width: usize, num_drafts: usize) -> bool {
        self.enabled
            && (1..=MAX_CHAIN_WIDTH).contains(&width)
            && (2..=MAX_CHAIN_DRAFTS).contains(&num_drafts)
    }

    /// Start a round over `width` sequences at position 0.
    pub(super) fn begin(&self, width: usize) -> Result<ChainRound<'_>> {
        ensure!(self.enabled, "propose chain: not enabled");
        ensure!(
            (1..=MAX_CHAIN_WIDTH).contains(&width),
            "propose chain: width {width} outside 1..={MAX_CHAIN_WIDTH}"
        );
        self.width.store(width as u32, Ordering::Relaxed);
        self.pos.store(0, Ordering::Relaxed);
        self.active.store(true, Ordering::Release);
        Ok(ChainRound(self))
    }

    /// Move the active round to draft position `pos`.
    pub(super) fn set_pos(&self, pos: usize) {
        self.pos.store(pos as u32, Ordering::Relaxed);
    }

    fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    fn slot(&self, base: DevicePtr, pos: usize) -> DevicePtr {
        base.offset(pos * self.width.load(Ordering::Relaxed) as usize * 4)
    }

    /// Read back the round's ids (and log-probabilities when `with_lp`) in
    /// one copy each: `ids[j * width + i]` is sequence i's draft j.
    pub(super) fn read(
        &self,
        gpu: &dyn GpuBackend,
        num_drafts: usize,
        with_lp: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let width = self.width.load(Ordering::Relaxed) as usize;
        ensure!(
            num_drafts <= MAX_CHAIN_DRAFTS,
            "propose chain: {num_drafts} drafts"
        );
        let mut buf = vec![0u8; num_drafts * width * 4];
        gpu.copy_d2h(self.ids, &mut buf)?;
        let ids = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let lp = if with_lp {
            gpu.copy_d2h(self.lp, &mut buf)?;
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };
        Ok((ids, lp))
    }
}

/// MTP head v2 state (one per `MtpHead`).
pub(crate) struct HeadV2 {
    /// The ranked map when the draft head was built ranked, else `None`
    /// (id-ranked: row r is token r).
    ranked: Option<Arc<RankedHead>>,
    /// Rows of the head's vocabulary projection (the GEMV width).
    rows: u32,
    pub(super) chain: ChainState,
    /// Attention window of this head in drafter KV rows: `u32::MAX` follows
    /// the process default ([`super::window::default_kv_window`]), 0 = none.
    window: AtomicU32,
}

impl HeadV2 {
    /// Resolve the v2 state for a head whose vocabulary projection is the
    /// packed weight at `lm_head_weight` with `rows` rows.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        lm_head_weight: DevicePtr,
        rows: usize,
    ) -> Result<Self> {
        let ranked = vocab_rank::ranked_for(lm_head_weight);
        if ranked.is_none() && vocab_rank::RankMode::resolved() == vocab_rank::RankMode::Freq {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "mtp_vocab_rank=freq: this MTP head drafts with the first-N ids (a ranked head \
                     is built only as the dedicated draft head of a BF16 or FP8 main head, and only \
                     when the LM-head setup line above says so)"
                );
            });
        }
        if let Some(r) = ranked.as_ref() {
            ensure!(
                r.row_to_id.len() >= rows,
                "ranked draft head holds {} rows, the MTP head reads {rows}",
                r.row_to_id.len()
            );
        }
        vocab_rank::set_draft_rows(rows);
        let chain = if crate::layers::ops::port_rows::mtp_propose_graph::row() {
            let gather_k = crate::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed");
            if gather_k.0 == 0 {
                tracing::warn!(
                    "mtp_propose_graph: embed_from_argmax::batched_embed is not in this build; \
                     the propose chain stays per-draft"
                );
                ChainState::off()
            } else {
                let cells = MAX_CHAIN_DRAFTS * MAX_CHAIN_WIDTH;
                let c = ChainState {
                    enabled: true,
                    gather_k,
                    ids: gpu.alloc(cells * 4)?,
                    lp: gpu.alloc(cells * 4)?,
                    active: AtomicBool::new(false),
                    pos: AtomicU32::new(0),
                    width: AtomicU32::new(0),
                };
                tracing::info!(
                    "mtp_propose_graph: device-resident propose chain armed \
                     (up to {MAX_CHAIN_DRAFTS} drafts x {MAX_CHAIN_WIDTH} sequences, one readback per round)"
                );
                c
            }
        } else {
            ChainState::off()
        };
        Ok(Self {
            ranked,
            rows: rows as u32,
            chain,
            window: AtomicU32::new(u32::MAX),
        })
    }

    /// The token a head row holds (the row itself for an id-ranked head), for
    /// host-side diagnostics that index the logits (`AVAROK_MTP_SHADOW_TOPK`).
    pub(super) fn token_of_row(&self, row: usize) -> usize {
        match &self.ranked {
            Some(r) => r.row_to_id.get(row).map_or(row, |&t| t as usize),
            None => row,
        }
    }

    /// Whether the head holds token `id`.
    pub(super) fn holds(&self, id: u32) -> bool {
        match &self.ranked {
            Some(r) => vocab_rank::is_member(&r.in_head, id),
            None => id < self.rows,
        }
    }

    /// Account one committed token seen at a propose
    /// (`atlas_mtp_draft_out_of_head_total` when the head cannot hold it).
    pub(super) fn observe_committed(&self, id: u32) {
        if !self.holds(id) {
            vocab_rank::note_out_of_head();
        }
    }

    /// Step 1 hook: when a chain round is past its first position, gather the
    /// `n` input embeddings on the device from the previous position's ids and
    /// return true; otherwise false (the caller copies from the host ids).
    pub(super) fn chain_embed(
        &self,
        gpu: &dyn GpuBackend,
        embed_table: DevicePtr,
        out: DevicePtr,
        n: usize,
        hidden: u32,
        stream: u64,
    ) -> Result<bool> {
        let c = &self.chain;
        if !c.active() {
            return Ok(false);
        }
        let pos = c.pos.load(Ordering::Relaxed) as usize;
        if pos == 0 {
            return Ok(false);
        }
        ensure!(
            n == c.width.load(Ordering::Relaxed) as usize,
            "propose chain: width changed mid-round"
        );
        ops::batched_embed(
            gpu,
            c.gather_k,
            c.slot(c.ids, pos - 1),
            embed_table,
            out,
            n as u32,
            hidden,
            stream,
        )?;
        Ok(true)
    }

    /// Argmax hook: map the `n` head rows at `ids` to token ids in place
    /// (ranked head) and, during a chain round, store them in the round's slot.
    /// Returns true when the ids were collected by the chain (the caller then
    /// skips its readback).
    pub(super) fn after_argmax(
        &self,
        gpu: &dyn GpuBackend,
        ids: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        let chained = self.chain.active();
        let copy = if chained {
            self.chain.slot(
                self.chain.ids,
                self.chain.pos.load(Ordering::Relaxed) as usize,
            )
        } else {
            DevicePtr::NULL
        };
        if let Some(r) = &self.ranked {
            KernelLaunch::new(gpu, r.remap_k)
                .grid([(n as u32).div_ceil(128), 1, 1])
                .block([128, 1, 1])
                .arg_ptr(ids)
                .arg_ptr(r.row_to_id_dev)
                .arg_u32(n as u32)
                .arg_u32(r.row_to_id.len() as u32)
                .arg_ptr(copy)
                .launch(stream)?;
        } else if chained {
            gpu.copy_d2d_async(ids, copy, n * 4, stream)?;
        }
        Ok(chained)
    }

    /// Batched-propose hook: during a chain round, keep the position's `n`
    /// top-1 log-probabilities (FP32 at `lp`) for the round's readback.
    pub(super) fn collect_lp(
        &self,
        gpu: &dyn GpuBackend,
        lp: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        if self.chain.active() {
            let pos = self.chain.pos.load(Ordering::Relaxed) as usize;
            gpu.copy_d2d_async(lp, self.chain.slot(self.chain.lp, pos), n * 4, stream)?;
        }
        Ok(())
    }

    /// Grammar hook for a RANKED head: the bitmask is in token-id space and
    /// the logits in row space, so translate the mask through the row map and
    /// take the CPU argmax there (the id-ranked branch below the hook would
    /// read row indices as token ids). Stages the chosen id for
    /// `embed_from_argmax` exactly as that branch does. `None` for an
    /// id-ranked head or an unconstrained row: the caller's branches run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn grammar_argmax(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        v: u32,
        grammar_bitmask: Option<&[i32]>,
        out_ptr: DevicePtr,
        embed: Option<(KernelHandle, DevicePtr, DevicePtr, DevicePtr, u32)>,
        position: usize,
        stream: u64,
    ) -> Result<Option<u32>> {
        let (Some(r), Some(bitmask)) = (&self.ranked, grammar_bitmask) else {
            return Ok(None);
        };
        let rows = (v as usize).min(r.row_to_id.len());
        let mut bf16_buf = vec![0u8; rows * 2];
        gpu.copy_d2h(logits, &mut bf16_buf)?;
        let Some(tok) = vocab_rank::masked_argmax_rows(&bf16_buf, &r.row_to_id[..rows], bitmask)
        else {
            tracing::warn!(
                "MTP grammar mask allowed zero tokens of the ranked draft head at pos {position}; \
                 returning 0 as pad-draft (will be rejected at verify)."
            );
            return Ok(Some(0));
        };
        if let Some((embed_k, embed_table, embed_target, token_id_dev, hidden)) = embed {
            gpu.copy_h2d(&tok.to_le_bytes(), out_ptr)?;
            ops::embed_from_argmax(
                gpu,
                embed_k,
                out_ptr,
                embed_table,
                embed_target,
                token_id_dev,
                hidden,
                stream,
            )?;
        }
        Ok(Some(tok))
    }

    /// Set this head's attention window in drafter KV rows (`None` or 0 = no
    /// bound; see [`super::window`]). Overrides the process default from then
    /// on.
    pub(super) fn set_kv_window(&self, window: Option<u32>) {
        let w = window.map_or(0, |w| w.min(u32::MAX - 1));
        self.window.store(w, Ordering::Relaxed);
    }

    /// The window in force: this head's if set, else the process default.
    pub(super) fn kv_window(&self) -> Option<u32> {
        match self.window.load(Ordering::Relaxed) {
            u32::MAX => super::window::default_kv_window(),
            0 => None,
            w => Some(w),
        }
    }

    /// The block table and length the drafter's paged attention reads: the
    /// whole table (borrowed, unchanged) with no window, else the sink blocks
    /// plus the blocks of the last `window` rows.
    pub(super) fn window_view<'a>(
        &self,
        table: &'a [u32],
        len: usize,
        block_size: usize,
    ) -> (Cow<'a, [u32]>, usize) {
        match self.kv_window() {
            None => (Cow::Borrowed(table), len),
            Some(w) => {
                super::window::view(table, len, block_size, w as usize, super::window::SINK_ROWS)
            }
        }
    }
}

#[cfg(test)]
#[path = "head_v2_tests.rs"]
mod head_v2_tests;
