// SPDX-License-Identifier: AGPL-3.0-only

//! Batched ctx precompute for the cross-sequence propose.
//!
//! The per-sequence `precompute_ctx_kv` streams the drafter's `fc` (262 MB)
//! and fused-KV (105 MB) weights once per SEQUENCE per step over a 2-4 row
//! tail. At C=16 that is 16 serial passes, ~41 ms of GPU per step, plus the
//! host stalls between them (2026-09-02 nsys, C=16 gamma-4 prose: 63 ms of a
//! 270 ms step). Every op in that pipeline is row-independent (GEMM rows,
//! per-row norms, per-row RoPE, per-row cache writes), so the tails of all
//! n sequences can be gathered into one contiguous `[M, L_t * h_t]` slab and
//! pushed through the SAME pipeline once — byte-identical output, one weight
//! stream, one slot-mapping upload, one positions upload.
//!
//! Sequences whose tail does not fit the staging budget run the unchanged
//! per-sequence chunk loop.
//
// provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashProposerState, PRECOMPUTE_BATCH_ROWS};
use crate::layer::ForwardContext;
use crate::speculative::ProposerState;

const BLOCK_SIZE: usize = 16;

static ENGAGED_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl BlockDiffusionDraftHead {
    /// One fc + fused-KV precompute over the uncommitted ctx tail of every
    /// sequence in `states`. Advances each sequence's `ctx_committed` to its
    /// `ctx_len` on success. Sequences with an empty tail are untouched.
    pub(super) fn precompute_ctx_kv_batched(
        &self,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * bf16;

        // Collect (index, committed, count) per sequence with a non-empty tail.
        let mut jobs: Vec<(usize, usize, usize)> = Vec::with_capacity(states.len());
        for (i, st) in states.iter_mut().enumerate() {
            let Some(d) = st.as_any_mut().downcast_mut::<DflashProposerState>() else {
                anyhow::bail!("precompute_ctx_kv_batched: not a DFlash proposer state");
            };
            if d.block_table_dev.is_none() {
                anyhow::bail!("precompute_ctx_kv_batched: seq {i} has no drafter block table");
            }
            let committed = d.ctx_committed.min(d.ctx_len);
            let count = d.ctx_len - committed;
            if count > 0 {
                jobs.push((i, committed, count));
            }
        }
        if jobs.is_empty() {
            return Ok(());
        }

        // Oversized tails (a serial-append stretch) take the per-sequence
        // chunk loop; the rest are batched in groups that fit the staging.
        let mut batched: Vec<(usize, usize, usize)> = Vec::with_capacity(jobs.len());
        let budget = PRECOMPUTE_BATCH_ROWS.min(self.ctx_window);
        for &(i, committed, count) in &jobs {
            if count > budget {
                let d = states[i]
                    .as_any_mut()
                    .downcast_mut::<DflashProposerState>()
                    .expect("checked above");
                self.precompute_ctx_tail_single(d, committed, ctx, stream)?;
            } else {
                batched.push((i, committed, count));
            }
        }

        // Group budget: the staging slab holds PRECOMPUTE_BATCH_ROWS rows, but
        // `fc_proj`, `fused_kv_out` and `slot_mapping_dev` are sized for
        // `ctx_window` rows, so the budget is the smaller of the two (an
        // `ATLAS_DFLASH_CTX_WINDOW` below 256 would otherwise overrun them).
        let mut lo = 0usize;
        while lo < batched.len() {
            // Greedy group: consecutive jobs whose rows sum to <= budget.
            let mut hi = lo;
            let mut rows = 0usize;
            while hi < batched.len() && rows + batched[hi].2 <= budget {
                rows += batched[hi].2;
                hi += 1;
            }
            debug_assert!(hi > lo);

            // Gather: one D2D per sequence into the contiguous staging slab;
            // positions and paged-cache slots built host-side in row order.
            let mut positions: Vec<i32> = Vec::with_capacity(rows);
            let mut slots: Vec<u8> = Vec::with_capacity(rows * 8);
            let mut row_off = 0usize;
            for &(i, committed, count) in &batched[lo..hi] {
                let d = states[i]
                    .as_any_mut()
                    .downcast_mut::<DflashProposerState>()
                    .expect("checked above");
                let src = d.ctx_hidden_acc.offset(committed * ctx_slot_bytes);
                let dst = self.scratch.precompute_in.offset(row_off * ctx_slot_bytes);
                gpu.copy_d2d_async(src, dst, count * ctx_slot_bytes, stream)?;
                positions.extend_from_slice(&d.ctx_positions[committed..committed + count]);
                // Slots come from the HOST block table. The per-sequence
                // path fills them on-device from `block_table_dev`; both are
                // grown together in `propose.rs` (host push, then one H2D of
                // the whole table), so they agree. A device-only growth path
                // would break that invariant silently: pin it here.
                debug_assert!(
                    d.block_table_dev.is_some() && !d.block_table.is_empty(),
                    "precompute_ctx_kv_batched: host/device drafter block tables out of step"
                );
                for idx in committed..committed + count {
                    let logical = idx / BLOCK_SIZE;
                    let physical = *d.block_table.get(logical).ok_or_else(|| {
                        anyhow::anyhow!(
                            "precompute_ctx_kv_batched: ctx slot {idx} beyond block table ({} blocks)",
                            d.block_table.len()
                        )
                    })? as usize;
                    let slot = (physical * BLOCK_SIZE + idx % BLOCK_SIZE) as i64;
                    slots.extend_from_slice(&slot.to_le_bytes());
                }
                row_off += count;
            }
            debug_assert_eq!(row_off, rows);
            gpu.copy_h2d(&slots, self.scratch.slot_mapping_dev)?;

            self.precompute_ctx_kv(
                self.scratch.precompute_in,
                0,
                rows,
                &positions,
                self.scratch.slot_mapping_dev,
                ctx,
                stream,
                true,
            )?;

            for &(i, _, _) in &batched[lo..hi] {
                let d = states[i]
                    .as_any_mut()
                    .downcast_mut::<DflashProposerState>()
                    .expect("checked above");
                d.ctx_committed = d.ctx_len;
            }

            if !ENGAGED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "DFlash BATCHED ctx precompute ENGAGED: {} seqs, {} rows in one fc+KV pass \
                     (per-seq loop: {} passes)",
                    hi - lo,
                    rows,
                    hi - lo,
                );
            }
            lo = hi;
        }
        Ok(())
    }

    /// The unchanged per-sequence tail precompute (mirrors the Phase I loop
    /// in `propose.rs`), used for a tail larger than the batched staging.
    fn precompute_ctx_tail_single(
        &self,
        dstate: &mut DflashProposerState,
        committed: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.ctx_window > 0,
            "DFlash precompute: ctx_window=0 but ctx tail of {} slots needs precompute",
            dstate.ctx_len - committed,
        );
        let slot_mapping = &self.scratch.slot_mapping_dev;
        let mut chunk_start = committed;
        while chunk_start < dstate.ctx_len {
            let chunk_count = (dstate.ctx_len - chunk_start).min(self.ctx_window);
            crate::layers::ops::fill_slots_from_block_table(
                ctx.gpu,
                self.kernels.fill_slots,
                *slot_mapping,
                dstate.block_table_dev.unwrap(),
                chunk_start as u32,
                chunk_count as u32,
                BLOCK_SIZE as u32,
                stream,
            )?;
            let slot_positions = &dstate.ctx_positions[chunk_start..chunk_start + chunk_count];
            self.precompute_ctx_kv(
                dstate.ctx_hidden_acc,
                chunk_start,
                chunk_count,
                slot_positions,
                *slot_mapping,
                ctx,
                stream,
                true,
            )?;
            chunk_start += chunk_count;
        }
        dstate.ctx_committed = dstate.ctx_len;
        Ok(())
    }
}
