// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's per-hook work behind the selector's lock
//! (port lane L32): the key-sketch update after each K write, the query
//! capture in prefill, the decode and verify plan (the device rebuild of this
//! step's view tables into the fixed arena), and the paged prefill plan of a
//! conversation's next turn. A failing sketch or capture turns the view off
//! instead of failing the forward.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::metrics::{Counters, SkipReason};
use super::state::Inner;
use crate::layers::qwen3_attention::sparse_seam::{
    PAD_SEQ_ID, PrefillPlan, SeamCtx, SeqBatch, SparsePlan,
};

impl Inner {
    /// `on_kv_written`: update the sketch and flag sequences it missed.
    pub(crate) fn record_written(
        &mut self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
        counters: &Counters,
    ) -> Result<()> {
        let Some(core) = self.core.as_ref().filter(|_| self.ready(c.gpu)) else {
            return Ok(());
        };
        let gaps = match core.record_written(c, s, first_pos, n_new) {
            Ok(g) => g,
            Err(e) => {
                let why = format!("the key sketch update failed: {e:#}");
                self.fail(c.gpu, c.in_graph, counters, &why);
                return Ok(());
            }
        };
        if gaps.iter().any(|&g| g > 0) {
            let slots = self.stream_slots(s);
            for (i, &g) in gaps.iter().enumerate() {
                if g > 0
                    && let Some(Some(slot)) = slots.get(i)
                    && let Some(arm) = self.arms.get_mut(slot)
                {
                    arm.gap = true;
                }
            }
        }
        Ok(())
    }

    /// `on_query` in prefill: forward to the capture with the streams named.
    pub(crate) fn capture(
        &mut self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
        counters: &Counters,
    ) -> Result<()> {
        if !self.ready(c.gpu) || self.arms.values().all(|a| a.kept_span.is_none()) {
            return Ok(());
        }
        if let Err(e) = self.capture_named(c, s, q_pre_rope, q) {
            let why = format!("the query capture failed: {e:#}");
            self.fail(c.gpu, c.in_graph, counters, &why);
        }
        Ok(())
    }

    fn capture_named(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<()> {
        let Some(core) = self.core.as_ref() else {
            return Ok(());
        };
        if !s.seq_ids.is_empty() {
            return core.capture(c, s, q_pre_rope, q).map(|_| ());
        }
        let ids: Vec<u64> = self
            .stream_slots(s)
            .into_iter()
            .map(|o| o.unwrap_or(PAD_SEQ_ID))
            .collect();
        if ids.iter().all(|&i| i == PAD_SEQ_ID) {
            return Ok(());
        }
        let named = SeqBatch {
            n: s.n,
            block_table: s.block_table,
            seq_lens: s.seq_lens,
            max_blocks_per_seq: s.max_blocks_per_seq,
            host_seq_lens: s.host_seq_lens,
            seq_ids: &ids,
            rows_per_seq: s.rows_per_seq,
            q_row_stride: s.q_row_stride,
            q_rows: s.q_rows,
            first_pos: s.first_pos,
            new_tokens: s.new_tokens,
            host_block_tables: s.host_block_tables,
            block_table_ptrs: s.block_table_ptrs,
        };
        core.capture(c, &named, q_pre_rope, q).map(|_| ())
    }

    /// Decode and verify: rebuild this step's view tables on the device (a
    /// row with no view keeps its real pages and length).
    pub(crate) fn decode_plan(
        &mut self,
        c: &SeamCtx,
        s: &SeqBatch,
        counters: &Counters,
    ) -> Result<SparsePlan> {
        if !self.ready(c.gpu) {
            if c.in_graph && self.off.is_none() && !self.noted_not_ready {
                self.noted_not_ready = true;
                counters.skip(SkipReason::NotReady);
                tracing::warn!(
                    "bounded KV view: a decode graph was captured before the view's buffers existed; \
                     that graph keeps the dense tables"
                );
            }
            return Ok(SparsePlan::Dense);
        }
        let Some(dev) = self.dev.as_ref() else {
            return Ok(SparsePlan::Dense);
        };
        let launched = dev.launch(
            c.gpu,
            s.block_table,
            s.seq_lens,
            s.n,
            s.max_blocks_per_seq,
            c.kv.block_size,
            c.stream,
        )?;
        match launched {
            Some((block_table, seq_lens)) => Ok(SparsePlan::View {
                block_table,
                seq_lens,
                max_blocks_per_seq: s.max_blocks_per_seq,
            }),
            None => {
                let shape = (s.n, s.max_blocks_per_seq);
                if !self.noted_shapes.contains(&shape) {
                    self.noted_shapes.push(shape);
                    counters.skip(SkipReason::ArenaShape);
                    tracing::warn!(
                        "bounded KV view: a {}-row x {}-block step does not fit the view table arena; \
                         that shape attends its full context",
                        shape.0,
                        shape.1,
                    );
                }
                Ok(SparsePlan::Dense)
            }
        }
    }

    /// A paged prefill chunk of a sequence that continues a conversation:
    /// the conversation's view plus every page from its tail on (this
    /// chunk's included). Dense otherwise.
    pub(crate) fn prefill_plan(&mut self, c: &SeamCtx, s: &SeqBatch) -> Result<PrefillPlan> {
        if !self.ready(c.gpu) || s.n != 1 {
            return Ok(PrefillPlan::Dense);
        }
        let (Some(&row), Some(&first), Some(&extent)) = (
            s.host_block_tables.first(),
            s.first_pos.first(),
            s.host_seq_lens.first(),
        ) else {
            return Ok(PrefillPlan::Dense);
        };
        let Some(Some(slot)) = self.stream_slots(s).first().copied() else {
            return Ok(PrefillPlan::Dense);
        };
        if let Some((cs, cf, ce, kv_len, table)) = self.prefill_cache
            && (cs, cf, ce) == (slot, first, extent)
        {
            return Ok(PrefillPlan::View {
                block_table: table,
                kv_len,
            });
        }
        let bs = c.kv.block_size;
        let Some(shape) = self
            .arms
            .get(&slot)
            .and_then(|a| a.prior.as_ref())
            .map(|p| &p.shape)
        else {
            return Ok(PrefillPlan::Dense);
        };
        // Every query of the chunk must sit at or after the tail page, or the
        // compacted positions of the dropped pages would cross it.
        if first < shape.tail_page * bs {
            return Ok(PrefillPlan::Dense);
        }
        let Some((table, kv_len)) = shape.compose(row, extent, bs) else {
            return Ok(PrefillPlan::Dense);
        };
        let Some(dev) = self.dev.as_ref() else {
            return Ok(PrefillPlan::Dense);
        };
        let Some(ptr) = dev.upload_prefill(c.gpu, &table, c.stream)? else {
            return Ok(PrefillPlan::Dense);
        };
        self.prefill_cache = Some((slot, first, extent, kv_len, ptr));
        Ok(PrefillPlan::View {
            block_table: ptr,
            kv_len,
        })
    }
}
