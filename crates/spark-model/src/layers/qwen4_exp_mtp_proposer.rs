// SPDX-License-Identifier: AGPL-3.0-only

//! `DraftProposer` over the qwen4_exp MTP head.
//!
//! # What this connects
//!
//! The head drafts the next token from (last token, the target's four-stream
//! highway) at a measured 86.5–95.5% accept. This adapts it to the engine's
//! proposer interface so the scheduler can actually verify and commit those
//! drafts — the step that turns accept rate into tokens/s.
//!
//! # Two contract mismatches, both handled here
//!
//! **`target_hidden` is the wrong input.** The trait hands every proposer the
//! COLLAPSED `[1, hidden]` BF16 row. The MTP combiner consumes the FP32
//! four-stream highway instead, which is still live in `ctx.buffers.hc_streams()`
//! when `propose` runs: the last layer's `hc_head` READS the streams and writes
//! the collapsed hidden elsewhere. So the argument is deliberately ignored and
//! the highway is read from the context.
//!
//! **Chaining is autoregressive through the DRAFT's own arena.** For
//! `num_drafts > 1`, draft j+1 needs the highway the draft body just produced,
//! not the target's. That lives in the head's private arena, so the chain is
//! natural: pass `arena.hc_streams()` as the next step's input. Draft 1 reads
//! the target's highway; every later draft reads its own.

use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;
use crate::layers::qwen4_exp_mtp::{Qwen4ExpMtpHead, Qwen4ExpMtpState};
use crate::speculative::{DraftProposer, ProposerState};

pub struct Qwen4ExpMtpProposerState {
    pub inner: Qwen4ExpMtpState,
}

impl ProposerState for Qwen4ExpMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl DraftProposer for Qwen4ExpMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(Qwen4ExpMtpProposerState {
            inner: self.alloc_state(gpu)?,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        last_token: u32,
        // IGNORED — see the module docs: this is the collapsed row, and the
        // combiner needs the four-stream highway.
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp MTP: wrong proposer state type"))?;

        // Apply any rewind `after_verify` deferred (it has no GPU handle).
        if st.inner.pending_rewind > 0 {
            let rows = st.inner.pending_rewind;
            self.rewind_draft(&mut st.inner, rows, ctx.gpu, stream)?;
            st.inner.pending_rewind = 0;
        }

        // Snapshot the DRAFT body's own carry before advancing it, so
        // `after_verify` can unwind the rejected rows.
        st.inner.pre_draft_aux = self.snapshot_draft_aux(&st.inner, ctx.gpu, stream)?;
        st.inner.last_num_drafted = num_drafts;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        // Draft 1 reads the TARGET's highway; each later draft reads the one the
        // draft body itself just wrote, in the head's own arena.
        let mut streams = target_highway_row(
            ctx.buffers.hc_streams(),
            ctx.hc_row_offset,
            ctx.config.hc_mult,
            ctx.config.hidden_size,
        );

        for j in 0..num_drafts {
            let h_out = self.draft_h_out();
            self.draft_hidden(
                token,
                streams,
                position + j,
                &mut st.inner,
                h_out,
                ctx,
                stream,
            )?;
            token = self.draft_token_with_grammar(
                h_out,
                ctx,
                stream,
                if j == 0 { grammar_bitmask } else { None },
            )?;
            drafts.push(token);
            streams = self.draft_streams();
        }
        Ok(drafts)
    }

    /// Unwind the draft body's own state for every REJECTED row.
    ///
    /// This is the draft-side mirror of the target-side `rollback_verify_hc`.
    /// Both sides advance independently — the draft body has its own seq_len,
    /// its own single-layer KV pool and its own QSA carry — so both must be
    /// rewound, and forgetting either leaves that side one row ahead. The QSA
    /// ingest asserts `pos == ingested`, so the failure is loud on the next
    /// draft rather than silent, which is the one mercy here.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp MTP: wrong proposer state type"))?;
        let _ = stream;
        let drafted = st.inner.last_num_drafted;
        st.inner.pending_rewind = drafted.saturating_sub(num_accepted);
        st.inner.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, _state: &mut dyn ProposerState) -> Result<()> {
        // The head owns every device buffer the draft touches (its arena, its
        // KV pool); per-sequence state holds only host bookkeeping plus the
        // body's layer state, which drops with the box.
        Ok(())
    }
}

/// Select the target highway that produced the last committed token.
fn target_highway_row(base: DevicePtr, row: usize, streams: usize, hidden: usize) -> DevicePtr {
    base.offset(row * streams * hidden * std::mem::size_of::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_highway_rows_use_full_fp32_stream_stride() {
        let base = DevicePtr(4096);
        for row in 0..4 {
            assert_eq!(
                target_highway_row(base, row, 4, 2560).0,
                4096 + (row * 40960) as u64
            );
        }
        // A serial bootstrap after wider verify explicitly selects row zero.
        assert_eq!(target_highway_row(base, 0, 4, 2560).0, base.0);
    }
}
