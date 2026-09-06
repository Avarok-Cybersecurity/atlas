// SPDX-License-Identifier: AGPL-3.0-only

//! PLE Marconi aux-state: serialize / restore the per-sequence lexical
//! carry (token history + conv state) that rides the SSM snapshots.
//! Split from `layer.rs` for the ≤500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{PleLayer, PleSeqState};
use crate::layers::ple::ids::ple_ngram_ids;

impl PleLayer {
    /// Marconi aux blob: `[hist_len u32][history u32s][conv f32 bytes]`.
    /// The whole per-sequence carry — a prefix hit restoring KV+SSM without
    /// this would run the n-gram hash on the PREVIOUS request's history.
    pub fn snapshot_aux(
        &self,
        st: &PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        let mut blob = Vec::with_capacity(4 + st.history.len() * 4 + conv_bytes);
        blob.extend_from_slice(&(st.history.len() as u32).to_le_bytes());
        for t in &st.history {
            blob.extend_from_slice(&t.to_le_bytes());
        }
        let off = blob.len();
        blob.resize(off + conv_bytes, 0);
        gpu.copy_d2h_on_stream(st.conv, &mut blob[off..], stream)?;
        Ok(blob)
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit.
    pub fn restore_aux(
        &self,
        st: &mut PleSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 4, "PLE aux blob truncated");
        let n = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        anyhow::ensure!(
            blob.len() == 4 + n * 4 + conv_bytes,
            "PLE aux blob size mismatch"
        );
        st.history = blob[4..4 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        st.prestaged_va = None;
        gpu.copy_h2d_async(&blob[4 + n * 4..], st.conv, stream)?;
        Ok(())
    }
}

impl PleLayer {
    /// Fresh sequence: EOS-filled history and a zeroed conv state.
    pub(super) fn reset(
        &self,
        st: &mut PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        st.history = vec![self.dims.eos_token_id; self.dims.context_len()];
        st.prestaged_va = None;
        let zeros = vec![0u8; self.state_len * self.hc_mult * self.hidden * 4];
        gpu.copy_h2d_async(&zeros, st.conv, stream)?;
        Ok(())
    }

    /// Hoisted per-step HOST work for decode under CUDA graphs: the n-gram
    /// hash, the NVMe fault-in and the slot upload into the stable
    /// `slots_dev` buffer. All three are capture-illegal (the upload reads
    /// pageable memory, which invalidates a recording graph with status
    /// 901), so the scheduler calls this BEFORE graph replay/capture — the
    /// same phasing decode_a already gives the `token_ids` upload. `forward`
    /// then consumes `prestaged_va` and enqueues only stable-buffer kernels.
    ///
    /// History advances HERE; the prestaged `forward` must not advance it
    /// again.
    pub fn prestage(
        &self,
        st: &mut PleSeqState,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }
        let mut window = st.history.clone();
        window.extend_from_slice(tokens);
        let all = ple_ngram_ids(&self.dims, &window);
        let rows = &all[all.len() - tokens.len()..];
        let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let va = self.gather_host(&flat, gpu, stream)?;
        let keep = self.dims.context_len();
        st.history = window[window.len() - keep..].to_vec();
        st.prestaged_va = Some(va);
        st.last_staged_va = va;
        Ok(())
    }
}

// ── Per-row carry checkpoints for the K-row speculative verify ──
//
// The mHC K-row verify (`qwen3_ssm/trait_decode_batched_hc.rs`) advances THREE
// per-row carries in one pass. Two of them already had a mechanism:
//
//   * the SSM `h_state`/`conv_state` are written per row into pool
//     intermediates by the conv+GDN kernels, and
//   * QSA's `ingested`/`pooled` are contiguous marks, so
//     `QsaIndexer::align_seq_state` rewinds them to an ABSOLUTE position with
//     no snapshot at all.
//
// PLE was the one with neither. Its conv is a rolling FP32 state and its
// history is a fixed-length window whose oldest ids have already rolled off,
// so nothing about it can be reconstructed by truncation — a partial accept
// left it ADVANCED over the rejected rows, which is the documented corruption
// class. These three calls give it the same per-row granularity the other two
// carries have.
impl PleLayer {
    /// Start a K-row verify: drop any snapshots a previous verify left. The
    /// device slots are kept — the next verify reuses them.
    pub fn begin_verify_rows(&self, st: &mut PleSeqState) {
        st.verify_rows.clear();
    }

    /// Record "the carry after row `t`". Called once per row boundary a
    /// partial accept can land on — rows `0..K-1`, matching `hc_publish_rows`;
    /// the last row needs none because a full accept keeps the live state.
    ///
    /// Stream-ordered, no sync: the token history is host state and is
    /// cloned; the FP32 conv state is copied device-to-device into slot `t`
    /// on `stream`. The previous host blob (`copy_d2h_on_stream`) was a
    /// stream sync per row — see `PleSeqState::verify_rows`.
    pub fn push_verify_row(
        &self,
        st: &mut PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let t = st.verify_rows.len();
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        while st.verify_conv.len() <= t {
            st.verify_conv.push(gpu.alloc(conv_bytes)?);
        }
        gpu.copy_d2d_async(st.conv, st.verify_conv[t], conv_bytes, stream)?;
        st.verify_rows.push(st.history.clone());
        Ok(())
    }

    /// Rewind the carry to the boundary after row `t`, i.e. to a commit of
    /// `t + 1` rows.
    ///
    /// Errors rather than silently no-opping when the row was never recorded:
    /// a missing snapshot means the carry stays ahead of the sequence, which
    /// is precisely the silent desync this exists to prevent.
    pub fn rewind_verify_row(
        &self,
        st: &mut PleSeqState,
        t: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let history = st.verify_rows.get(t).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "PLE verify rewind to row {t}, but only {} row snapshots were \
                 recorded. The carry would be left ADVANCED over the rejected \
                 rows — the documented degeneration class — so this refuses.",
                st.verify_rows.len()
            )
        })?;
        let slot = st
            .verify_conv
            .get(t)
            .copied()
            .filter(|p| !p.is_null())
            .ok_or_else(|| {
                anyhow::anyhow!("PLE verify rewind to row {t}: conv snapshot slot missing")
            })?;
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        gpu.copy_d2d_async(slot, st.conv, conv_bytes, stream)?;
        st.history = history;
        st.prestaged_va = None;
        Ok(())
    }
}
