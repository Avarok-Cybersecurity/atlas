// SPDX-License-Identifier: AGPL-3.0-only

//! What the indexer must put back after a rejected draft.
//!
//! Split from `qsa.rs` on the 500-line cap, along the seam the pair already
//! forms: everything else in that file computes a selection, and these two
//! preserve and restore the state that computing it consumed. That is a
//! different question, and the one speculative decoding gets wrong — a draft
//! that is rejected must leave the indexer exactly as it found it, or the
//! next step selects against a prefix that never happened.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

/// Header: `[ingested u64][pooled u64]`, then the raw keys.
const QSA_AUX_HEADER: usize = 16;

impl QsaIndexer {
    /// Marconi aux blob: `[ingested u64][pooled u64][raw_keys bf16 bytes]`.
    /// Raw keys are a deterministic function of the token prefix, so the
    /// snapshot IS the indexer state; block keys are re-pooled on restore
    /// (one kernel) rather than serialized.
    /// Byte length of this sequence's blob — computed entirely on the HOST
    /// (`ingested` is a host-side mark), which is what lets the collect lay out
    /// every layer's slice before issuing a single copy.
    pub fn aux_blob_len(&self, st: &QsaSeqState) -> usize {
        QSA_AUX_HEADER + st.ingested * self.hd as usize * 2
    }

    /// Fill `dst` with the blob WITHOUT synchronising: the 16-byte header is
    /// written host-side, the raw keys are ENQUEUED. The caller owns the one
    /// trailing `synchronize` and must not read `dst` before it.
    ///
    /// This is the single writer of the blob format; [`Self::snapshot_aux`] is
    /// a thin wrapper over it, so the batched and legacy paths cannot drift
    /// apart and desync `restore_aux`'s parse.
    pub fn snapshot_aux_into(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        let want = self.aux_blob_len(st);
        anyhow::ensure!(
            dst.len() == want,
            "QSA aux blob: dst is {} B, plan said {want} B",
            dst.len()
        );
        dst[..8].copy_from_slice(&(st.ingested as u64).to_le_bytes());
        dst[8..QSA_AUX_HEADER].copy_from_slice(&(st.pooled as u64).to_le_bytes());
        if want > QSA_AUX_HEADER {
            gpu.copy_d2h_async(st.raw_keys, &mut dst[QSA_AUX_HEADER..], stream)?;
        }
        Ok(())
    }

    pub fn snapshot_aux(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let mut blob = vec![0u8; self.aux_blob_len(st)];
        self.snapshot_aux_into(st, gpu, stream, &mut blob)?;
        // The legacy contract is "bytes are readable on return", which the old
        // `copy_d2h_on_stream` provided by draining inside the copy. Same
        // ordering, one explicit sync instead of an implicit one — and, as
        // before, no sync at all when the header is the whole blob and nothing
        // was enqueued.
        if blob.len() > QSA_AUX_HEADER {
            gpu.synchronize(stream)?;
        }
        Ok(blob)
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit:
    /// upload the raw keys, reset the counters, re-pool the block keys.
    pub fn restore_aux(
        &self,
        st: &mut QsaSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 16, "QSA aux blob truncated");
        let ingested = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let pooled = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
        let hd = self.hd as usize;
        anyhow::ensure!(
            blob.len() == 16 + ingested * hd * 2,
            "QSA aux blob size mismatch"
        );
        anyhow::ensure!(ingested <= self.max_tokens, "QSA aux exceeds key cache");
        if ingested > 0 {
            gpu.copy_h2d_async(&blob[16..], st.raw_keys, stream)?;
        }
        st.ingested = ingested;
        st.pooled = 0;
        if pooled > 0 {
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                st.raw_keys,
                self.k_norm_w,
                st.block_keys,
                0,
                pooled as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            st.pooled = pooled;
        }
        Ok(())
    }
}
