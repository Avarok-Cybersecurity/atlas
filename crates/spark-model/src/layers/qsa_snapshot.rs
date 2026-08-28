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

impl QsaIndexer {
    /// Marconi aux blob: `[ingested u64][pooled u64][raw_keys bf16 bytes]`.
    /// Raw keys are a deterministic function of the token prefix, so the
    /// snapshot IS the indexer state; block keys are re-pooled on restore
    /// (one kernel) rather than serialized.
    pub fn snapshot_aux(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let hd = self.hd as usize;
        let key_bytes = st.ingested * hd * 2;
        let mut blob = Vec::with_capacity(16 + key_bytes);
        blob.extend_from_slice(&(st.ingested as u64).to_le_bytes());
        blob.extend_from_slice(&(st.pooled as u64).to_le_bytes());
        let off = blob.len();
        blob.resize(off + key_bytes, 0);
        if key_bytes > 0 {
            gpu.copy_d2h_on_stream(st.raw_keys, &mut blob[off..], stream)?;
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
