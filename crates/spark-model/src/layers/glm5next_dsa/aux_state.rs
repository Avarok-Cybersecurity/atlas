// SPDX-License-Identifier: AGPL-3.0-only

//! Marconi aux state for the DSA indexer — the half of a GLM-5.3 sequence that
//! a KV-only prefix-cache hit cannot reconstruct.
//!
//! # Why a blob and not a rewind
//!
//! [`Glm5NextDsaState`] is append-only with a host cursor, so `rewind_to` is an
//! exact restore *within a live sequence*: the rows are still there and only the
//! cursor moved. A prefix-cache hit is not that. It hands the prefix to a
//! sequence whose indexer buffer was never written, so there is nothing to
//! rewind to — and the rows cannot be recomputed either, because `k_normed` and
//! `gate` are projections of the *hidden* state (`indexer.wk`,
//! `index_kpool_compress_gate`) and `wk · hidden` cannot be inverted out of the
//! rank-512 MLA latent (see the module doc on [`super::state`]). The rows have
//! to travel with the snapshot.
//!
//! # The precedent this copies
//!
//! `layers/qsa_snapshot.rs` does exactly this for qwen4_exp's QSA indexer:
//! serialize the ingested keys plus the cursor, and on restore re-derive by
//! kernel whatever is cheaper to recompute than to carry. DSA has nothing in
//! the second category — pooling happens inside the selector kernel each step
//! and is never stored — so the blob is the whole reachable cache and nothing
//! else.
//!
//! # 🔴 Fail-closed, deliberately
//!
//! Every rejection here is a hard `Err`, never a skip. A DSA restore that
//! silently applied a *stale but correctly sized* blob would select over another
//! sequence's keys and return a wrong answer under HTTP 200 — the A25/A55 shape.
//! The size and geometry checks below are what make a mismatched blob
//! unrepresentable rather than merely unlikely.
//!
//! # Cost
//!
//! `len * (4 * index_head_dim + 1)` bytes per DSA layer — 513 B/token/layer at
//! GLM-5.3's `index_head_dim = 128`, 5,643 B/token across its 11 text DSA
//! layers. The blob carries `len` rows, NOT `capacity`: a 4K prefix costs 4K
//! rows even on a serve declaring `--max-seq-len 131072`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;

use super::state::Glm5NextDsaState;

/// `[len u64][index_head_dim u64]`, little-endian.
const HEADER_BYTES: usize = 16;

/// Bytes a blob occupies for `len` rows at `index_head_dim`.
///
/// Mirrors [`super::state::indexer_state_bytes`] but over `len`, not
/// `capacity` — the snapshot carries what was written, not what was reserved.
fn blob_bytes(len: usize, index_head_dim: usize) -> usize {
    HEADER_BYTES + len * (index_head_dim * 4 + 1)
}

impl Glm5NextDsaState {
    /// Serialize the reachable indexer cache `[0, len)` for a Marconi snapshot.
    ///
    /// Only `len` rows are copied. The rows in `[len, capacity)` are either
    /// never written or already unreachable (`rewind_to` leaves stale rows
    /// behind on purpose), so carrying them would be both wasteful and a way to
    /// resurrect state the sequence has disowned.
    pub fn snapshot_blob(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        let len = self.len();
        let d = self.index_head_dim();
        let key_bytes = len * d * 2;

        let mut blob = vec![0u8; blob_bytes(len, d)];
        blob[..8].copy_from_slice(&(len as u64).to_le_bytes());
        blob[8..16].copy_from_slice(&(d as u64).to_le_bytes());

        if len > 0 {
            let (k_off, g_off, v_off) = self.blob_offsets(len, d);
            gpu.copy_d2h_on_stream(self.k_normed, &mut blob[k_off..k_off + key_bytes], stream)?;
            gpu.copy_d2h_on_stream(self.gate, &mut blob[g_off..g_off + key_bytes], stream)?;
            gpu.copy_d2h_on_stream(self.valid, &mut blob[v_off..v_off + len], stream)?;
        }
        Ok(blob)
    }

    /// Restore a snapshot's indexer rows into this (freshly allocated) state.
    ///
    /// 🔴 Refuses rather than repairs. A blob that is truncated, geometrically
    /// different, internally inconsistent, or longer than this sequence's
    /// reservation is an `Err`; the caller (`apply_aux_states`) propagates it
    /// with `?` and the prefix-cache hit fails loudly instead of serving from a
    /// half-written indexer.
    pub fn restore_blob(&mut self, blob: &[u8], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        ensure!(
            blob.len() >= HEADER_BYTES,
            "DSA aux blob truncated: {} bytes, need at least {HEADER_BYTES} for the header",
            blob.len()
        );
        let len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let d = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;

        let want_d = self.index_head_dim();
        ensure!(
            d == want_d,
            "DSA aux blob index_head_dim {d} != this layer's {want_d} — the snapshot was \
             taken under a different model geometry"
        );
        ensure!(
            blob.len() == blob_bytes(len, d),
            "DSA aux blob size mismatch: {} bytes for {len} rows at head_dim {d}, expected {}",
            blob.len(),
            blob_bytes(len, d)
        );
        // Capacity is `--max-seq-len`-derived and can legitimately differ between
        // the serve that wrote the snapshot and the one restoring it (a tiered or
        // spilled blob outlives a process). A shorter reservation is a refusal,
        // not a truncation: the rows past `capacity` have nowhere to land and a
        // clamped `len` would select over a prefix while MLA held the full
        // context — the wrong answer `advance` already refuses to produce.
        self.ensure_room_through(len)?;

        let key_bytes = len * d * 2;
        if len > 0 {
            let (k_off, g_off, v_off) = self.blob_offsets(len, d);
            gpu.copy_h2d_async(&blob[k_off..k_off + key_bytes], self.k_normed, stream)?;
            gpu.copy_h2d_async(&blob[g_off..g_off + key_bytes], self.gate, stream)?;
            gpu.copy_h2d_async(&blob[v_off..v_off + len], self.valid, stream)?;
        }
        // Cursor last, and through the existing guarded pair rather than a new
        // setter: an early-returning `?` above must leave the state at the
        // length it already had, so a failed restore cannot look like a partial
        // success to `decode_k`'s lockstep check.
        self.rewind_to(0)?;
        self.advance(len)?;
        Ok(())
    }

    /// `(k_normed, gate, valid)` byte offsets within a blob of `len` rows.
    fn blob_offsets(&self, len: usize, d: usize) -> (usize, usize, usize) {
        let key_bytes = len * d * 2;
        (
            HEADER_BYTES,
            HEADER_BYTES + key_bytes,
            HEADER_BYTES + 2 * key_bytes,
        )
    }
}

#[cfg(test)]
mod tests;
