// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence DSA indexer state — the cache the selector reads every decode step.
//!
//! # Why this exists at all
//!
//! The indexer scores each pool from `k_normed` and `gate`, both projections of the
//! *hidden* state (`indexer.wk`, `index_kpool_compress_gate`). Neither is recoverable
//! from the MLA latent — `wk · hidden` cannot be inverted out of a rank-512 compression —
//! so the indexer needs its own cache stream alongside the KV cache. HF does the same
//! thing, keeping indexer state on a per-layer `DynamicIndexedLayer` via
//! `past_key_values.update_indexer`.
//!
//! # No new subsystem
//!
//! `TransformerLayer::alloc_state` is called once per sequence and `LayerState` is an
//! `Any` downcast hook — the same mechanism `qwen3_ssm` uses for recurrent state. This is
//! that, with a bigger buffer.
//!
//! # 🪤 Flat, not paged
//!
//! `dsa_kpool_compress` and `dsa_index_scores` index `k[raw * D + d]` **linearly**, and
//! pools are built over absolute positions from the first valid token. So this is one
//! contiguous per-sequence buffer, not block-table paged. The MLA latent stays paged; only
//! the indexer stream is flat.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaConfig;
use super::select::{DsaSelectGeometry, TOPK_SMEM_CEILING};
use crate::layer::LayerState;

/// Longest context DSA can select over, in tokens.
///
/// 🔴 Set by `dsa_topk_pools`, which bitonic-sorts the padded pool axis in shared memory:
/// 49,152 B / 8 B rounds **down to the power of two** 4,096 pools, times `index_kpool`.
/// With the paged gather decode this is now the ONLY DSA context ceiling left — the
/// 12,288-key masked-attention limit belonged to the oracle path, which is not the serve
/// path. Still far below GLM's advertised 262,144; lifting it needs a segmented/radix
/// select, deliberately out of scope.
pub fn max_dsa_context(cfg: &Glm5NextDsaConfig) -> usize {
    let max_pools = {
        let slots = TOPK_SMEM_CEILING / 8;
        (slots + 1).next_power_of_two() / 2
    };
    max_pools * cfg.index_kpool
}

/// One sequence's indexer cache for one DSA layer.
///
/// Allocated once at sequence creation and never grown: the ceiling above is a hard cap,
/// so a fixed reservation is both correct and small — at `index_head_dim = 128` and
/// `index_kpool = 4` that is 8 MiB per layer per sequence, ~92 MiB across the 11 text DSA
/// layers.
pub struct Glm5NextDsaState {
    /// `[capacity, index_head_dim]` BF16 — LayerNorm'd indexer keys.
    /// 🪤 `indexer.k_norm` is an `nn.LayerNorm` **with a bias**, not an RMSNorm. The bias
    /// is applied when this is written; a `.weight`-only binder silently drops both the
    /// mean subtraction and the bias.
    pub k_normed: DevicePtr,
    /// `[capacity, index_head_dim]` BF16 — the compress-gate projection.
    pub gate: DevicePtr,
    /// `[capacity]` u8 — per-position validity.
    pub valid: DevicePtr,
    /// Tokens written so far. The selector reads `[0, len)`.
    len: usize,
    capacity: usize,
    index_head_dim: usize,
}

impl Glm5NextDsaState {
    /// Reserve for the whole addressable context. `alloc_state` has no length argument, so
    /// the cap — not the prompt — sizes this.
    pub fn alloc(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        cfg.validate()?;
        let capacity = max_dsa_context(cfg);
        let d = cfg.index_head_dim;
        Ok(Self {
            k_normed: gpu.alloc(capacity * d * 2)?,
            gate: gpu.alloc(capacity * d * 2)?,
            valid: gpu.alloc(capacity)?,
            len: 0,
            capacity,
            index_head_dim: d,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Byte offset of row `pos` in `k_normed` / `gate`.
    pub fn row_offset(&self, pos: usize) -> usize {
        pos * self.index_head_dim * 2
    }

    /// Advance after writing `n` rows at `[len, len + n)`.
    ///
    /// Refuses rather than wrapping or truncating: past the cap the selector cannot sort
    /// the pool axis at all, so a silently clamped length would select over a prefix while
    /// the MLA cache held the full context — a wrong answer, not a crash.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        let want = self.len + n;
        if want > self.capacity {
            bail!(
                "DSA indexer cache: {want} tokens exceeds the {} the top-k select can \
                 sort. DSA cannot serve past this context; a segmented/radix select is \
                 the fix, not a bigger buffer.",
                self.capacity
            );
        }
        self.len = want;
        Ok(())
    }

    /// Plan a selection over everything cached so far.
    /// Rewind to `n` rows after a rejected speculative draft.
    ///
    /// The rows in `[n, len)` are left in the cache but become unreachable: the selector reads
    /// `[0, len)` and the next write starts at `n`, so they are overwritten before anything
    /// can select over them. Only shrinks — growing is `advance`'s job, and a request to
    /// "rewind" forward would mean the caller lost track of where the sequence is.
    pub fn rewind_to(&mut self, n: usize) -> Result<()> {
        if n > self.len {
            bail!(
                "DSA indexer rewind to {n} from {}: rewind only shrinks; a forward 'rewind' \
                 means the caller lost the sequence position",
                self.len
            );
        }
        self.len = n;
        Ok(())
    }

    /// Put the counter where a RUN step would have left it, for a step served by a replayed
    /// CUDA graph. `seq_len` is the sequence length before this step's `k` rows.
    ///
    /// 🔴 The same lockstep reconcile `decode_k` does on the eager path, and for the same
    /// reason: a K-row verify writes K rows and the scheduler keeps only the accepted prefix,
    /// so the counter is AHEAD by (k - accepted) whenever a draft was rejected. `decode_k`
    /// rewinds on entry; a replay never calls it, so a plain `advance(k)` compounds that drift
    /// every step. See ANOMALIES A56 — the drafter writes its indexer rows at `len()`, so the
    /// drift moves those rows on top of ones the target selects over.
    pub fn sync_to(&mut self, seq_len: usize, k: usize) -> Result<()> {
        match self.len.cmp(&seq_len) {
            std::cmp::Ordering::Greater => self.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA indexer cache holds {} tokens but the replayed step starts at {seq_len} \
                 — rows are MISSING, not merely stale.",
                self.len
            ),
            std::cmp::Ordering::Equal => {}
        }
        self.advance(k)
    }

    pub fn geometry(&self, cfg: &Glm5NextDsaConfig, q_rows: usize) -> Result<DsaSelectGeometry> {
        DsaSelectGeometry::plan(cfg, self.len, q_rows)
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.k_normed, self.gate, self.valid] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

impl LayerState for Glm5NextDsaState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests;
