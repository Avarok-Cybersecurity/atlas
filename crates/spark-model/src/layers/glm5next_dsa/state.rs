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
use super::select::DsaSelectGeometry;
use crate::layer::LayerState;

/// Longest context DSA can select over, in tokens.
///
/// 🟢 This used to be a KERNEL limit — `dsa_topk_pools` bitonic-sorted the whole padded pool
/// axis in shared memory, capping the serve at 4,096 pools = **16,384 tokens** whatever
/// `--max-seq-len` claimed (ANOMALIES A62). The select is tiled now, so the kernel imposes
/// nothing and this is purely an ALLOCATION decision: how many rows of indexer cache each
/// sequence reserves, which is `--max-seq-len` rounded down to a whole pool.
///
/// 🪤 It is charged per sequence per DSA layer at `2 · index_head_dim` BF16 + 1 B a token —
/// 5,643 B/token across GLM-5.3's 11 text DSA layers, and the indexer is REPLICATED, so EP
/// does not halve it. `Glm5NextSkeleton::state_budget` must carry the same number or the
/// serve allocates past its own `--gpu-memory-utilization` (the A59 class of cliff).
pub fn max_dsa_context(cfg: &Glm5NextDsaConfig) -> usize {
    // Whole pools only: a trailing partial pool is not a pool (`contiguous_pool_count`).
    (cfg.max_context / cfg.index_kpool) * cfg.index_kpool
}

/// One sequence's indexer cache for one DSA layer.
///
/// Allocated once at sequence creation and never grown: [`max_dsa_context`] is a hard cap,
/// so a fixed reservation is correct. At `index_head_dim = 128` that is `513 B` a token a
/// layer — 8 MiB per layer (~92 MiB over the 11 text DSA layers) at a 16,384-token context,
/// and 64 MiB per layer (~736 MiB) at 131,072.
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

    /// Would `n` more rows fit? Ask BEFORE writing them, not after.
    ///
    /// 🔴 `advance` is too late on its own. `indexer_forward` GEMMs `k_normed` and `gate`
    /// straight into row `len()` and only then advances, so at `len == capacity` the write
    /// lands on row `capacity` — 256 B past `k_normed`/`gate` and 1 B past `valid`. CUDA
    /// reports that asynchronously as `CUDA_ERROR_ILLEGAL_ADDRESS (700)` at the next
    /// synchronize, and a 700 is **sticky**: every later CUDA call in the context fails, so
    /// one over-length prompt takes the serve down for every subsequent request while
    /// `/v1/models`, `/health` and `/health/live` all keep answering 200. Checking first
    /// turns that into a plain per-request error. ANOMALIES **A62** (the overrun) and
    /// **A60** (the non-recovering serve it explains).
    pub fn ensure_room(&self, n: usize) -> Result<()> {
        self.ensure_room_through(self.len + n)
    }

    /// The same refusal for an ABSOLUTE end position.
    ///
    /// 🔴 The graph-replay path knows where the sequence will END (`seq_len + k`) but not
    /// where this counter currently sits: a rejected draft leaves it AHEAD, and `sync_to`
    /// rewinds it only after the replay has already written. Asking in absolute terms is
    /// what makes the check answerable before `launch_graph`. A62.
    pub fn ensure_room_through(&self, end: usize) -> Result<()> {
        if end > self.capacity {
            bail!(
                "DSA indexer cache: {end} tokens exceeds the {} rows reserved for this \
                 sequence. The indexer cache is sized from --max-seq-len at sequence \
                 creation and never grows; raise --max-seq-len (and re-check the memory \
                 budget) to serve a longer context.",
                self.capacity
            );
        }
        Ok(())
    }

    /// Advance after writing `n` rows at `[len, len + n)`.
    ///
    /// Refuses rather than wrapping or truncating: past the reservation there is no row to
    /// write, and a silently clamped length would select over a prefix while the MLA cache
    /// held the full context — a wrong answer, not a crash.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        self.ensure_room(n)?;
        self.len += n;
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

    /// Release the per-sequence device buffers.
    ///
    /// Takes `&mut self` rather than `self` because the only caller reaches the
    /// state through `&mut dyn ProposerState` and cannot move out of it. The
    /// by-value signature this replaces was inherently call-once; the caller
    /// (`Glm5NextMtpHead::free_state`) now owns that guard via
    /// `Glm5NextMtpProposerState::released`.
    ///
    /// 🔴 Every buffer freed here is baked into captured CUDA graphs, so this
    /// MUST run after `free_sequence` has destroyed the slot's `decode_graph`
    /// and `verify2/3/4_graph` — which it does (ANOMALIES A56 put that teardown
    /// in place, and it sits ~110 lines above the `free_state` call site).
    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
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
