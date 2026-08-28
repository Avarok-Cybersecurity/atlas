// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 DSA token selection — the production launcher for the indexer pipeline.
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`.
//!
//! This is the `examples/dsa_indexer_microtest.rs` GATE-4 pipeline lifted out of the
//! example and given a launcher a real layer can call. The kernels, their argument
//! order and their numerics are already proven against HF 5.16.1 on real weights;
//! nothing here re-derives them. What this module owns is the part the microtest did
//! by hand and a layer cannot: **geometry, capacity and refusal**.
//!
//! ```text
//! k_normed, gate, valid, ape  -> dsa_kpool_compress   -> pool keys / indices / valid
//! q, weights, q_pos           -> dsa_index_scores     -> [Q, P] scores + candidacy
//!                             -> dsa_topk_pools       -> [Q, select_k] pool ids
//!                             -> dsa_expand_selection -> [Q, out_width] token ids
//! ```
//!
//! # 🔴 Hard context ceiling — this launcher REFUSES rather than truncates
//!
//! `dsa_topk_pools` bitonic-sorts the padded pool axis in shared memory: `NP2` floats
//! plus `NP2` ints, `NP2` the next power of two at or above the pool count. Against
//! the 49,152 B runtime shared-memory ceiling that caps `NP2` at 4,096 — i.e.
//! **4,096 pools, 16,384 tokens of context at `index_kpool = 4`**. The kernel does not
//! silently truncate and neither does this launcher: past the ceiling
//! [`DsaSelectGeometry::plan`] fails, naming the limit. Lifting it means replacing the
//! bitonic select with a segmented/radix select — deliberately out of scope.
//!
//! # 🪤 Compaction is the identity here, and that is a derived fact, not an assumption
//!
//! [`crate::layers::glm5next_dsa_ref::kept_pools`] keeps pool `p` only when **every**
//! one of its `kpool` slots is in range and valid, with pooling starting at the first
//! valid token. Over a contiguous, unpadded cache — every decode step at batch 1 —
//! that set is exactly the prefix `0 .. seq / kpool`, so the compacted array is a
//! prefix of the full one and `dsa_compact_pools` would copy a buffer onto itself.
//! This launcher therefore uses the full arrays in place and takes the prefix.
//! `contiguous_pool_count` is proven equal to the reference for every sequence length
//! in `tests`. A **left-padded** batch breaks the prefix property and genuinely needs
//! the compaction arm — not built, and [`DsaSelectGeometry::plan`] is documented as
//! contiguous-only.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// Runtime shared-memory ceiling the top-k select is budgeted against, matching
/// `SMEM_CEILING` in `examples/dsa_indexer_microtest.rs`.
pub const TOPK_SMEM_CEILING: usize = 49_152;

/// Threads per block for `dsa_index_scores`, and the bytes of shared memory it
/// reduces through. Taken verbatim from the proven microtest launch.
const SCORES_BLOCK: u32 = 128;
/// Threads per block for `dsa_topk_pools` and `dsa_expand_selection`.
const ROW_BLOCK: u32 = 256;

/// Most pools `dsa_topk_pools` can sort: the largest **power of two** whose padded
/// `[f32, i32]` pair fits the shared-memory ceiling. 49,152 B / 8 B is 6,144, but the
/// bitonic sort pads up to a power of two, so the usable cap is 4,096.
fn max_pools_for_smem() -> usize {
    let slots = TOPK_SMEM_CEILING / 8;
    (slots + 1).next_power_of_two() / 2
}

/// Pools kept over a contiguous, unpadded cache of `seq` tokens.
///
/// A pool needs all `kpool` slots, so the trailing partial pool is not a pool. Proven
/// against `glm5next_dsa_ref::kept_pools` in `tests`.
pub fn contiguous_pool_count(kpool: usize, seq: usize) -> usize {
    seq / kpool
}

/// Launch geometry for one selection pass — every count the four kernels need, and
/// every capacity check, decided before a single pointer is touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaSelectGeometry {
    /// Tokens resident in the indexer cache.
    pub seq: usize,
    /// Query rows selecting this pass (1 on a batch-1 decode step).
    pub q_rows: usize,
    /// Pools `dsa_kpool_compress` writes, including the trailing partial one.
    pub n_pools_full: usize,
    /// Pools that are real — the prefix everything downstream reads.
    pub n_pools: usize,
    /// Pools selected per query.
    pub select_k: usize,
    /// Emitted index-row width.
    pub out_width: usize,
    /// Padded pool axis the bitonic sort runs over.
    pub topk_np2: usize,
    /// Shared memory `dsa_topk_pools` needs, in bytes.
    pub topk_smem: usize,
    /// Channels per compress block.
    pub index_head_dim: usize,
    pub index_heads: usize,
    pub index_kpool: usize,
}

impl DsaSelectGeometry {
    /// Plan a selection over a **contiguous, unpadded** cache.
    ///
    /// Fails rather than truncates when the pool axis outgrows the top-k kernel's
    /// shared-memory budget.
    pub fn plan(cfg: &Glm5NextDsaConfig, seq: usize, q_rows: usize) -> Result<Self> {
        cfg.validate()?;
        if q_rows == 0 {
            bail!("DSA select: q_rows must be > 0");
        }
        if seq == 0 {
            bail!("DSA select: seq must be > 0");
        }
        let kp = cfg.index_kpool;
        let n_pools = contiguous_pool_count(kp, seq);
        // 🔴 `n_pools == 0` is a LEGAL regime, not a refusal. HF 5.16.1
        // `Glm5NextTextIndexer.forward` has no short-sequence branch: below
        // `index_kpool` tokens `pool_valid` is all-false, `keep = pool_valid.any(0)`
        // empties the pool axis, and `select_k = min(index_topk // index_kpool, 0)`
        // is 0 — so nothing is selected and `append_visible_tail` supplies the raw
        // visible tokens. Since every token of a sub-pool sequence is in the
        // incomplete pool, that IS dense attention; the sparse path is unchanged
        // from `seq >= index_kpool` on. `glm5next_dsa_ref::{kept_pools,
        // expand_selection}` already model this; only this launcher refused it,
        // which stopped the first forward at layer 3 (2026-08-28).
        let topk_np2 = n_pools.next_power_of_two().max(2);
        let topk_smem = topk_np2 * 8;
        if topk_smem > TOPK_SMEM_CEILING {
            // The sort runs over the PADDED axis, so the real cap is the largest
            // power of two that fits — 4,096, not the 6,144 the raw byte budget
            // suggests. Quoting the unrounded number in the error would send the
            // reader looking for 24,576 tokens of context that never work.
            let max_pools = max_pools_for_smem();
            bail!(
                "DSA select: {seq} tokens make {n_pools} pools, needing {topk_smem} B of \
                 shared memory for the bitonic top-k against a {TOPK_SMEM_CEILING} B \
                 ceiling. dsa_topk_pools caps at {max_pools} pools = {} tokens at \
                 index_kpool={kp}. Lifting this needs a segmented/radix select, not a \
                 bigger launch.",
                max_pools * kp,
            );
        }
        Ok(Self {
            seq,
            q_rows,
            n_pools_full: seq.div_ceil(kp),
            n_pools,
            select_k: cfg.select_k(n_pools),
            out_width: cfg.out_width(),
            topk_np2,
            topk_smem,
            index_head_dim: cfg.index_head_dim,
            index_heads: cfg.index_heads,
            index_kpool: kp,
        })
    }

    /// Bytes of each scratch region this pass writes, in [`DsaSelectScratch`] order.
    fn scratch_bytes(&self) -> [usize; 6] {
        [
            self.n_pools_full * self.index_head_dim * 4, // pool_keys   f32
            self.n_pools_full * self.index_kpool * 4,    // pool_indices i32
            self.n_pools_full,                           // pool_valid  u8
            self.q_rows * self.n_pools * 4,              // scores      f32
            self.q_rows * self.n_pools,                  // valid_cand  u8
            self.q_rows * self.select_k * 4,             // selected    i32
        ]
    }
}

/// Device-side inputs to a selection pass. Every one is owned by the caller; this
/// module allocates nothing but its own scratch.
#[derive(Debug, Clone, Copy)]
pub struct DsaSelectInputs {
    /// `[seq, index_head_dim]` BF16 — indexer keys, **already LayerNorm'd**.
    /// 🪤 `indexer.k_norm` is an `nn.LayerNorm` with a bias, not an RMSNorm.
    pub k_normed: DevicePtr,
    /// `[seq, index_head_dim]` BF16 — `index_kpool_compress_gate` projection.
    pub gate: DevicePtr,
    /// `[seq]` u8 — per-key validity.
    pub valid: DevicePtr,
    /// `[index_kpool, index_head_dim]` **f32** — the APE table.
    /// 🪤 BF16 on disk, f32 to the kernel; the loader must upconvert. This is the
    /// #347 dtype-mismatch class, so the width is stated here rather than inferred.
    pub ape: DevicePtr,
    /// `[q_rows, index_heads, index_head_dim]` f32.
    pub q: DevicePtr,
    /// `[q_rows, index_heads]` f32, **already carrying the `index_heads^-0.5`
    /// factor** — `dsa_index_scores` does not apply it.
    pub weights: DevicePtr,
    /// `[q_rows]` i32 — absolute position of each query.
    pub q_pos: DevicePtr,
    /// `[q_rows]` u8 — a zero row selects nothing and stays all `-1`.
    pub q_mask: DevicePtr,
    /// Index of the first valid key; pooling starts here so left padding is skipped.
    pub first_key: i32,
}

/// Scratch the pipeline writes through, allocated once and reused across steps.
///
/// Sized from a worst-case geometry so a growing context never reallocates mid-serve;
/// [`Self::fits`] refuses a pass that would outgrow it rather than overrunning.
#[derive(Debug)]
pub struct DsaSelectScratch {
    pool_keys: DevicePtr,
    pool_indices: DevicePtr,
    pool_valid: DevicePtr,
    scores: DevicePtr,
    valid_cand: DevicePtr,
    selected: DevicePtr,
    /// `[q_rows, out_width]` i32 token ids, `-1` where nothing was selected. The
    /// result of the pass; fully written by `dsa_expand_selection` on every path.
    tokens: DevicePtr,
    capacity: [usize; 6],
    tokens_bytes: usize,
}

impl DsaSelectScratch {
    /// Allocate for the worst case this layer will ever see.
    pub fn alloc(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        geom: &DsaSelectGeometry,
    ) -> Result<Self> {
        let capacity = geom.scratch_bytes();
        let tokens_bytes = geom.q_rows * cfg.out_width() * 4;
        Ok(Self {
            pool_keys: gpu.alloc(capacity[0])?,
            pool_indices: gpu.alloc(capacity[1])?,
            pool_valid: gpu.alloc(capacity[2])?,
            scores: gpu.alloc(capacity[3])?,
            valid_cand: gpu.alloc(capacity[4])?,
            selected: gpu.alloc(capacity[5])?,
            tokens: gpu.alloc(tokens_bytes)?,
            capacity,
            tokens_bytes,
        })
    }

    /// `[q_rows, out_width]` i32 selection produced by the last pass.
    pub fn tokens(&self) -> DevicePtr {
        self.tokens
    }

    /// Whether `geom` fits what was allocated. Checked on every pass: a context that
    /// grew past the reservation must fail loudly, not scribble past a buffer — the
    /// A25 recv-buffer class of bug.
    pub fn fits(&self, cfg: &Glm5NextDsaConfig, geom: &DsaSelectGeometry) -> Result<()> {
        let want = geom.scratch_bytes();
        for (i, (w, c)) in want.iter().zip(self.capacity.iter()).enumerate() {
            if w > c {
                bail!(
                    "DSA select: scratch region {i} needs {w} B but only {c} B was \
                     reserved ({} tokens, {} pools, {} query rows)",
                    geom.seq,
                    geom.n_pools,
                    geom.q_rows
                );
            }
        }
        let want_tokens = geom.q_rows * cfg.out_width() * 4;
        if want_tokens > self.tokens_bytes {
            bail!(
                "DSA select: selection output needs {want_tokens} B but only {} B was \
                 reserved",
                self.tokens_bytes
            );
        }
        Ok(())
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.pool_keys,
            self.pool_indices,
            self.pool_valid,
            self.scores,
            self.valid_cand,
            self.selected,
            self.tokens,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// Run the four selection kernels, leaving `[q_rows, out_width]` token ids in
/// [`DsaSelectScratch::tokens`].
///
/// Launch geometry and argument order are transcribed from the GATE-4 arm of
/// `examples/dsa_indexer_microtest.rs`, which is the numerically-proven reference.
/// The pass is enqueued on `stream` and **not** synchronised — the caller sequences it
/// with the attention that consumes the selection.
#[allow(clippy::too_many_arguments)]
pub fn select_tokens(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    inputs: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    stream: u64,
) -> Result<()> {
    scratch.fits(cfg, geom)?;

    let d = geom.index_head_dim;
    let kp = geom.index_kpool;

    // 🪤 Below `index_kpool` tokens there are no pools to score or sort, and a
    // zero-extent grid is an illegal launch, not a no-op. Stages 2 and 3 are
    // skipped; stage 1 still runs (`n_pools_full >= 1`) and stage 4 still runs,
    // writing the whole row -1 and then appending the visible tail — which is the
    // entire selection in this regime.
    let has_pools = geom.n_pools > 0;

    // ── 1. pool compression ─────────────────────────────────────────────────────
    // Launched over the FULL pool count, exactly as the microtest does; the trailing
    // partial pool is written, marked invalid, and never read downstream.
    KernelLaunch::new(gpu, kernels.kpool_compress)
        .grid([geom.n_pools_full as u32, 1, 1])
        .block([d.min(1024) as u32, 1, 1])
        .arg_ptr(inputs.k_normed)
        .arg_ptr(inputs.gate)
        .arg_ptr(inputs.valid)
        .arg_ptr(inputs.ape)
        .arg_ptr(scratch.pool_keys)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.pool_valid)
        .arg_u32(geom.seq as u32)
        .arg_u32(d as u32)
        .arg_u32(kp as u32)
        .arg_i32(inputs.first_key)
        .launch(stream)?;

    // 🪤 No `dsa_compact_pools` launch: over a contiguous cache the kept set is the
    // prefix `0..n_pools`, so compaction is a buffer-to-itself copy. See the module
    // header — a left-padded batch would need it.

    // ── 2. per-(query, pool) index scores ───────────────────────────────────────
    if has_pools {
        KernelLaunch::new(gpu, kernels.index_scores)
            .grid([geom.n_pools as u32, geom.q_rows as u32, 1])
            .block([SCORES_BLOCK, 1, 1])
            .shared_mem(SCORES_BLOCK)
            .arg_ptr(inputs.q)
            .arg_ptr(scratch.pool_keys)
            .arg_ptr(inputs.weights)
            .arg_ptr(scratch.pool_indices)
            .arg_ptr(scratch.pool_valid)
            .arg_ptr(inputs.valid)
            .arg_ptr(inputs.q_pos)
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.valid_cand)
            .arg_u32(geom.q_rows as u32)
            .arg_u32(geom.n_pools as u32)
            .arg_u32(geom.index_heads as u32)
            .arg_u32(d as u32)
            .arg_u32(kp as u32)
            .arg_u32(geom.seq as u32)
            .arg_f32((d as f32).powf(-0.5))
            .launch(stream)?;

        // ── 3. top-k over pools ─────────────────────────────────────────────────────
        // Capacity was refused at plan time; this launch cannot overflow shared memory.
        KernelLaunch::new(gpu, kernels.topk_pools)
            .grid([geom.q_rows as u32, 1, 1])
            .block([ROW_BLOCK, 1, 1])
            .shared_mem(geom.topk_smem as u32)
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.selected)
            .arg_u32(geom.q_rows as u32)
            .arg_u32(geom.n_pools as u32)
            .arg_u32(geom.topk_np2 as u32)
            .arg_u32(geom.select_k as u32)
            .launch(stream)?;
    }

    // ── 4. expand pools to raw token ids ────────────────────────────────────────
    KernelLaunch::new(gpu, kernels.expand_selection)
        .grid([geom.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.selected)
        .arg_ptr(scratch.pool_indices)
        .arg_ptr(scratch.valid_cand)
        .arg_ptr(inputs.valid)
        .arg_ptr(inputs.q_pos)
        .arg_ptr(inputs.q_mask)
        .arg_ptr(scratch.tokens)
        .arg_u32(geom.q_rows as u32)
        .arg_u32(geom.n_pools as u32)
        .arg_u32(kp as u32)
        .arg_u32(geom.seq as u32)
        .arg_u32(geom.select_k as u32)
        .arg_u32(geom.out_width as u32)
        .arg_i32(inputs.first_key)
        .arg_i32(cfg.always_select_tail as i32)
        .launch(stream)?;

    Ok(())
}

#[cfg(test)]
mod tests;
