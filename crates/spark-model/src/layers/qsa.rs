// SPDX-License-Identifier: AGPL-3.0-only

//! The Qwen3.8-Flash-Next QSA indexer — decode-side sparse-attention
//! selection (#753 phase G).
//!
//! Reference: `Qwen4ExpTextQSAIndexer`. The attention layer's INPUT (the
//! hc_pre mixed output) is projected to 4 query heads + 1 raw key per token;
//! the visible prefix is grouped into 4-token blocks whose keys are
//! mean-pooled, k_layernormed and roped at the block's first position; each
//! query attends the top-512 blocks by `sum_h relu(q_h . k_b)/sqrt(128)`,
//! plus the incomplete tail. At or below `budget + ratio - 1` (2051) visible
//! tokens the selection is PROVABLY all-visible — the inert regime the port
//! served in until now.
//!
//! v1 SCOPE (decode-side): raw keys are ingested during prefill and decode;
//! selection runs at DECODE steps once the visible prefix exceeds the inert
//! bound, and feeds the EXISTING paged decode attention through a gathered
//! contiguous scratch + identity block table. Prefill queries beyond the
//! inert bound still run dense (a one-time WARN documents the divergence;
//! per-query prefill selection is stage 2). Single sequence, BF16 KV only.
//!
//! CUDA graphs: selection does a host top-k on the scores (D2H), which can
//! never sit inside a captured graph — a layer carrying an indexer vetoes
//! decode-graph capture entirely (graphs measured speed-NEUTRAL on GB10, so
//! this costs nothing).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

#[path = "qsa_select.rs"]
mod qsa_select;
#[path = "qsa_snapshot.rs"]
mod qsa_snapshot;
#[cfg(test)]
#[path = "qsa_tests.rs"]
mod tests;

/// One decode step's selection: contiguous NHD `k/v` scratch + identity table.
pub struct QsaSelection {
    pub k_scratch: DevicePtr,
    pub v_scratch: DevicePtr,
    pub table_dev: DevicePtr,
    pub seq_len_dev: DevicePtr,
    pub n_sel: u32,
    pub max_blocks: u32,
}

pub struct QsaSeqState {
    /// Tokens whose raw keys are in `raw_keys` (contiguous from 0).
    ingested: usize,
    /// Complete 4-token blocks pooled into `block_keys`.
    pooled: usize,
    /// Identity block table upload done (needs block_size, known lazily).
    table_len: usize,
    /// [max_tokens, hd] BF16 — this sequence's raw indexer keys.
    raw_keys: DevicePtr,
    /// [max_tokens/ratio, hd] BF16 — this sequence's pooled block keys.
    block_keys: DevicePtr,
}

pub struct QsaIndexer {
    qk_proj_w: DevicePtr, // [ (n_heads+1)*hd, hidden ] BF16 row-major
    q_norm_w: DevicePtr,  // [hd]
    k_norm_w: DevicePtr,  // [hd]

    n_heads: u32,
    hd: u32,
    ratio: u32,
    budget: u32,
    block_topk: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    hidden: u32,
    nkv_attn: u32,
    hd_attn: u32,
    max_tokens: usize,

    k_pool_k: KernelHandle,
    k_qprep_k: KernelHandle,
    k_score_k: KernelHandle,
    k_gather_k: KernelHandle,
    k_qprep_rows_k: KernelHandle,
    k_score_rows_k: KernelHandle,
    /// Tensor-core split-q scorer. `try_kernel` — absent on any target
    /// whose shadow predates it, which falls back to the scalar path.
    k_score_rows_tc_k: KernelHandle,
    k_prefill_attn_k: KernelHandle,

    qk_scratch: DevicePtr, // [INGEST_SLAB, (n_heads+1)*hd] BF16
    q_post: DevicePtr,     // [n_heads, hd] F32
    scores_dev: DevicePtr, // [max_tokens/ratio] F32
    sel_dev: DevicePtr,    // [budget + ratio] i32
    k_scratch: DevicePtr,  // [budget+ratio, nkv_attn, hd_attn] BF16
    v_scratch: DevicePtr,
    table_dev: DevicePtr,   // [ceil((budget+ratio)/8)] i32 (any block_size >= 8)
    seq_len_dev: DevicePtr, // [1] i32
    /// The sequence's REAL block table, uploaded per prefill-select call —
    /// chunk-0 metadata carries no device table (cache-skip attention is
    /// contiguous), so the host Vec is the source of truth.
    prefill_table_dev: DevicePtr, // [ceil(max_tokens/8)] i32
}

/// Prefill ingest GEMM slab (rows), bounding `qk_scratch`.
const INGEST_SLAB: usize = 2048;

impl QsaIndexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qk_proj_w: DevicePtr,
        q_norm_w: DevicePtr,
        k_norm_w: DevicePtr,
        n_heads: usize,
        hd: usize,
        ratio: usize,
        budget: usize,
        rot: usize,
        theta: f32,
        eps: f32,
        hidden: usize,
        nkv_attn: usize,
        hd_attn: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            ratio > 0 && budget.is_multiple_of(ratio),
            "QSA: budget % ratio != 0"
        );
        let max_tokens: usize = std::env::var("ATLAS_QSA_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768);
        let block_topk = budget / ratio;
        let qk_width = (n_heads + 1) * hd;
        let sel_cap = budget + ratio;
        Ok(Self {
            qk_proj_w,
            q_norm_w,
            k_norm_w,
            n_heads: n_heads as u32,
            hd: hd as u32,
            ratio: ratio as u32,
            budget: budget as u32,
            block_topk: block_topk as u32,
            rot: rot as u32,
            theta,
            eps,
            hidden: hidden as u32,
            nkv_attn: nkv_attn as u32,
            hd_attn: hd_attn as u32,
            max_tokens,
            k_pool_k: gpu.kernel("qsa_indexer", "qsa_block_pool")?,
            k_qprep_k: gpu.kernel("qsa_indexer", "qsa_qprep")?,
            k_score_k: gpu.kernel("qsa_indexer", "qsa_score")?,
            k_gather_k: gpu.kernel("qsa_indexer", "qsa_gather")?,
            k_qprep_rows_k: gpu.kernel("qsa_indexer", "qsa_qprep_rows")?,
            k_score_rows_k: gpu.kernel("qsa_indexer", "qsa_score_rows")?,
            k_score_rows_tc_k: crate::layers::try_kernel(gpu, "qsa_indexer", "qsa_score_rows_tc"),
            k_prefill_attn_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn")?,
            qk_scratch: gpu.alloc(INGEST_SLAB * qk_width * 2)?,
            q_post: gpu.alloc(n_heads * hd * 4)?,
            scores_dev: gpu.alloc(max_tokens / ratio * 4)?,
            sel_dev: gpu.alloc(sel_cap * 4)?,
            k_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            v_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            table_dev: gpu.alloc(sel_cap.div_ceil(8) * 4)?,
            seq_len_dev: gpu.alloc(4)?,
            prefill_table_dev: gpu.alloc(max_tokens.div_ceil(8) * 4)?,
        })
    }

    /// The largest visible prefix whose selection is provably all-visible.
    /// One sequence's indexer carry: counters + raw/pooled key buffers
    /// (per-seq CONTENT; launch scratch stays layer-owned — steps serialize).
    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<QsaSeqState> {
        let hd = self.hd as usize;
        let ratio = self.ratio as usize;
        Ok(QsaSeqState {
            ingested: 0,
            pooled: 0,
            table_len: 0,
            raw_keys: gpu.alloc(self.max_tokens * hd * 2)?,
            block_keys: gpu.alloc(self.max_tokens / ratio * hd * 2)?,
        })
    }

    /// Release one sequence's indexer carry.
    ///
    /// `QsaSeqState` holds bare `DevicePtr`s, so dropping the struct frees
    /// nothing. Without this every finished sequence left
    /// `max_tokens * hd * 2` (raw) + `max_tokens/ratio * hd * 2` (pooled)
    /// bytes on the device for EACH full-attention layer — at 200K context
    /// and 12 such layers, ~739 MB per request. On unified memory that is
    /// invisible to RSS and to `nvidia-smi`, so it surfaced only as the host
    /// running out of RAM with no process to blame.
    ///
    /// Idempotent: each pointer is nulled as it is freed, so a second call
    /// (or a release after a partial failure) cannot double-free. A failure
    /// on the first buffer still attempts the second — leaking the rest
    /// because the first free failed is the bug this exists to prevent.
    pub fn release_seq_state(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for p in [&mut st.raw_keys, &mut st.block_keys] {
            if p.is_null() {
                continue;
            }
            if let Err(e) = gpu.free(*p) {
                first_err.get_or_insert(e);
            }
            *p = DevicePtr(0);
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn inert_bound(&self) -> usize {
        (self.budget + self.ratio - 1) as usize
    }

    fn qk_width(&self) -> usize {
        (self.n_heads as usize + 1) * self.hd as usize
    }

    /// Ingest `num_tokens` prefill tokens starting at `seq_start`: project
    /// qk, park the raw keys, pool freshly complete blocks. `seq_start == 0`
    /// resets the sequence (single-seq v1, PLE-style).
    pub fn prefill_ingest(
        &self,
        st: &mut QsaSeqState,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_start: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if seq_start == 0 {
            st.ingested = 0;
            st.pooled = 0;
        }
        anyhow::ensure!(
            seq_start == st.ingested,
            "QSA: prefill chunk starts at {seq_start} but {} tokens are \
             ingested — a prefix-cache skip bypassed the indexer. Serve \
             qwen4_exp with the prefix cache disabled until QSA learns to \
             re-ingest cached prefixes.",
            st.ingested
        );
        anyhow::ensure!(
            seq_start + num_tokens <= self.max_tokens,
            "QSA: {} tokens exceeds ATLAS_QSA_MAX_TOKENS={}",
            seq_start + num_tokens,
            self.max_tokens
        );

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        let mut off = 0usize;
        while off < num_tokens {
            let ts = INGEST_SLAB.min(num_tokens - off);
            ops::cublas_bf16_proj_dense(
                hidden.offset((off) * self.hidden as usize * 2),
                self.qk_proj_w,
                self.qk_scratch,
                ts as u32,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (prefill)")?;
            // Raw key = the last hd columns of each row.
            gpu.copy_d2d_2d_async(
                self.qk_scratch.offset(self.n_heads as usize * hd * 2),
                qkw * 2,
                st.raw_keys.offset((seq_start + off) * hd * 2),
                hd * 2,
                hd * 2,
                ts,
                stream,
            )?;
            off += ts;
        }
        st.ingested = seq_start + num_tokens;
        self.pool_new_blocks(st, gpu, stream)
    }

    fn pool_new_blocks(
        &self,
        st: &mut QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let complete = st.ingested / self.ratio as usize;
        if complete > st.pooled {
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                st.raw_keys,
                self.k_norm_w,
                st.block_keys,
                st.pooled as u32,
                (complete - st.pooled) as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            st.pooled = complete;
        }
        Ok(())
    }

    // `prefill_select`: see `qsa_select.rs` (#[path] child, ≤500 LoC split).

    /// Decode-step ingest + selection for the token at `pos` (0-based;
    /// `pos + 1` visible). `None` inside the inert bound (dense is exact).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        pos: usize,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<QsaSelection>> {
        anyhow::ensure!(
            pos == st.ingested,
            "QSA: decode at pos {pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        anyhow::ensure!(
            pos < self.max_tokens,
            "QSA: pos {pos} >= ATLAS_QSA_MAX_TOKENS"
        );

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        // qk GEMV for this token; row 0 of the scratch.
        ops::cublas_bf16_proj_dense(
            normed,
            self.qk_proj_w,
            self.qk_scratch,
            1,
            qkw as u32,
            self.hidden,
            stream,
        )
        .context("QSA qk projection (decode)")?;
        gpu.copy_d2d_async(
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            st.raw_keys.offset(pos * hd * 2),
            hd * 2,
            stream,
        )?;
        st.ingested = pos + 1;
        self.pool_new_blocks(st, gpu, stream)?;

        let visible = pos + 1;
        let complete = visible / self.ratio as usize;
        if complete <= self.block_topk as usize {
            return Ok(None); // provably all-visible: dense path is exact
        }

        // q prep + block scores.
        ops::qsa_qprep(
            gpu,
            self.k_qprep_k,
            self.qk_scratch,
            self.q_norm_w,
            self.q_post,
            self.n_heads,
            self.hd,
            self.rot,
            pos as u32,
            self.theta,
            self.eps,
            stream,
        )?;
        ops::qsa_score(
            gpu,
            self.k_score_k,
            self.q_post,
            st.block_keys,
            self.scores_dev,
            complete as u32,
            self.n_heads,
            self.hd,
            stream,
        )?;

        // Host top-k over the block scores (D2H — decode graphs are vetoed
        // whenever an indexer is present, so this is never inside a capture).
        let mut raw = vec![0u8; complete * 4];
        gpu.copy_d2h_on_stream(self.scores_dev, &mut raw, stream)?;
        let scores: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut order: Vec<u32> = (0..complete as u32).collect();
        // torch.topk returns the k largest, ties broken by LOWER index —
        // sort by (-score, index) and take the first k for identical sets.
        order.sort_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut blocks: Vec<u32> = order[..self.block_topk as usize].to_vec();
        blocks.sort_unstable();

        let ratio = self.ratio as usize;
        let mut sel: Vec<i32> = Vec::with_capacity(self.budget as usize + ratio);
        for b in &blocks {
            let base = *b as i32 * self.ratio as i32;
            for r in 0..self.ratio as i32 {
                sel.push(base + r);
            }
        }
        for t in complete * ratio..visible {
            sel.push(t as i32);
        }
        let n_sel = sel.len() as u32;

        let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async(&sel_bytes, self.sel_dev, stream)?;
        ops::qsa_gather(
            gpu,
            self.k_gather_k,
            k_pool,
            v_pool,
            block_table_dev,
            self.sel_dev,
            self.k_scratch,
            self.v_scratch,
            n_sel,
            block_size,
            self.nkv_attn,
            self.hd_attn,
            stream,
        )?;

        // Identity table + seq_len for the scratch-as-paged-cache view.
        let pages = (n_sel as usize).div_ceil(block_size as usize);
        if st.table_len < pages {
            let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&ident, self.table_dev, stream)?;
            st.table_len = pages;
        }
        gpu.copy_h2d_async(&(n_sel as i32).to_le_bytes(), self.seq_len_dev, stream)?;

        Ok(Some(QsaSelection {
            k_scratch: self.k_scratch,
            v_scratch: self.v_scratch,
            table_dev: self.table_dev,
            seq_len_dev: self.seq_len_dev,
            n_sel,
            max_blocks: pages as u32,
        }))
    }
}
