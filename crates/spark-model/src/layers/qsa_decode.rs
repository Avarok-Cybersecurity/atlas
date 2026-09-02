// SPDX-License-Identifier: AGPL-3.0-only

//! The DECODE selection tail for the QSA indexer, split from `qsa.rs` for the
//! ≤500 LoC cap. Child module of `qsa` (via `#[path]`) so the indexer's
//! private fields stay reachable without widening their visibility.
//!
//! Input: this step's block scores, already in `scores_dev`. Output:
//! `sel_dev` — the expanded token-index array the gather consumes — plus
//! `seq_len_dev` and the identity block table that turn the gathered scratch
//! into a paged cache the stock decode attention can read.
//!
//! Two implementations, byte-identical by construction and asserted so by
//! `decode_select_parity_probe`:
//!
//! * **device** (default): `qsa_topk_rows` -> `qsa_expand_sel`. No host
//!   transfer at all.
//! * **host** (`ATLAS_QSA_HOST_TOPK=1`, the same A/B switch the prefill path
//!   uses): D2H every score, sort on the CPU, H2D the expansion. The D2H is
//!   `copy_d2h_on_stream`, i.e. a full stream drain, and it ran ONCE PER QSA
//!   LAYER PER SEQUENCE PER STEP — 12 full-attention layers times C
//!   sequences of drains per decode step.
//!
//! `n_sel` never needs reading back: it is `block_topk * ratio + (visible -
//! complete * ratio)`, all four terms host-known. That is what lets the
//! selection stay entirely on device while the gather launch, the identity
//! table and the attention call still get their host-side sizes.

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;

use super::QsaIndexer;
use crate::layers::ops;

impl QsaIndexer {
    /// Selected-token count for one decode step. Pure host arithmetic: the
    /// top-k always yields exactly `block_topk` complete blocks (the caller
    /// only reaches here when `complete > block_topk`), each expanding to
    /// `ratio` tokens, plus the incomplete tail `complete*ratio..visible`.
    pub(super) fn decode_n_sel(&self, complete: usize, visible: usize) -> u32 {
        self.budget + (visible - complete * self.ratio as usize) as u32
    }

    /// Is the device selection path armed?
    ///
    /// Kill switch: `ATLAS_QSA_HOST_TOPK=1` forces the host path. This is the
    /// switch #820 introduced for the prefill device top-k; decode obeys the
    /// same one rather than adding a second knob for the same decision.
    fn decode_device_select(&self) -> bool {
        static HOST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let host = *HOST.get_or_init(|| std::env::var("ATLAS_QSA_HOST_TOPK").as_deref() == Ok("1"));
        !host && self.k_topk_rows_k.0 != 0 && self.k_expand_sel_k.0 != 0
    }

    /// Build `sel_dev` + `seq_len_dev` + the identity table for one decode
    /// step, and return `n_sel`.
    ///
    /// `table_len` is the caller's memo of how many identity entries are
    /// already uploaded (host path only; the device path rewrites them every step
    /// for free inside the expansion kernel).
    pub(super) fn decode_build_sel(
        &self,
        table_len: &mut usize,
        complete: usize,
        visible: usize,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<u32> {
        // `table_dev` is sized `ceil((budget+ratio)/8)` entries, so a page
        // smaller than 8 slots would overrun it. Every served config uses 16;
        // refusing beats a silent out-of-bounds write.
        anyhow::ensure!(
            block_size >= 8,
            "QSA: decode block_size {block_size} < 8 — the identity table is \
             sized for pages of at least 8 slots"
        );
        if self.decode_device_select() {
            self.decode_build_sel_device(table_len, complete, visible, block_size, gpu, stream)
        } else {
            self.decode_build_sel_host(table_len, complete, visible, block_size, gpu, stream)
        }
    }

    /// Device path: radix top-k, then a shared-memory ascending sort and the
    /// expansion, both on the stream. Zero D2H, zero H2D.
    pub(super) fn decode_build_sel_device(
        &self,
        table_len: &mut usize,
        complete: usize,
        visible: usize,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<u32> {
        let n_sel = self.decode_n_sel(complete, visible);
        let pos = visible - 1;
        ops::qsa_topk_rows(
            gpu,
            self.k_topk_rows_k,
            self.scores_dev,
            self.lists_dev,
            1,
            complete as u32, // single row: stride is the row itself
            self.block_topk,
            pos as u32,
            self.ratio,
            stream,
        )
        .context("QSA decode top-k (device)")?;
        let sel_cap = self.budget as usize + self.ratio as usize;
        ops::qsa_expand_sel(
            gpu,
            self.k_expand_sel_k,
            self.lists_dev,
            spark_runtime::gpu::DevicePtr(0), // rows==1: first_pos carries it
            self.sel_dev,
            self.seq_len_dev,
            self.table_dev,
            1,
            self.block_topk,
            self.ratio,
            pos as u32,
            sel_cap as u32,
            sel_cap.div_ceil(8) as u32,
            block_size,
            stream,
        )
        .context("QSA decode sort+expand (device)")?;
        *table_len = (n_sel as usize).div_ceil(block_size as usize);
        Ok(n_sel)
    }

    /// Host path (`ATLAS_QSA_HOST_TOPK=1`): the original implementation, kept
    /// verbatim as the A/B reference the parity probe compares against.
    pub(super) fn decode_build_sel_host(
        &self,
        table_len: &mut usize,
        complete: usize,
        visible: usize,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<u32> {
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
        debug_assert_eq!(n_sel, self.decode_n_sel(complete, visible));

        let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async(&sel_bytes, self.sel_dev, stream)?;

        let pages = (n_sel as usize).div_ceil(block_size as usize);
        if *table_len < pages {
            let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&ident, self.table_dev, stream)?;
            *table_len = pages;
        }
        gpu.copy_h2d_async(&(n_sel as i32).to_le_bytes(), self.seq_len_dev, stream)?;
        Ok(n_sel)
    }

    /// Run BOTH decode selection paths over the SAME device-resident block
    /// scores and hand back what each wrote.
    ///
    /// Returns `(device_sel, host_sel, device_seq_len, host_seq_len)`. The
    /// device path must reproduce the host path byte for byte — the selected
    /// SET, the ascending order, the expansion and the tail — so the caller
    /// asserts plain equality. Exposed (rather than kept inside `#[cfg(test)]`)
    /// so the `qsa_decode_select_parity` example can drive it on real hardware
    /// without widening any field's visibility.
    pub fn decode_select_parity_probe(
        &self,
        scores: &[f32],
        visible: usize,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<(Vec<i32>, Vec<i32>, u32, u32)> {
        let ratio = self.ratio as usize;
        let complete = visible / ratio;
        anyhow::ensure!(
            scores.len() == complete,
            "parity probe: {} scores for visible {visible} (ratio {ratio}) — \
             expected {complete}",
            scores.len()
        );
        anyhow::ensure!(
            complete > self.block_topk as usize,
            "parity probe: complete {complete} <= block_topk {} — decode \
             early-outs there, nothing to compare",
            self.block_topk
        );
        anyhow::ensure!(
            complete <= self.max_tokens / ratio,
            "parity probe: {complete} blocks exceeds the scores buffer"
        );
        anyhow::ensure!(
            self.k_topk_rows_k.0 != 0 && self.k_expand_sel_k.0 != 0,
            "parity probe: this target ships no qsa_topk_rows/qsa_expand_sel"
        );

        let bytes: Vec<u8> = scores.iter().flat_map(|v| v.to_le_bytes()).collect();
        let read_sel = |n: u32| -> Result<Vec<i32>> {
            let mut b = vec![0u8; n as usize * 4];
            gpu.copy_d2h(self.sel_dev, &mut b)?;
            Ok(b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_seq_len = || -> Result<u32> {
            let mut b = [0u8; 4];
            gpu.copy_d2h(self.seq_len_dev, &mut b)?;
            Ok(i32::from_le_bytes(b) as u32)
        };
        // Poison both outputs between runs so a path that fails to write is
        // caught as a mismatch rather than reading the other path's result.
        let poison: Vec<u8> = vec![0xEE; (self.budget as usize + ratio) * 4];

        gpu.copy_h2d_async(&poison, self.sel_dev, stream)?;
        gpu.copy_h2d_async(&bytes, self.scores_dev, stream)?;
        let mut tl = 0usize;
        let n_dev =
            self.decode_build_sel_device(&mut tl, complete, visible, block_size, gpu, stream)?;
        gpu.synchronize(stream)?;
        let dev = read_sel(n_dev)?;
        let dev_len = read_seq_len()?;

        gpu.copy_h2d_async(&poison, self.sel_dev, stream)?;
        gpu.copy_h2d_async(&bytes, self.scores_dev, stream)?;
        let mut tl = 0usize;
        let n_host =
            self.decode_build_sel_host(&mut tl, complete, visible, block_size, gpu, stream)?;
        gpu.synchronize(stream)?;
        let host = read_sel(n_host)?;
        let host_len = read_seq_len()?;

        anyhow::ensure!(
            n_dev == n_host,
            "parity probe: n_sel {n_dev} (device) vs {n_host} (host)"
        );
        Ok((dev, host, dev_len, host_len))
    }
}
