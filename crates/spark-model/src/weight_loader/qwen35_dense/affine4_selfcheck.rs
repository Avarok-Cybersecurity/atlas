// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time self-check of the 4-bit (`W4Linear`) routes on the real weights, and the
//! per-row fallback a failed check selects.
//!
//! **WHAT.** The first time a layer type (FFN, GDN, attention) installs 4-bit weights, each of that layer's
//! projections decodes the same four input rows twice: once as one `rows = 4` call, once as four `rows = 1`
//! calls. The `W4Linear` contract requires the two to be byte-equal (row invariance: an MTP verify row must
//! equal the decode of the same token, or speculative decoding at temperature 0 stops being exact), and
//! requires the split-K ticket region to be all zero again afterwards. The inputs are fixed dyadic BF16
//! values, so the check is deterministic and needs no reference.
//!
//! **ON FAILURE** the serve does not abort. The verdict is logged, `atlas_w4_selfcheck_failures_total`
//! counts it, and every weight of that layer type is wrapped in [`PerRow`], whose `decode(rows)` runs
//! `rows` single-row decodes: verify keeps the M = 1 bits, at the M = 1 cost per row (the existing
//! per-row path). The other layer types keep their batched decode.
//!
//! This is TensorFold's "row-invariant verify + load-time real-weight self-check",
//! implemented for every 4-bit format (the module name is the original, affine-only one). Lane L23b.

use std::collections::HashMap;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::port::w4::scratch::{self, ScratchSpec};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Scratch};
use crate::port::w4_stats::{self, W4Role};

/// Rows of the batched call the single-row decodes are compared against.
pub const ROWS: usize = 4;

/// What one weight's check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every byte of the `rows = 4` output equals the four `rows = 1` outputs, and the tickets are zero.
    Pass,
    /// The first differing BF16 output: `row` of the four, output column `col`.
    Mismatch { row: usize, col: usize },
    /// The outputs agree but the call left a split-K ticket nonzero (a later launch would read it).
    TicketsDirty,
    /// The weight's decode window is narrower than [`ROWS`]: nothing batched to compare.
    Skipped,
}

impl Outcome {
    /// The verdict selects the per-row fallback.
    pub fn failed(self) -> bool {
        matches!(self, Self::Mismatch { .. } | Self::TicketsDirty)
    }
}

/// `rows x k` BF16 inputs: multiples of 1/64 in [-1, 1), exact in BF16, varied across rows and columns.
pub fn input_rows(rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k * 2);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..rows * k {
        // splitmix64
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let v = ((z % 128) as f32 - 64.0) / 64.0;
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

/// Check one weight: `ROWS` single-row decodes against one `ROWS`-row decode, then the tickets.
/// Synchronizes `stream`; allocates and frees its own input and output buffers; reserves the shared W4
/// scratch for `w` (which the layer would reserve anyway).
pub fn row_invariance(gpu: &dyn GpuBackend, w: &dyn W4Linear, stream: u64) -> Result<Outcome> {
    if w.max_decode_rows() < ROWS {
        return Ok(Outcome::Skipped);
    }
    let (n, k) = (w.n(), w.k());
    let s = scratch::acquire(gpu, &ScratchSpec::for_linear(w), stream)?;
    let x = gpu.alloc(ROWS * k * 2)?;
    let batched = gpu.alloc(ROWS * n * 2)?;
    let single = gpu.alloc(ROWS * n * 2)?;
    let run = || -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        gpu.copy_h2d(&input_rows(ROWS, k), x)?;
        w.decode(gpu, x, ROWS, batched, &s, stream)?;
        for r in 0..ROWS {
            w.decode(
                gpu,
                x.offset(r * k * 2),
                1,
                single.offset(r * n * 2),
                &s,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let mut a = vec![0u8; ROWS * n * 2];
        let mut b = vec![0u8; ROWS * n * 2];
        let mut t = vec![0u8; scratch::TICKET_BYTES];
        gpu.copy_d2h(batched, &mut a)?;
        gpu.copy_d2h(single, &mut b)?;
        gpu.copy_d2h(s.tickets, &mut t)?;
        Ok((a, b, t))
    };
    let got = run();
    for p in [x, batched, single] {
        gpu.free(p)?;
    }
    let (a, b, t) = got?;
    if t.iter().any(|&v| v != 0) {
        // The region is shared by every 4-bit weight on the backend and must be zero on entry to the next
        // call: put it back before reporting, so the per-row fallback starts from a clean region.
        gpu.memset_async(s.tickets, 0, scratch::TICKET_BYTES, stream)?;
        gpu.synchronize(stream)?;
    }
    if let Some(i) = (0..ROWS * n).find(|&i| a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2]) {
        return Ok(Outcome::Mismatch {
            row: i / n,
            col: i % n,
        });
    }
    if t.iter().any(|&v| v != 0) {
        return Ok(Outcome::TicketsDirty);
    }
    Ok(Outcome::Pass)
}

/// A `W4Linear` whose batched decode is `rows` single-row decodes of the inner weight: the per-row path a
/// failed self-check selects. Prefill and dequant are the inner weight's.
pub struct PerRow {
    inner: W4Box,
}

/// Wrap `w` (idempotent in effect: wrapping twice gives the same bits).
pub fn per_row(w: W4Box) -> W4Box {
    std::sync::Arc::new(PerRow { inner: w })
}

impl W4Linear for PerRow {
    fn format(&self) -> W4Format {
        self.inner.format()
    }
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn k(&self) -> usize {
        self.inner.k()
    }
    fn device_bytes(&self) -> usize {
        self.inner.device_bytes()
    }
    fn max_decode_rows(&self) -> usize {
        self.inner.max_decode_rows()
    }
    fn scratch_bytes(&self, max_rows: usize) -> usize {
        self.inner.scratch_bytes(max_rows)
    }
    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            (1..=self.max_decode_rows()).contains(&rows),
            "per-row {}: decode of {rows} rows outside 1..={}",
            self.format(),
            self.max_decode_rows()
        );
        let (n, k) = (self.n(), self.k());
        for r in 0..rows {
            self.inner.decode(
                gpu,
                x.offset(r * k * 2),
                1,
                out.offset(r * n * 2),
                s,
                stream,
            )?;
        }
        Ok(())
    }
    fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<bool> {
        self.inner.prefill(gpu, x, m, out, s, stream)
    }
    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        self.inner.dequant_bf16(gpu, out, stream)
    }
}

/// The verdict of each layer type, taken on its first installed layer.
#[derive(Default)]
pub struct SelfChecks {
    per_row: HashMap<W4Role, bool>,
}

impl SelfChecks {
    /// Whether `role` failed its check (its weights are wrapped in [`PerRow`]).
    pub fn per_row(&self, role: W4Role) -> bool {
        self.per_row.get(&role).copied().unwrap_or(false)
    }

    /// Whether `role` has been checked on this load.
    pub fn checked(&self, role: W4Role) -> bool {
        self.per_row.contains_key(&role)
    }

    /// Check `boxes` the first time `role` installs (every projection of that layer), remember the verdict,
    /// and return the boxes to install: unchanged on a pass, each wrapped in [`PerRow`] on a failure (this
    /// layer and every later one of the role). `layer` names the layer in the log.
    pub fn apply<const N: usize>(
        &mut self,
        gpu: &dyn GpuBackend,
        role: W4Role,
        layer: usize,
        boxes: [W4Box; N],
        stream: u64,
    ) -> Result<[W4Box; N]> {
        if !self.checked(role) {
            let mut failed = false;
            for (i, b) in boxes.iter().enumerate() {
                let o = row_invariance(gpu, &**b, stream)?;
                if o.failed() {
                    tracing::warn!(
                        "[avarok] W4 self-check {} layer {layer} projection {i} ({} [{}x{}]): {o:?}; \
                         the {} batched rows now run one row at a time (M = 1 bits)",
                        role.label(),
                        b.format(),
                        b.n(),
                        b.k(),
                        role.label(),
                    );
                    w4_stats::note_selfcheck_failure();
                    failed = true;
                    break;
                }
                tracing::debug!(
                    "W4 self-check {} layer {layer} projection {i}: {o:?}",
                    role.label()
                );
            }
            self.per_row.insert(role, failed);
        }
        Ok(if self.per_row(role) {
            boxes.map(per_row)
        } else {
            boxes
        })
    }
}

#[cfg(test)]
#[path = "affine4_selfcheck_tests.rs"]
mod tests;
