// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/affine4_kernels.zig:127-144 (`decode`, `decodeMany`: rows <= 16, the input's group sums
// shared by the projections of one input). Copyright 2026 TensorFold contributors. Licensed under the
// Apache License, Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT
// License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust as an implementation of Atlas's `W4Linear` contract;
// the group sums and slice partials are carved from the caller's `W4Scratch` workspace; `decode_pair`
// recognises an affine partner through `W4Linear::as_any`.

//! Responsibility: [`AffineLinear`], the affine 4-bit implementation of `W4Linear` (contract 3.7.2 of the
//! port plan), and [`release`], which frees an affine `W4Box`'s device buffers once.
//!
//! Decode (rows 1..=16): one group-sums launch, then the GEMV (`kernels::gemv`). The scratch it uses is
//! `W4Scratch::ws` only (`ws_bytes` checked): the sums at offset 0, the slice partials after them. The
//! weight's split-K tickets are its own buffer (zero at rest), so `W4Scratch::tickets` is untouched;
//! `xq` and `xs` are left to lane L17b's prompt GEMM. Row r's bits do not depend on `rows`.
//!
//! `decode_pair` of two affine weights with the same K and group computes the group sums once and runs
//! one `_seg` launch when the workspace holds both weights' partials, else two GEMVs over the same sums;
//! both give the bits of two `decode` calls. Any other partner falls back to two `decode` calls.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels;
use super::weight::{AffineWeight, MAX_DECODE_ROWS, align_scratch};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Scratch};

/// The `W4Linear` face of an [`AffineWeight`] (4-bit codes only).
#[derive(Debug)]
pub struct AffineLinear {
    w: AffineWeight,
    /// Set once by [`release`]; every later call is refused instead of reading freed memory.
    released: AtomicBool,
}

/// The affine descriptor behind `w`, if `w` is a live [`AffineLinear`].
pub fn affine_of(w: &dyn W4Linear) -> Option<AffineWeight> {
    let a = w.as_any()?.downcast_ref::<AffineLinear>()?;
    (!a.released.load(Ordering::Acquire)).then_some(a.w)
}

/// Free the device buffers of an affine `W4Box`. Returns `false` (and frees nothing) when `w` is not an
/// [`AffineLinear`] or was released before, so a second call is harmless; calls on a released weight
/// fail. The caller guarantees no launch still reads the weight.
pub fn release(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<bool> {
    let Some(a) = w.as_any().and_then(|x| x.downcast_ref::<AffineLinear>()) else {
        return Ok(false);
    };
    if a.released.swap(true, Ordering::AcqRel) {
        return Ok(false);
    }
    a.w.release(gpu)?;
    Ok(true)
}

impl AffineLinear {
    /// Box `w` as a `W4Box`.
    pub fn boxed(w: AffineWeight) -> W4Box {
        std::sync::Arc::new(Self {
            w,
            released: AtomicBool::new(false),
        })
    }

    /// The descriptor (device pointers, shape, handles).
    pub fn weight(&self) -> &AffineWeight {
        &self.w
    }

    fn live(&self) -> Result<()> {
        ensure!(
            !self.released.load(Ordering::Acquire),
            "affine: [{}x{}] was released",
            self.w.n,
            self.w.k
        );
        Ok(())
    }

    fn carve(
        &self,
        s: &W4Scratch,
        rows: usize,
        work_bytes: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let sums = self.w.sums_bytes(rows);
        let need = sums + align_scratch(work_bytes);
        ensure!(
            !s.ws.is_null() && s.ws_bytes >= need,
            "affine: decode of [{}x{}] at {rows} rows needs {need} scratch bytes, the workspace has {}",
            self.w.n,
            self.w.k,
            s.ws_bytes
        );
        Ok((s.ws, s.ws.offset(sums)))
    }
}

fn check_rows(rows: usize) -> Result<()> {
    ensure!(
        (1..=MAX_DECODE_ROWS).contains(&rows),
        "affine: decode takes 1..={MAX_DECODE_ROWS} rows, got {rows}"
    );
    Ok(())
}

fn check_io(x: DevicePtr, out: DevicePtr) -> Result<()> {
    ensure!(
        x.0.is_multiple_of(16),
        "affine: the input must be 16-byte aligned (cp.async), got {:#x}",
        x.0
    );
    ensure!(
        out.0.is_multiple_of(4),
        "affine: the output must be 4-byte aligned, got {:#x}",
        out.0
    );
    Ok(())
}

impl W4Linear for AffineLinear {
    fn format(&self) -> W4Format {
        W4Format::Affine {
            bits: self.w.bits,
            group: self.w.group,
        }
    }

    fn n(&self) -> usize {
        self.w.n
    }

    fn k(&self) -> usize {
        self.w.k
    }

    fn device_bytes(&self) -> usize {
        self.w.device_bytes()
    }

    fn max_decode_rows(&self) -> usize {
        MAX_DECODE_ROWS
    }

    fn scratch_bytes(&self, max_rows: usize) -> usize {
        self.w.scratch_bytes(max_rows)
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
        self.live()?;
        check_rows(rows)?;
        check_io(x, out)?;
        let (xs, work) = self.carve(s, rows, self.w.work_bytes())?;
        let w = &self.w;
        kernels::group_sums(gpu, &w.kernels, x, rows, w.k, w.group, xs, stream)?;
        kernels::gemv(gpu, x, xs, w, out, rows, work, stream)
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
        self.live()?;
        super::prefill::run(&self.w, gpu, x, m, out, s, self.w.int8_prefill, stream)
    }

    fn decode_pair(
        &self,
        other: &dyn W4Linear,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out_a: DevicePtr,
        out_b: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        self.live()?;
        let a = &self.w;
        // A partner shares K and the group, and is a different weight (one weight's tickets cannot
        // serve two members of one launch).
        let partner = affine_of(other)
            .filter(|b| b.k == a.k && b.group == a.group && b.bits == 4 && b.tickets != a.tickets);
        let Some(b) = partner else {
            self.decode(gpu, x, rows, out_a, s, stream)?;
            return other.decode(gpu, x, rows, out_b, s, stream);
        };
        check_rows(rows)?;
        check_io(x, out_a)?;
        check_io(x, out_b)?;
        let fused = a.sums_bytes(rows) + align_scratch(a.work_bytes() + b.work_bytes());
        if !s.ws.is_null() && s.ws_bytes >= fused {
            let (xs, work) = (s.ws, s.ws.offset(a.sums_bytes(rows)));
            kernels::group_sums(gpu, &a.kernels, x, rows, a.k, a.group, xs, stream)?;
            return kernels::gemv_seg(gpu, x, xs, &[(a, out_a), (&b, out_b)], rows, work, stream);
        }
        // Not enough room for both partials: the same sums, two launches, the same bits.
        let (xs, work) = self.carve(s, rows, a.work_bytes().max(b.work_bytes()))?;
        kernels::group_sums(gpu, &a.kernels, x, rows, a.k, a.group, xs, stream)?;
        kernels::gemv(gpu, x, xs, a, out_a, rows, work, stream)?;
        kernels::gemv(gpu, x, xs, &b, out_b, rows, work, stream)
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        self.live()?;
        kernels::dequant_bf16(gpu, &self.w, out, stream)
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}
