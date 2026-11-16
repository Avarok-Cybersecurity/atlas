// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:413-439 (AffineNKernels.decode: the 16-row window and the shape-fixed
// split choice). Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version
// 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See
// TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: the port sprint's `W4Linear` contract; scratch comes
// from the caller's `W4Scratch`, and a missing split scratch runs the bit-identical unsplit form
// instead of failing; prefill declines to the caller's BF16 fallback.
//! Responsibility: `W4Linear` for [`AffineNWeight`] (2/3/5/6/8-bit MLX affine, lane L17c): decode of
//! 1..=16 rows with row-invariant bits, the scratch it needs, the declined prefill and the BF16
//! dequant the caller's fallback and the oracles use.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{AffineNWeight, kernels, layout};
use crate::port::w4::scratch::TICKET_SLOTS;
use crate::port::w4::{W4Format, W4Linear, W4Scratch};

impl AffineNWeight {
    fn launch_args(
        &self,
        x: DevicePtr,
        xs: DevicePtr,
        rows: usize,
        out: DevicePtr,
    ) -> kernels::Decode {
        kernels::Decode {
            x,
            xs,
            w: self.w,
            scales_t: self.scales_t,
            biases_t: self.biases_t,
            out,
            n: self.n,
            k: self.k,
            bits: self.bits,
            rows,
            split: None,
            grid: self.grid,
        }
    }

    /// The split form's `(work, tickets)` when the shape splits and `s` holds enough scratch. The
    /// tickets are zero on entry by the `W4Scratch` contract, and the split kernel leaves them zero.
    fn split_scratch(&self, s: &W4Scratch) -> Option<(DevicePtr, DevicePtr)> {
        let need = layout::split_work_bytes(self.n, self.k);
        let tiles = layout::tiles(self.n);
        let fits = need > 0 && tiles < layout::SPLIT_TILES_MAX && tiles <= TICKET_SLOTS;
        (fits && !s.ws.is_null() && s.ws_bytes >= need && !s.tickets.is_null())
            .then_some((s.ws, s.tickets))
    }
}

impl W4Linear for AffineNWeight {
    fn format(&self) -> W4Format {
        W4Format::Affine {
            bits: self.bits,
            group: layout::GROUP as u16,
        }
    }

    fn n(&self) -> usize {
        self.n
    }

    fn k(&self) -> usize {
        self.k
    }

    fn device_bytes(&self) -> usize {
        layout::device_bytes(self.n, self.k, self.bits)
    }

    fn max_decode_rows(&self) -> usize {
        layout::MAX_DECODE_ROWS
    }

    /// The split form's fp32 partials (`ws`); 0 when the shape never splits. The group sums (`xs`,
    /// `rows * k / 64` f32) and the tickets (`layout::tickets_bytes`) are separate fields.
    fn scratch_bytes(&self, _max_rows: usize) -> usize {
        layout::split_work_bytes(self.n, self.k)
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
        ensure!(
            (1..=layout::MAX_DECODE_ROWS).contains(&rows),
            "affine widths: decode takes 1..={} rows, got {rows}",
            layout::MAX_DECODE_ROWS
        );
        ensure!(
            !s.xs.is_null(),
            "affine widths: W4Scratch.xs is null (needs {} bytes)",
            layout::sums_bytes(rows, self.k)
        );
        kernels::launch_sums(gpu, &self.kernels, x, s.xs, rows, self.k, stream)?;
        let mut d = self.launch_args(x, s.xs, rows, out);
        d.split = self.split_scratch(s);
        kernels::launch_decode(gpu, &self.kernels, &d, stream)
    }

    /// Declined (`Ok(false)`) for every `m`: the narrow and 8-bit widths have no prompt GEMM here, so
    /// the caller dequantizes through [`W4Linear::dequant_bf16`] and runs its BF16 GEMM.
    fn prefill(
        &self,
        _gpu: &dyn GpuBackend,
        _x: DevicePtr,
        m: usize,
        _out: DevicePtr,
        _s: &W4Scratch,
        _stream: u64,
    ) -> Result<bool> {
        self.live()?;
        if m == 0 {
            bail!("affine widths: prefill of 0 rows");
        }
        Ok(false)
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        self.live()?;
        let d = self.launch_args(DevicePtr::NULL, DevicePtr::NULL, 0, out);
        kernels::launch_dequant(gpu, &self.kernels, &d, stream)
    }

    /// `Some(self)`, so `super::release` (and a later fused pair) can recognize the concrete type.
    /// `decode_pair` keeps the trait's default (two decodes), which is byte-equal by definition.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}
