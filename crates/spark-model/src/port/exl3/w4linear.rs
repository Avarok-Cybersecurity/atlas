// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/exl3.zig:114-148 (`Weight`: trellis, scales, shape, strides and the shape's plan) and
// :219-284 (`matmul`: 1..128 rows, rotation then linear, partials when the plan splits), after ExLlamaV3
// (turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust as an implementation of Atlas's `W4Linear` contract;
// the rotated input and the partials are carved from the caller's `W4Scratch::ws`; each weight owns its
// zero-at-rest split counters; decode serves 1..=16 rows, prefill 17..=128 rows on the same kernel (the
// same bits) and declines above; the BF16 dequant fallback is Atlas's.

//! Responsibility: [`Exl3Linear`], the EXL3 trellis implementation of `W4Linear` (contract 3.7.2 of the
//! port plan), its upload from host bytes ([`Exl3Linear::upload`]), and [`release`], which frees an EXL3
//! `W4Box`'s device buffers once.
//!
//! A call is two launches on the caller's stream: the input rotation (`x` bf16 -> `xh` fp16 in
//! `W4Scratch::ws` at offset 0) and the linear (partials after `xh` when the plan splits K). `tickets`,
//! `xq` and `xs` of the scratch are untouched. Row r's bits do not depend on `rows`: the plan comes from
//! the shape alone (see `plan.rs` and the kernel header).
//!
//! `decode_pair` keeps the trait default (two decodes): the two members of a pair have their own input
//! scales, so their rotated inputs differ and nothing is shared.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::format::{Exl3Codebook, Exl3Parts, u16_bytes};
use super::kernels::{self, Exl3Kernels, LinearArgs};
use super::plan::{
    MAX_DECODE_ROWS, MAX_KERNEL_ROWS, Plan, counter_slots, plan_for_device, scratch_layout, strides,
};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Scratch};

/// The device buffers and launch shape of one EXL3 matrix.
#[derive(Clone, Copy, Debug)]
pub struct Exl3Weight {
    pub k: usize,
    pub n: usize,
    pub halves: u16,
    pub codebook: Exl3Codebook,
    /// The trellis words, `[k/16, n/16, 4 * halves]` u32.
    pub trellis: DevicePtr,
    /// fp16 `[k]` (sign words already expanded).
    pub suh: DevicePtr,
    /// fp16 `[n]`.
    pub svh: DevicePtr,
    /// `i32 [8 * n/128]` arrival counters, zero at rest.
    pub counters: DevicePtr,
    pub plan: Plan,
    pub kernels: Exl3Kernels,
}

impl Exl3Weight {
    pub fn trellis_bytes(&self) -> usize {
        super::format::trellis_bytes(self.k, self.n, self.halves)
    }

    pub fn counter_bytes(&self) -> usize {
        counter_slots(self.n) * 4
    }

    /// Resident bytes: trellis, both scale vectors, the counters.
    pub fn device_bytes(&self) -> usize {
        self.trellis_bytes() + 2 * self.k + 2 * self.n + self.counter_bytes()
    }

    /// The launch view of the linear and dequant kernels.
    pub fn args(&self) -> LinearArgs {
        let (stride_k, stride_nb) = strides(self.n, self.halves);
        LinearArgs {
            trellis: self.trellis,
            stride_k,
            stride_nb,
            svh: self.svh,
            counters: self.counters,
            n: self.n,
            k: self.k,
            plan: self.plan,
        }
    }

    /// Bytes of `ws` a call with up to `rows` rows uses.
    pub fn scratch_bytes(&self, rows: usize) -> usize {
        scratch_layout(self.plan, rows, self.k, self.n).total
    }

    fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.trellis, self.suh, self.svh, self.counters] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// The `W4Linear` face of an [`Exl3Weight`].
#[derive(Debug)]
pub struct Exl3Linear {
    w: Exl3Weight,
    /// Set once by [`release`]; every later call is refused instead of reading freed memory.
    released: AtomicBool,
}

fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8], what: &str) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    if let Err(e) = gpu.copy_h2d(bytes, p) {
        let _ = gpu.free(p);
        return Err(e.context(format!("exl3: upload of the {what}")));
    }
    Ok(p)
}

/// The four device buffers of one weight; every allocation made is pushed to `held` first, so the
/// caller can free them all when a later step fails.
fn upload_all(
    gpu: &dyn GpuBackend,
    trellis: &[u8],
    suh: &[u8],
    svh: &[u8],
    counters_len: usize,
    stream: u64,
    held: &mut Vec<DevicePtr>,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
    let t = upload_bytes(gpu, trellis, "trellis")?;
    held.push(t);
    let s_u = upload_bytes(gpu, suh, "input scales")?;
    held.push(s_u);
    let s_v = upload_bytes(gpu, svh, "output scales")?;
    held.push(s_v);
    let c = gpu.alloc(counters_len)?;
    held.push(c);
    gpu.memset_async(c, 0, counters_len, stream)?;
    Ok((t, s_u, s_v, c))
}

impl Exl3Linear {
    /// Check `parts`, resolve the kernels of its width and plan, and copy it to the device: the trellis
    /// byte for byte, the scales as fp16 (sign words expanded), the counters zeroed on `stream`. Not
    /// gated by the `weights_exl3` row (the microtest and tests call it directly); the route entry is
    /// [`super::from_exl3_tensors`].
    pub fn upload(gpu: &dyn GpuBackend, parts: &Exl3Parts<'_>, stream: u64) -> Result<W4Box> {
        Ok(std::sync::Arc::new(Self::upload_weight(
            gpu, parts, stream,
        )?))
    }

    /// As [`Exl3Linear::upload`], unboxed.
    pub fn upload_weight(gpu: &dyn GpuBackend, parts: &Exl3Parts<'_>, stream: u64) -> Result<Self> {
        parts.check()?;
        let plan = plan_for_device(parts.k, parts.n, gpu.sm_count().unwrap_or(1));
        let kernels = Exl3Kernels::resolve(gpu, parts.halves, parts.codebook, plan)?;
        let suh = u16_bytes(&parts.suh_bits());
        let svh = u16_bytes(&parts.svh_bits());
        let counters_len = counter_slots(parts.n) * 4;
        let mut held = Vec::with_capacity(4);
        let (trellis, suh, svh, counters) = match upload_all(
            gpu,
            parts.trellis,
            &suh,
            &svh,
            counters_len,
            stream,
            &mut held,
        ) {
            Ok(v) => v,
            Err(e) => {
                for p in held {
                    let _ = gpu.free(p);
                }
                return Err(e);
            }
        };
        Ok(Self {
            w: Exl3Weight {
                k: parts.k,
                n: parts.n,
                halves: parts.halves,
                codebook: parts.codebook,
                trellis,
                suh,
                svh,
                counters,
                plan,
                kernels,
            },
            released: AtomicBool::new(false),
        })
    }

    /// The descriptor (device pointers, shape, plan, handles).
    pub fn weight(&self) -> &Exl3Weight {
        &self.w
    }

    fn live(&self) -> Result<()> {
        ensure!(
            !self.released.load(Ordering::Acquire),
            "exl3: [{} -> {}] was released",
            self.w.k,
            self.w.n
        );
        Ok(())
    }

    fn check_io(
        &self,
        x: DevicePtr,
        out: DevicePtr,
        s: &W4Scratch,
        rows: usize,
    ) -> Result<DevicePtr> {
        ensure!(
            x.0.is_multiple_of(8),
            "exl3: the input must be 8-byte aligned, got {:#x}",
            x.0
        );
        ensure!(
            out.0.is_multiple_of(8),
            "exl3: the output must be 8-byte aligned, got {:#x}",
            out.0
        );
        let layout = scratch_layout(self.w.plan, rows, self.w.k, self.w.n);
        ensure!(
            !s.ws.is_null() && s.ws.0.is_multiple_of(16) && s.ws_bytes >= layout.total,
            "exl3: [{} -> {}] at {rows} rows needs {} aligned scratch bytes, the workspace has {}",
            self.w.k,
            self.w.n,
            layout.total,
            s.ws_bytes
        );
        Ok(s.ws.offset(layout.z_offset))
    }

    /// Rotation then linear for `rows` in `1..=128` (decode and the short prefill share it).
    fn run(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        let z = self.check_io(x, out, s, rows)?;
        let w = &self.w;
        kernels::rot_in(gpu, &w.kernels, x, w.suh, s.ws, rows, w.k, stream)?;
        kernels::linear(gpu, &w.kernels, &w.args(), s.ws, out, z, rows, stream)
    }
}

/// The descriptor behind `w`, if `w` is a live [`Exl3Linear`].
pub fn exl3_of(w: &dyn W4Linear) -> Option<Exl3Weight> {
    let e = w.as_any()?.downcast_ref::<Exl3Linear>()?;
    (!e.released.load(Ordering::Acquire)).then_some(e.w)
}

/// Free the device buffers of an EXL3 `W4Box`. Returns `false` (and frees nothing) when `w` is not an
/// [`Exl3Linear`] or was released before; calls on a released weight fail. The caller guarantees no
/// launch still reads the weight.
pub fn release(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<bool> {
    let Some(e) = w.as_any().and_then(|x| x.downcast_ref::<Exl3Linear>()) else {
        return Ok(false);
    };
    if e.released.swap(true, Ordering::AcqRel) {
        return Ok(false);
    }
    e.w.release(gpu)?;
    Ok(true)
}

impl W4Linear for Exl3Linear {
    fn format(&self) -> W4Format {
        W4Format::Exl3 {
            bpw_x100: self.w.halves * 50,
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
        ensure!(
            (1..=MAX_DECODE_ROWS).contains(&rows),
            "exl3: decode takes 1..={MAX_DECODE_ROWS} rows, got {rows}"
        );
        self.run(gpu, x, rows, out, s, stream)
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
        if m == 0 {
            return Ok(true);
        }
        if m > MAX_KERNEL_ROWS {
            // Above one launch the kernel would decode the whole weight once per 16 rows: the BF16
            // dequant and the caller's GEMM are faster.
            return Ok(false);
        }
        self.run(gpu, x, m, out, s, stream)?;
        Ok(true)
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        self.live()?;
        ensure!(
            out.0.is_multiple_of(8),
            "exl3: the dequant target must be 8-byte aligned, got {:#x}",
            out.0
        );
        let w = &self.w;
        kernels::dequant(gpu, &w.kernels, &w.args(), w.suh, out, stream)
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}
