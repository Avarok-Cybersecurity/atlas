// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/affine4_kernels.zig:107-125 (`pack`: MLX words, scales and biases into the tiles).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust; the MLX words come from the GPU quantizer of a BF16
// checkpoint matrix (upstream loads MLX checkpoints and does not convert at load), and the staging
// buffers are freed as soon as the tiles are written.

//! Responsibility: `port::affine::quantize_from_bf16`, the 3.7.2 constructor: BF16 `[n, k]` on the
//! device -> an affine weight in the lane GEMV's tiled layout, as a `W4Box`.
//!
//! Steps, all on `stream`: quantize to MLX layout (`affine4_quantize_bf16_g<group>`), pack the codes into
//! the tiles, transpose the scales and biases to `(k/group, npad)`, zero the tickets, then synchronize
//! once to read the non-finite flag and free the MLX staging buffers. The caller frees the BF16 source
//! (the load discipline of `fp8_from_bf16`: the peak is the checkpoint plus one projection's staging).
//! Widths other than 4 bits go to `super::widths::quantize_from_bf16` (lane L17c) unchanged.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels::{self, AffineKernels};
use super::weight::{AffineWeight, GROUPS, Layout, MAX_CLIP_SEARCH};
use crate::port::w4::{W4Box, W4Opts};

/// Check `opts` and the shape for the 4-bit path, before anything is allocated.
pub fn check(n: usize, k: usize, opts: &W4Opts) -> Result<()> {
    let g = opts.group as usize;
    ensure!(
        opts.bits == 4,
        "affine: the 4-bit quantizer got bits = {}",
        opts.bits
    );
    ensure!(
        GROUPS.contains(&opts.group),
        "affine: group {g} is not one of {GROUPS:?}"
    );
    ensure!(n > 0 && k > 0, "affine: empty weight [{n}x{k}]");
    ensure!(
        k.is_multiple_of(g),
        "affine: K {k} is not a multiple of the group {g}"
    );
    ensure!(
        opts.clip_search <= MAX_CLIP_SEARCH,
        "affine: clip search {} exceeds {MAX_CLIP_SEARCH}",
        opts.clip_search
    );
    let groups = n.saturating_mul(k / g);
    ensure!(
        n <= i32::MAX as usize
            && k <= i32::MAX as usize
            && groups < u32::MAX as usize
            && n.checked_mul(k).is_some_and(|e| e <= i64::MAX as usize),
        "affine: [{n}x{k}] exceeds the quantizer's index range"
    );
    Ok(())
}

/// The quantizer's flag word: `u32::MAX` when every group was representable, else the first bad group.
pub fn check_flag(flag: u32, n: usize, k: usize, group: u16) -> Result<()> {
    if flag == u32::MAX {
        return Ok(());
    }
    let kg = (k / group as usize).max(1);
    let gid = flag as usize;
    bail!(
        "affine: weight [{n}x{k}] has a non-finite or unrepresentable group {gid} \
         (row {}, inputs {}..{}); it cannot be quantized to affine 4-bit",
        gid / kg,
        gid % kg * group as usize,
        (gid % kg + 1) * group as usize
    )
}

/// `port::affine::quantize_from_bf16`: quantize the BF16 matrix `src` (`[n, k]` row-major, n outputs of
/// k inputs) on the device into an affine weight. `opts.bits` other than 4 dispatch to lane L17c's
/// `port::affine::widths`; `opts.int8_prefill` selects lane L17b's int8 prompt GEMM in `prefill`.
/// The source stays allocated; the caller frees it.
pub fn quantize_from_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    opts: &W4Opts,
    stream: u64,
) -> Result<W4Box> {
    if opts.bits != 4 {
        return super::widths::quantize_from_bf16(gpu, src, n, k, opts, stream);
    }
    let w = quantize_weight(gpu, src, n, k, opts, stream)?;
    Ok(super::w4linear::AffineLinear::boxed(w))
}

/// The 4-bit body of [`quantize_from_bf16`], returning the bare descriptor (the microtest and the other
/// affine lanes build on it).
pub fn quantize_weight(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    opts: &W4Opts,
    stream: u64,
) -> Result<AffineWeight> {
    check(n, k, opts)?;
    let ks = AffineKernels::resolve(gpu)?;
    let kg = k / opts.group as usize;
    let mut staging = Staging::default();
    if let Err(e) = staging.alloc(gpu, n, k, kg) {
        let _ = staging.free(gpu);
        return Err(e.context("affine: staging allocation"));
    }
    let mut w = AffineWeight {
        words: DevicePtr::NULL,
        scales: DevicePtr::NULL,
        biases: DevicePtr::NULL,
        tickets: DevicePtr::NULL,
        n,
        k,
        npad: Layout::of(n, k, opts.group as usize).npad,
        group: opts.group,
        bits: 4,
        int8_prefill: opts.int8_prefill,
        kernels: ks,
    };
    let built = build(gpu, &ks, src, opts, &staging, &mut w, stream);
    if built.is_err() {
        // Drain the stream before freeing anything a queued launch may still write.
        let _ = gpu.synchronize(stream);
    }
    // The staging buffers go whatever happened; `build` synchronized before returning Ok.
    let freed = staging.free(gpu);
    if let Err(e) = built.and(freed) {
        let _ = w.release(gpu);
        return Err(e);
    }
    Ok(w)
}

/// The MLX-layout buffers the quantizer writes and the pack reads.
struct Staging {
    words: DevicePtr,
    scales: DevicePtr,
    biases: DevicePtr,
    flag: DevicePtr,
}

impl Default for Staging {
    fn default() -> Self {
        Self {
            words: DevicePtr::NULL,
            scales: DevicePtr::NULL,
            biases: DevicePtr::NULL,
            flag: DevicePtr::NULL,
        }
    }
}

impl Staging {
    fn alloc(&mut self, gpu: &dyn GpuBackend, n: usize, k: usize, kg: usize) -> Result<()> {
        self.words = gpu.alloc(n * k / 2)?;
        self.scales = gpu.alloc(n * kg * 2)?;
        self.biases = gpu.alloc(n * kg * 2)?;
        self.flag = gpu.alloc(4)?;
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first = Ok(());
        for p in [self.words, self.scales, self.biases, self.flag] {
            if !p.is_null()
                && let Err(e) = gpu.free(p)
                && first.is_ok()
            {
                first = Err(e);
            }
        }
        first
    }
}

/// Allocate `w`'s buffers and fill them from `src` through the staging buffers; synchronizes `stream`
/// and checks the quantizer's flag before returning `Ok`.
fn build(
    gpu: &dyn GpuBackend,
    ks: &AffineKernels,
    src: DevicePtr,
    opts: &W4Opts,
    st: &Staging,
    w: &mut AffineWeight,
    stream: u64,
) -> Result<()> {
    let (n, k, group) = (w.n, w.k, w.group);
    let layout = w.layout();
    let kg = w.kg();
    w.words = gpu.alloc(layout.words)?;
    w.scales = gpu.alloc(layout.scales)?;
    w.biases = gpu.alloc(layout.scales)?;
    w.tickets = gpu.alloc(w.tickets_bytes())?;
    gpu.memset_async(st.flag, 0xFF, 4, stream)?;
    gpu.memset_async(w.tickets, 0, w.tickets_bytes(), stream)?;
    kernels::quantize(
        gpu,
        ks,
        src,
        n,
        k,
        group,
        opts.clip_search,
        st.words,
        st.scales,
        st.biases,
        st.flag,
        stream,
    )?;
    kernels::pack_tiles(gpu, ks, st.words, n, k, group, w.npad, w.words, stream)?;
    kernels::transpose_pad16(gpu, ks, st.scales, n, kg, w.npad, w.scales, stream)?;
    kernels::transpose_pad16(gpu, ks, st.biases, n, kg, w.npad, w.biases, stream)?;
    gpu.synchronize(stream)?;
    let mut flag = [0u8; 4];
    gpu.copy_d2h(st.flag, &mut flag)?;
    check_flag(u32::from_le_bytes(flag), n, k, group)
}
