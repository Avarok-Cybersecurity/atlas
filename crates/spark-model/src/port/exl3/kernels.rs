// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/exl3.zig:50-55 (the entry names) and :219-284 (`matmul`: the rotation launch, then the
// linear launch over (N / 128) * SK blocks of WK warps), after ExLlamaV3 (turboderp-org/exllamav3, MIT,
// Copyright (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust over Atlas's `KernelLaunch`; handles resolve through
// `try_target_kernel` (an absent module issues no lookup); bf16 in and out only, no bias, no PDL, no
// mid-M or projection-group kernels; the BF16 dequant launcher is new for Atlas.

//! Responsibility: the EXL3 route's kernel handles ([`Exl3Kernels`], one set per `(halves, codebook,
//! warps)`), the entry-point names of `kernels/ada/common/exl3_linear_ada.cu` and `exl3_hadamard_ada.cu`,
//! and one launcher per entry point.
//!
//! Every launcher is capture-safe: no allocation, no host synchronization, no host read. The only state a
//! launch leaves behind is in its outputs (the split-K counters return to zero).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::format::{Exl3Codebook, HADAMARD_DIM, width_supported};
use super::plan::{DEQUANT_SMEM, DEQUANT_THREADS, MAX_KERNEL_ROWS, Plan, ROT_THREADS, linear_smem};

/// The linear module (`kernels/ada/common/exl3_linear_ada.cu`).
pub const LINEAR_MODULE: &str = "exl3_linear_ada";
/// The rotation and dequant module (`kernels/ada/common/exl3_hadamard_ada.cu`).
pub const HADAMARD_MODULE: &str = "exl3_hadamard_ada";
/// The input rotation entry.
pub const ROT_IN_ENTRY: &str = "exl3_ada_rot_in_bf16";

/// The linear entry of one width, codebook and warp count.
pub fn linear_entry(halves: u16, cb: Exl3Codebook, wk: u32) -> String {
    format!("exl3_ada_linear_{halves}_{}_{wk}", cb.id())
}

/// The BF16 dequant entry of one width and codebook.
pub fn dequant_entry(halves: u16, cb: Exl3Codebook) -> String {
    format!("exl3_ada_dequant_bf16_{halves}_{}", cb.id())
}

/// The three handles one EXL3 matrix launches.
#[derive(Clone, Copy, Debug)]
pub struct Exl3Kernels {
    pub rot_in: KernelHandle,
    pub linear: KernelHandle,
    pub dequant: KernelHandle,
}

/// True when this build carries both EXL3 modules (the Ada targets). Issues no kernel lookup, so a
/// route's decision table can ask it on any target.
pub fn available(gpu: &dyn GpuBackend) -> bool {
    gpu.has_module(LINEAR_MODULE) && gpu.has_module(HADAMARD_MODULE)
}

fn need(gpu: &dyn GpuBackend, module: &str, entry: &str) -> Result<KernelHandle> {
    let h = crate::layers::try_target_kernel(gpu, module, entry);
    if h.0 == 0 {
        bail!(
            "exl3 kernel {module}::{entry} is not compiled for this target \
             (the EXL3 route needs the Ada build, AVAROK_TARGET_HW=ada)"
        );
    }
    Ok(h)
}

impl Exl3Kernels {
    /// Resolve the handles of one matrix's width, codebook and plan. Errors before any lookup when the
    /// modules are absent, and names the first missing entry otherwise.
    pub fn resolve(gpu: &dyn GpuBackend, halves: u16, cb: Exl3Codebook, p: Plan) -> Result<Self> {
        ensure!(
            available(gpu),
            "exl3 modules {LINEAR_MODULE} and {HADAMARD_MODULE} are not compiled for this target \
             (the EXL3 route needs the Ada build, AVAROK_TARGET_HW=ada)"
        );
        ensure!(
            width_supported(halves, cb),
            "exl3: no kernels for {halves} half-bits with {cb:?}"
        );
        ensure!(
            p.wk == 4 || p.wk == 8,
            "exl3: no kernels for {} warps",
            p.wk
        );
        Ok(Self {
            rot_in: need(gpu, HADAMARD_MODULE, ROT_IN_ENTRY)?,
            linear: need(gpu, LINEAR_MODULE, &linear_entry(halves, cb, p.wk))?,
            dequant: need(gpu, HADAMARD_MODULE, &dequant_entry(halves, cb))?,
        })
    }
}

fn i32_of(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).with_context(|| format!("exl3: {what} = {v} does not fit i32"))
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).with_context(|| format!("exl3: {what} = {v} does not fit u32"))
}

/// `xh [rows, k]` fp16 = the rotated, scaled `x [rows, k]` bf16 (`exl3_ada_rot_in_bf16`).
#[allow(clippy::too_many_arguments)]
pub fn rot_in(
    gpu: &dyn GpuBackend,
    kern: &Exl3Kernels,
    x: DevicePtr,
    suh: DevicePtr,
    xh: DevicePtr,
    rows: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(HADAMARD_DIM),
        "exl3 rot_in: k={k} is not a multiple of 128"
    );
    ensure!(rows >= 1, "exl3 rot_in: no rows");
    let grid_x = u32_of((k / HADAMARD_DIM).div_ceil(4), "rot_in grid x")?;
    KernelLaunch::new(gpu, kern.rot_in)
        .grid([grid_x, u32_of(rows, "rows")?, 1])
        .block([ROT_THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(suh)
        .arg_ptr(xh)
        .arg_i32(i32_of(rows, "rows")?)
        .arg_i32(i32_of(k, "k")?)
        .launch(stream)
}

/// The device side of one matrix a launch reads.
#[derive(Clone, Copy, Debug)]
pub struct LinearArgs {
    pub trellis: DevicePtr,
    pub stride_k: u64,
    pub stride_nb: u64,
    pub svh: DevicePtr,
    pub counters: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub plan: Plan,
}

/// `y [rows, n]` bf16 from the rotated `xh [rows, k]` (`exl3_ada_linear_<K2>_<CB>_<WK>`). `z` holds the
/// split partials when `plan.sk > 1` (ignored otherwise); `rows` in `1..=128`.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    gpu: &dyn GpuBackend,
    kern: &Exl3Kernels,
    a: &LinearArgs,
    xh: DevicePtr,
    y: DevicePtr,
    z: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=MAX_KERNEL_ROWS).contains(&rows),
        "exl3 linear: rows={rows} outside 1..={MAX_KERNEL_ROWS}"
    );
    let p = a.plan;
    ensure!(
        (a.k / 16).is_multiple_of((p.sk * p.wk) as usize),
        "exl3 linear: k={} does not split into {} x {} whole warp ranges",
        a.k,
        p.sk,
        p.wk
    );
    ensure!(
        p.sk == 1 || !z.is_null(),
        "exl3 linear: a split plan needs partial scratch"
    );
    let blocks = u32_of(a.n / HADAMARD_DIM * p.sk as usize, "linear blocks")?;
    KernelLaunch::new(gpu, kern.linear)
        .grid([blocks, 1, 1])
        .block([p.wk * 32, 1, 1])
        .shared_mem(linear_smem(p, rows))
        .arg_ptr(xh)
        .arg_ptr(a.trellis)
        .arg_u64(a.stride_k)
        .arg_u64(a.stride_nb)
        .arg_ptr(a.svh)
        .arg_ptr(y)
        .arg_ptr(if p.sk > 1 { z } else { DevicePtr::NULL })
        .arg_ptr(a.counters)
        .arg_i32(i32_of(a.n, "n")?)
        .arg_i32(p.sk as i32)
        .arg_i32(i32_of(rows, "rows")?)
        .arg_i32(i32_of(a.k, "k")?)
        .launch(stream)
}

/// The whole weight as bf16 `[n, k]` row-major (`exl3_ada_dequant_bf16_<K2>_<CB>`).
pub fn dequant(
    gpu: &dyn GpuBackend,
    kern: &Exl3Kernels,
    a: &LinearArgs,
    suh: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let grid = [
        u32_of(a.n / HADAMARD_DIM, "dequant grid x")?,
        u32_of(a.k / HADAMARD_DIM, "dequant grid y")?,
        1,
    ];
    KernelLaunch::new(gpu, kern.dequant)
        .grid(grid)
        .block([DEQUANT_THREADS, 1, 1])
        .shared_mem(DEQUANT_SMEM)
        .arg_ptr(a.trellis)
        .arg_u64(a.stride_k)
        .arg_u64(a.stride_nb)
        .arg_ptr(suh)
        .arg_ptr(a.svh)
        .arg_ptr(out)
        .arg_i32(i32_of(a.n, "n")?)
        .arg_i32(i32_of(a.k, "k")?)
        .launch(stream)
}
