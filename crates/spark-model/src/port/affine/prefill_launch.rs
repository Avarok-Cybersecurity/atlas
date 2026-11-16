// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/w4a8.zig:9-53 (the quantizer and prompt launch geometry, the flag reset before the quantizer)
// and zig/src/cuda/qlinear.zig:249-276 (the prompt GEMM's projection descriptors and shared-memory size).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust over `GpuBackend` and the op cache; the flag pointer passed to
// the GEMMs as a device-side gate instead of a host readback; `_many` grids on the int8 path; the SwiGLU fusion.

//! Responsibility: kernel handles and one launcher per entry point of `kernels/ada/common/affine_prefill_ada.cu`
//! and `affine_swiglu_sums_ada.cu` (lane L17b). Launchers check the shape rules the kernels rely on and never
//! decide a route; `super::run` and its siblings do.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{
    ACT_GROUPS_PER_CTA, ACT_THREADS, BF16_SMEM_BYTES, GROUP, INT8_SMEM_BYTES, Int8Act, Packed,
    THREADS, encode_parts, grid_blocks, i32_of, row_band,
};

/// One `(module, entry)` pair.
pub type Entry = (&'static str, &'static str);

/// The prompt GEMM module (`kernels/ada/common/affine_prefill_ada.cu`).
pub const GEMM_MODULE: &str = "affine_prefill_ada";
/// The activation-side module (`kernels/ada/common/affine_swiglu_sums_ada.cu`).
pub const ACT_MODULE: &str = "affine_swiglu_sums_ada";
pub const BF16_GEMM: Entry = (GEMM_MODULE, "affine4_prefill_bf16_ada");
pub const BF16_GEMM_MANY: Entry = (GEMM_MODULE, "affine4_prefill_bf16_many_ada");
pub const INT8_GEMM: Entry = (GEMM_MODULE, "affine4_w4a8_prefill_ada");
pub const INT8_GEMM_MANY: Entry = (GEMM_MODULE, "affine4_w4a8_prefill_many_ada");
pub const ACT_QUANT: Entry = (ACT_MODULE, "affine4_act_quant_ada");
pub const SWIGLU_ACT_QUANT: Entry = (ACT_MODULE, "affine4_swiglu_act_quant_ada");
/// Every entry, in lookup order ([`kernels_present`]).
pub const ENTRIES: [Entry; 6] = [
    BF16_GEMM,
    BF16_GEMM_MANY,
    INT8_GEMM,
    INT8_GEMM_MANY,
    ACT_QUANT,
    SWIGLU_ACT_QUANT,
];

/// The kernel where the target compiled its module, through the backend's op cache; `None` elsewhere (no
/// lookup is issued for an absent module, so a non-Ada boot audit sees nothing new).
pub fn handle(gpu: &dyn GpuBackend, (module, entry): Entry) -> Option<KernelHandle> {
    if !gpu.has_module(module) {
        return None;
    }
    gpu.op_cache()
        .kernel(gpu, module, entry)
        .ok()
        .filter(|h| h.0 != 0)
}

fn need(gpu: &dyn GpuBackend, e: Entry) -> Result<KernelHandle> {
    handle(gpu, e).with_context(|| format!("affine prefill kernel {}::{} is absent", e.0, e.1))
}

/// Resolve every entry. The loader calls this at construction so the lookups sit on the eager build path the
/// boot kernel audit sees; true when all six resolve.
pub fn kernels_present(gpu: &dyn GpuBackend) -> bool {
    ENTRIES.iter().all(|e| handle(gpu, *e).is_some())
}

fn aligned(p: DevicePtr, to: u64) -> bool {
    p.0.is_multiple_of(to)
}

fn act_grid(m: usize, k: usize) -> Result<[u32; 3]> {
    let y = (k / GROUP).div_ceil(ACT_GROUPS_PER_CTA);
    Ok([u32::try_from(m)?, u32::try_from(y)?, 1])
}

/// Quantize `m` rows of `x` ([m, k] BF16, row stride `ldx`) to int8 per 64-input group into `act`, clearing
/// and then raising `act.flags` word 0 on an exceptional group.
pub fn quantize_int8_rows(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    ldx: usize,
    m: usize,
    k: usize,
    act: &Int8Act,
    stream: u64,
) -> Result<()> {
    let h = need(gpu, ACT_QUANT)?;
    ensure!(
        k.is_multiple_of(GROUP) && ldx >= k,
        "affine int8 quantize: k {k}, ldx {ldx}"
    );
    gpu.memset_async(act.flags, 0, 4, stream)?;
    KernelLaunch::new(gpu, h)
        .grid(act_grid(m, k)?)
        .block([ACT_THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(act.xq)
        .arg_ptr(act.xa)
        .arg_ptr(act.xs)
        .arg_i32(i32_of(k, "k")?)
        .arg_i32(i32_of(ldx, "ldx")?)
        .arg_ptr(act.flags)
        .launch(stream)
}

/// h = bf16(silu(gate) * up) into `h` (row stride `ld_h`; may alias `gate` when `ld_h == ld_in`; null skips the
/// store), and h's int8 codes, scales and sums into `act`, in one launch. Byte-equal to `moe_silu_mul` then
/// [`quantize_int8_rows`].
#[allow(clippy::too_many_arguments)]
pub fn swiglu_quantize(
    gpu: &dyn GpuBackend,
    gate: DevicePtr,
    up: DevicePtr,
    ld_in: usize,
    h: DevicePtr,
    ld_h: usize,
    m: usize,
    k: usize,
    act: &Int8Act,
    stream: u64,
) -> Result<()> {
    let kh = need(gpu, SWIGLU_ACT_QUANT)?;
    ensure!(
        k.is_multiple_of(GROUP) && ld_in >= k && (h.is_null() || ld_h >= k),
        "affine SwiGLU quantize: k {k}, ld_in {ld_in}, ld_h {ld_h}"
    );
    ensure!(
        h != gate || ld_h == ld_in,
        "affine SwiGLU quantize: h may alias gate only with the same row stride"
    );
    gpu.memset_async(act.flags, 0, 4, stream)?;
    KernelLaunch::new(gpu, kh)
        .grid(act_grid(m, k)?)
        .block([ACT_THREADS, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(h)
        .arg_i32(i32_of(ld_in, "ld_in")?)
        .arg_i32(i32_of(ld_h, "ld_h")?)
        .arg_ptr(act.xq)
        .arg_ptr(act.xa)
        .arg_ptr(act.xs)
        .arg_i32(i32_of(k, "k")?)
        .arg_ptr(act.flags)
        .launch(stream)
}

fn check_bf16_io(w: &Packed, x: DevicePtr, ldx: usize) -> Result<()> {
    ensure!(
        w.servable() && ldx >= w.k && ldx.is_multiple_of(8),
        "affine BF16 prompt GEMM: {w:?}, ldx {ldx}"
    );
    ensure!(
        aligned(x, 16),
        "affine prompt GEMM: x must be 16-byte aligned (cp.async), got {:#x}",
        x.0
    );
    Ok(())
}

/// BF16 prompt GEMM `out[m, n] = x[m, k] . W^T` (x row stride `ldx`). A non-null `run_if` (an `Int8Act::flags`)
/// runs it only when that flag is raised: the device-side fallback of the int8 GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(
    gpu: &dyn GpuBackend,
    w: &Packed,
    x: DevicePtr,
    ldx: usize,
    m: usize,
    out: DevicePtr,
    run_if: DevicePtr,
    stream: u64,
) -> Result<()> {
    let h = need(gpu, BF16_GEMM)?;
    check_bf16_io(w, x, ldx)?;
    ensure!(
        aligned(out, 4),
        "affine prompt GEMM: out must be 4-byte aligned"
    );
    KernelLaunch::new(gpu, h)
        .grid([u32::try_from(grid_blocks(m, w.n))?, 1, 1])
        .block([THREADS, 1, 1])
        .shared_mem(BF16_SMEM_BYTES)
        .arg_ptr(x)
        .arg_ptr(w.words)
        .arg_ptr(w.scales)
        .arg_ptr(w.biases)
        .arg_ptr(out)
        .arg_i32(i32_of(m, "m")?)
        .arg_i32(i32_of(w.n, "n")?)
        .arg_i32(i32_of(w.k, "k")?)
        .arg_i32(i32_of(w.npad, "npad")?)
        .arg_i32(i32_of(ldx, "ldx")?)
        .arg_i32(i32_of(row_band(m, w.k), "band")?)
        .arg_ptr(run_if)
        .launch(stream)
}

/// W4A8 int8 prompt GEMM over `act` (filled by [`quantize_int8_rows`] or [`swiglu_quantize`] for these `m`
/// rows); skipped on the device when `act.flags` word 0 is raised.
pub fn gemm_int8(
    gpu: &dyn GpuBackend,
    w: &Packed,
    act: &Int8Act,
    m: usize,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let h = need(gpu, INT8_GEMM)?;
    ensure!(w.servable(), "affine int8 prompt GEMM: {w:?}");
    ensure!(
        aligned(act.xq, 16),
        "affine int8 prompt GEMM: xq must be 16-byte aligned"
    );
    KernelLaunch::new(gpu, h)
        .grid([u32::try_from(grid_blocks(m, w.n))?, 1, 1])
        .block([THREADS, 1, 1])
        .shared_mem(INT8_SMEM_BYTES)
        .arg_ptr(act.xq)
        .arg_ptr(act.xa)
        .arg_ptr(act.xs)
        .arg_ptr(w.words)
        .arg_ptr(w.scales)
        .arg_ptr(w.biases)
        .arg_ptr(out)
        .arg_i32(i32_of(m, "m")?)
        .arg_i32(i32_of(w.n, "n")?)
        .arg_i32(i32_of(w.k, "k")?)
        .arg_i32(i32_of(w.npad, "npad")?)
        .arg_i32(i32_of(row_band(m, w.k), "band")?)
        .arg_ptr(act.flags)
        .launch(stream)
}

/// BF16 prompt GEMM of up to four projections that share `x` in one grid; `run_if` as in [`gemm_bf16`].
pub fn gemm_bf16_many(
    gpu: &dyn GpuBackend,
    members: &[(Packed, DevicePtr)],
    x: DevicePtr,
    ldx: usize,
    m: usize,
    run_if: DevicePtr,
    stream: u64,
) -> Result<()> {
    let h = need(gpu, BF16_GEMM_MANY)?;
    let (parts, blocks) = encode_parts(members, m)?;
    let k = members[0].0.k;
    check_bf16_io(&members[0].0, x, ldx)?;
    let (mb, kb, lb, gb) = (
        i32_of(m, "m")?.to_le_bytes(),
        i32_of(k, "k")?.to_le_bytes(),
        i32_of(ldx, "ldx")?.to_le_bytes(),
        i32_of(row_band(m, k), "band")?.to_le_bytes(),
    );
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Bytes(&parts),
        KernelArg::Bytes(&mb),
        KernelArg::Bytes(&kb),
        KernelArg::Bytes(&lb),
        KernelArg::Bytes(&gb),
        KernelArg::Buffer(run_if),
    ];
    let grid = [u32::try_from(blocks)?, 1, 1];
    gpu.launch_typed(h, grid, [THREADS, 1, 1], BF16_SMEM_BYTES, stream, &args)
}

/// W4A8 int8 prompt GEMM of up to four projections over one quantized input.
pub fn gemm_int8_many(
    gpu: &dyn GpuBackend,
    members: &[(Packed, DevicePtr)],
    act: &Int8Act,
    m: usize,
    stream: u64,
) -> Result<()> {
    let h = need(gpu, INT8_GEMM_MANY)?;
    let (parts, blocks) = encode_parts(members, m)?;
    let k = members[0].0.k;
    ensure!(
        aligned(act.xq, 16),
        "affine int8 prompt GEMM: xq must be 16-byte aligned"
    );
    let (mb, kb, gb) = (
        i32_of(m, "m")?.to_le_bytes(),
        i32_of(k, "k")?.to_le_bytes(),
        i32_of(row_band(m, k), "band")?.to_le_bytes(),
    );
    let args = [
        KernelArg::Buffer(act.xq),
        KernelArg::Buffer(act.xa),
        KernelArg::Buffer(act.xs),
        KernelArg::Bytes(&parts),
        KernelArg::Bytes(&mb),
        KernelArg::Bytes(&kb),
        KernelArg::Bytes(&gb),
        KernelArg::Buffer(act.flags),
    ];
    let grid = [u32::try_from(blocks)?, 1, 1];
    gpu.launch_typed(h, grid, [THREADS, 1, 1], INT8_SMEM_BYTES, stream, &args)
}
