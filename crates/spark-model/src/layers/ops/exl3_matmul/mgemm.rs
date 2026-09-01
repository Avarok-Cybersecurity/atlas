// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-matrix (MoE) arm of the native EXL3 matmul wrappers, plus the
//! BF16<->FP16/FP32 boundary converters. Split from `exl3_matmul.rs` for the
//! 500-LoC cap; see the parent module for the shared contracts (locks
//! buffer, cooperative launches, smem raise, A_had sizing).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{
    BLOCKDIM, EXL3_SMEM_MAX, TILESIZE_K, TILESIZE_N, c_suffix, ensure_k_cb, raise_smem_once,
    resolve_gemm_shape,
};

/// Pointer-table multi-matrix GEMM (MoE form). `a`: fp16 `[bszm_in, m, k]`
/// (`bszm_in == 1` broadcasts); `b_list`/`suh_list`/`svh_list`: device
/// arrays of `bszm`-many device pointers; `c`: `[bszm_out, m, n]`.
/// `b_weights != None` triggers the grouped weighted reduction: each of the
/// `num_tokens` groups of `bszm/num_tokens` contiguous slots is summed into
/// row-block `t` of C (fp32 sums only when `c_fp32` — use fp32 C for MoE).
/// `a_had_capacity_elems` is the scratch capacity in halves and MUST cover
/// `bszm*m*k` (asserted here; undersize is silent OOB in the kernel).
#[allow(clippy::too_many_arguments)]
pub fn exl3_mgemm(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    b_list: DevicePtr,
    c: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    c_fp32: bool,
    locks: DevicePtr,
    suh_list: DevicePtr,
    a_had: DevicePtr,
    a_had_capacity_elems: usize,
    svh_list: DevicePtr,
    b_indices: Option<DevicePtr>,
    b_weights: Option<DevicePtr>,
    bszm_in: usize,
    bszm_out: usize,
    min_index: i32,
    max_index: i32,
    num_tokens: usize,
    size_n_list: Option<DevicePtr>,
    c_list: Option<DevicePtr>,
    force_shape: Option<usize>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    ensure_k_cb(k_bits, cb)?;
    let bszm = bszm_in.max(bszm_out);
    ensure!(bszm >= 1 && m >= 1, "exl3_mgemm: empty batch");
    ensure!(
        a_had_capacity_elems >= bszm * m * k,
        "exl3_mgemm: A_had scratch holds {a_had_capacity_elems} halves, needs bszm*m*k = {}",
        bszm * m * k
    );
    if min_index >= 0 {
        ensure!(bszm <= 128, "exl3_mgemm: index filtering caps bszm at 128");
        // The kernel's filter block dereferences B_indices unconditionally
        // (exl3_gemm_kernel.cuh) — filtering without indices is a NULL deref.
        ensure!(
            b_indices.is_some(),
            "exl3_mgemm: min_index >= 0 (expert filtering) requires b_indices"
        );
    }
    if b_weights.is_some() {
        ensure!(
            num_tokens >= 1 && bszm.is_multiple_of(num_tokens),
            "exl3_mgemm: num_tokens ({num_tokens}) must evenly divide bszm ({bszm})"
        );
    }
    // Upstream's host guard (exl3_gemm.cu:442-446): per-matrix output
    // widths/pointers are a standalone mode — they cannot combine with
    // token grouping, expert filtering, or weighted reduction, and they
    // only make sense together.
    if size_n_list.is_some() || c_list.is_some() {
        ensure!(
            size_n_list.is_some() && c_list.is_some(),
            "exl3_mgemm: size_n_list and c_list must be passed together"
        );
        ensure!(
            num_tokens == 1 && min_index < 0 && b_weights.is_none(),
            "exl3_mgemm: size_n_list/c_list cannot combine with token grouping, \
             filtering, or weighted reduction (upstream contract)"
        );
    }
    let shape = resolve_gemm_shape(k, n, k_bits, true, bszm_in, bszm_out, force_shape)?;

    // Upstream's caller-side grid computation (exl3_gemm.cu lines 599-605) —
    // the selector's own num_sms write is dead there and not reproduced.
    let total_sms = sm_count as usize;
    let tiles = ((k / TILESIZE_K[shape]) * (n / TILESIZE_N[shape])).max(1);
    let mut num_sms = tiles;
    if num_sms * bszm > total_sms {
        num_sms = (total_sms / bszm).max(1);
    }
    if num_sms <= total_sms && tiles / num_sms > 48 {
        num_sms = total_sms.min(num_sms * 2);
    }
    let concurrency = (total_sms / num_sms).min(bszm).max(1);

    let name = format!("exl3_mgemm_k{k_bits}_cb{cb}_sh{shape}_{}", c_suffix(c_fp32));
    let h = gpu.kernel("exl3_matmul", &name)?;
    raise_smem_once(gpu, h)?;
    KernelLaunch::new(gpu, h)
        .grid([num_sms as u32, 1, concurrency as u32])
        .block([BLOCKDIM[shape], 1, 1])
        .shared_mem(EXL3_SMEM_MAX)
        .cooperative()
        .arg_ptr(a)
        .arg_ptr(b_list)
        .arg_ptr(c)
        .arg_i32(m as i32)
        .arg_i32(k as i32)
        .arg_i32(n as i32)
        .arg_ptr(locks)
        .arg_ptr(suh_list)
        .arg_ptr(a_had)
        .arg_ptr(svh_list)
        .arg_ptr(b_indices.unwrap_or(DevicePtr::NULL))
        .arg_ptr(b_weights.unwrap_or(DevicePtr::NULL))
        .arg_i32(bszm_in as i32)
        .arg_i32(bszm_out as i32)
        .arg_i32(min_index)
        .arg_i32(max_index)
        .arg_i32(num_tokens as i32)
        .arg_ptr(size_n_list.unwrap_or(DevicePtr::NULL))
        .arg_ptr(c_list.unwrap_or(DevicePtr::NULL))
        .launch(stream)
}

fn convert_launch(
    gpu: &dyn GpuBackend,
    name: &str,
    input: DevicePtr,
    out: DevicePtr,
    n_elems: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", name)?;
    let grid = div_ceil(n_elems as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(out)
        .arg_u64(n_elems as u64)
        .launch(stream)
}

/// BF16 -> FP16 activation ingress (plain launch). |v| > 65504 saturates.
pub fn exl3_bf16_to_f16(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    out: DevicePtr,
    n_elems: usize,
    stream: u64,
) -> Result<()> {
    convert_launch(gpu, "exl3_bf16_to_f16", input, out, n_elems, stream)
}

/// FP16 C readback -> BF16 (plain launch).
pub fn exl3_f16_to_bf16(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    out: DevicePtr,
    n_elems: usize,
    stream: u64,
) -> Result<()> {
    convert_launch(gpu, "exl3_f16_to_bf16", input, out, n_elems, stream)
}

/// FP32 C readback -> BF16 (plain launch; preferred C dtype).
pub fn exl3_f32_to_bf16(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    out: DevicePtr,
    n_elems: usize,
    stream: u64,
) -> Result<()> {
    convert_launch(gpu, "exl3_f32_to_bf16", input, out, n_elems, stream)
}
