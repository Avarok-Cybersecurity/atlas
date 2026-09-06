// SPDX-License-Identifier: AGPL-3.0-only

//! Dense single-matrix GEMM launchers (`exl3_gemm_k{K}_cb{cb}_sh{S}_*`) —
//! split from `exl3_matmul.rs` for the 500-LoC cap. Three entry points over
//! ONE launch body:
//!
//! * [`exl3_gemm`] — raw f16 A, f16 or f32 C (upstream's kernel).
//! * [`exl3_gemm_abf16`] — raw BF16 A converted inside the input-Hadamard
//!   prologue (`_abf16` twins), f32 C.
//! * [`exl3_gemm_abf16_obf16`] — the above PLUS a BF16 copy of C stored by
//!   the output-Hadamard epilogue into a pitched destination (`_abf16_obf16`
//!   twins, two extra kernel arguments).
//!
//! Both fused twins are bit-identical to the converter-bracketed
//! [`exl3_gemm`] by construction: the prologue applies `exl3_bf16_to_f16`'s
//! arithmetic to every element it loads, the epilogue applies
//! `exl3_f32_to_bf16[_2d]`'s rounding to the very register values it stores
//! as f32. What they save is launches, not arithmetic — the win is a
//! hypothesis until the GPU A/B runs.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{
    BLOCKDIM, EXL3_SMEM_MAX, TILESIZE_K, TILESIZE_N, c_suffix, ensure_k_cb, raise_smem_once,
    resolve_gemm_shape,
};

/// Native EXL3 GEMM (any m; the kernel chunks rows internally in slabs of
/// 16). `a`: RAW fp16 `[m,k]`; `b_trellis`: u16 `[k/16, n/16, 16K]`; `c`:
/// fp16/fp32 `[m,n]` per `c_fp32` (no pre-zeroing needed). Cooperative
/// launch, grid `= max(min(k/TK * n/TN, sm_count), 1)`.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemm(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    b_trellis: DevicePtr,
    c: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    c_fp32: bool,
    locks: DevicePtr,
    suh: DevicePtr,
    a_had: DevicePtr,
    svh: DevicePtr,
    force_shape: Option<usize>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    exl3_gemm_impl(
        gpu,
        a,
        b_trellis,
        c,
        m,
        k,
        n,
        k_bits,
        cb,
        c_fp32,
        false,
        None,
        locks,
        suh,
        a_had,
        svh,
        force_shape,
        sm_count,
        stream,
    )
}

/// [`exl3_gemm`] over a RAW BF16 activation: the `_abf16` kernel twins convert
/// BF16 -> f16 inside the input-Hadamard prologue (bit-identical to
/// `exl3_bf16_to_f16` + `exl3_gemm`), saving the separate ingress launch. f32 C
/// only (the dense decode arm's contract); `a_had` must NOT alias `a` here.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemm_abf16(
    gpu: &dyn GpuBackend,
    a_bf16: DevicePtr,
    b_trellis: DevicePtr,
    c: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    locks: DevicePtr,
    suh: DevicePtr,
    a_had: DevicePtr,
    svh: DevicePtr,
    force_shape: Option<usize>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        a_bf16.0 != a_had.0,
        "exl3_gemm_abf16: A_had must not alias the BF16 activation"
    );
    exl3_gemm_impl(
        gpu,
        a_bf16,
        b_trellis,
        c,
        m,
        k,
        n,
        k_bits,
        cb,
        true,
        true,
        None,
        locks,
        suh,
        a_had,
        svh,
        force_shape,
        sm_count,
        stream,
    )
}

/// [`exl3_gemm_abf16`] whose epilogue ALSO stores the final C as BF16:
/// `dst_bf16[r * ld_dst + col] = bf16_rn(C[r, col])` for every `r < m`,
/// `col < n` — the `_abf16_obf16` twins. Byte-identical to
/// `exl3_gemm_abf16` + `exl3_f32_to_bf16[_2d]` by construction (the same
/// `__float2bfloat16_rn` on the same values), one launch fewer. `c` (f32
/// `[m, n]`) is still written and must not alias `dst_bf16`; `ld_dst >= n`,
/// in elements (`ld_dst == n` is the contiguous case). Elements past `n` in
/// each destination row are not touched.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemm_abf16_obf16(
    gpu: &dyn GpuBackend,
    a_bf16: DevicePtr,
    b_trellis: DevicePtr,
    c: DevicePtr,
    dst_bf16: DevicePtr,
    ld_dst: usize,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    locks: DevicePtr,
    suh: DevicePtr,
    a_had: DevicePtr,
    svh: DevicePtr,
    force_shape: Option<usize>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        a_bf16.0 != a_had.0,
        "exl3_gemm_abf16_obf16: A_had must not alias the BF16 activation"
    );
    ensure!(
        !dst_bf16.is_null() && dst_bf16.0 != c.0,
        "exl3_gemm_abf16_obf16: BF16 destination must be non-null and distinct from the f32 C"
    );
    ensure!(
        ld_dst >= n && i32::try_from(ld_dst).is_ok(),
        "exl3_gemm_abf16_obf16: destination row stride {ld_dst} must be >= n={n} and fit i32"
    );
    exl3_gemm_impl(
        gpu,
        a_bf16,
        b_trellis,
        c,
        m,
        k,
        n,
        k_bits,
        cb,
        true,
        true,
        Some((dst_bf16, ld_dst)),
        locks,
        suh,
        a_had,
        svh,
        force_shape,
        sm_count,
        stream,
    )
}

/// Kernel instance name for a dense GEMM launch; `a_bf16` selects the
/// BF16-ingress twin, `out_bf16` additionally the BF16-egress twin (both f32
/// C only; `out_bf16` implies `a_bf16`).
pub fn exl3_gemm_kernel_name(
    k_bits: u32,
    cb: u32,
    shape: usize,
    c_fp32: bool,
    a_bf16: bool,
    out_bf16: bool,
) -> String {
    let suffix = match (a_bf16, out_bf16) {
        (_, true) => "_abf16_obf16",
        (true, false) => "_abf16",
        (false, false) => "",
    };
    format!(
        "exl3_gemm_k{k_bits}_cb{cb}_sh{shape}_{}{suffix}",
        c_suffix(c_fp32)
    )
}

#[allow(clippy::too_many_arguments)]
fn exl3_gemm_impl(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    b_trellis: DevicePtr,
    c: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    c_fp32: bool,
    a_bf16: bool,
    out_bf16: Option<(DevicePtr, usize)>,
    locks: DevicePtr,
    suh: DevicePtr,
    a_had: DevicePtr,
    svh: DevicePtr,
    force_shape: Option<usize>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    ensure_k_cb(k_bits, cb)?;
    ensure!(m >= 1, "exl3_gemm: m == 0");
    ensure!(
        !a_bf16 || c_fp32,
        "exl3_gemm: the BF16-ingress twins exist for f32 C only"
    );
    ensure!(
        out_bf16.is_none() || a_bf16,
        "exl3_gemm: the BF16-egress twins exist on the BF16-ingress path only"
    );
    let shape = resolve_gemm_shape(k, n, k_bits, false, 1, 1, force_shape)?;
    let num_sms = ((k / TILESIZE_K[shape]) * (n / TILESIZE_N[shape]))
        .min(sm_count as usize)
        .max(1) as u32;
    let name = exl3_gemm_kernel_name(k_bits, cb, shape, c_fp32, a_bf16, out_bf16.is_some());
    let h = gpu.kernel("exl3_matmul", &name)?;
    raise_smem_once(gpu, h)?;
    let mut launch = KernelLaunch::new(gpu, h)
        .grid([num_sms, 1, 1])
        .block([BLOCKDIM[shape], 1, 1])
        .shared_mem(EXL3_SMEM_MAX)
        .cooperative()
        .arg_ptr(a)
        .arg_ptr(b_trellis)
        .arg_ptr(c)
        .arg_i32(m as i32)
        .arg_i32(k as i32)
        .arg_i32(n as i32)
        .arg_ptr(locks)
        .arg_ptr(suh)
        .arg_ptr(a_had)
        .arg_ptr(svh);
    if let Some((dst, ld)) = out_bf16 {
        launch = launch.arg_ptr(dst).arg_i32(ld as i32);
    }
    launch.launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_encode_the_fusion_arms() {
        assert_eq!(
            exl3_gemm_kernel_name(6, 2, 3, true, false, false),
            "exl3_gemm_k6_cb2_sh3_f32"
        );
        assert_eq!(
            exl3_gemm_kernel_name(6, 2, 3, false, false, false),
            "exl3_gemm_k6_cb2_sh3_f16"
        );
        assert_eq!(
            exl3_gemm_kernel_name(6, 2, 3, true, true, false),
            "exl3_gemm_k6_cb2_sh3_f32_abf16"
        );
        assert_eq!(
            exl3_gemm_kernel_name(6, 2, 3, true, true, true),
            "exl3_gemm_k6_cb2_sh3_f32_abf16_obf16"
        );
    }

    #[test]
    fn obf16_contract_checks() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        let gpu = MockGpuBackend::new();
        let a = gpu.alloc(2560 * 2).unwrap();
        let b = gpu.alloc(2560 * 10240 * 6 / 8).unwrap();
        let c = gpu.alloc(10240 * 4).unwrap();
        let dst = gpu.alloc(16384 * 2).unwrap();
        let locks = gpu.alloc(4096).unwrap();
        let suh = gpu.alloc(2560 * 2).unwrap();
        let a_had = gpu.alloc(2560 * 2).unwrap();
        let svh = gpu.alloc(10240 * 2).unwrap();
        let run = |dst: DevicePtr, ld: usize, a_had: DevicePtr| {
            exl3_gemm_abf16_obf16(
                &gpu, a, b, c, dst, ld, 1, 2560, 10240, 6, 2, locks, suh, a_had, svh, None, 48, 0,
            )
        };
        // Pitched (GDN arena row) and contiguous destinations both launch the
        // `_abf16_obf16` twin.
        run(dst, 16384, a_had).unwrap();
        run(dst, 10240, a_had).unwrap();
        let names: Vec<String> = gpu
            .kernel_lookups_snapshot()
            .into_iter()
            .map(|(_, f)| f)
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(
            names
                .iter()
                .all(|n| n == "exl3_gemm_k6_cb2_sh3_f32_abf16_obf16")
        );
        // Row stride narrower than n, null / aliased destination, aliased A_had.
        assert!(run(dst, 8192, a_had).is_err());
        assert!(run(DevicePtr(0), 10240, a_had).is_err());
        assert!(run(c, 10240, a_had).is_err());
        assert!(run(dst, 10240, a).is_err());
    }
}
