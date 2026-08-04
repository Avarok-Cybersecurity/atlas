// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 (E4M3) cuBLASLt GEMM paths (row-wise + 128-block scaled).

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::*;

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16 with ROW-WISE
/// scaling (OUTER_VEC): per-output-row weight scale `weight_scale[N]` and
/// per-token activation scale `act_scale[M]`. This is the fp8 path GB10/sm_121
/// actually supports (128-block fp8 is B200-only). ~1.8× the bf16 path.
/// cuBLAS folds `A_scale[i]·B_scale[j]` into the FP32 epilogue; with D=`[N,M]`,
/// i indexes weight rows (N) and j indexes tokens (M) — exactly row-wise.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_rowwise(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        let mode = SCALE_MODE_OUTER_VEC_32F;
        set(
            DESC_A_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8 rowwise: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16, with the
/// weight per-128×128-block FP32-scaled (matches Atlas's `Fp8Weight.row_scale`
/// layout exactly) and the activation cast at unit scale (post-RMSNorm acts sit
/// in fp8's range). ~1.8× the bf16 path (152 vs 85 TFLOPS on GB10).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        // FP8 block scaling requires BOTH operands use a 128-block mode (SCALAR
        // is rejected → status 7). Weight = per-128×128 block, activation =
        // per-[token,128-of-K] VEC128 (DeepSeek block-fp8 scheme).
        let a_mode = SCALE_MODE_BLK128X128_32F;
        let b_mode = SCALE_MODE_VEC128_32F;
        set(
            DESC_A_SCALE_MODE,
            &a_mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &b_mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_block_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}
