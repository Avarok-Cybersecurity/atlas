// SPDX-License-Identifier: AGPL-3.0-only
//! Dense CUTLASS GEMM host wrappers (BF16 + native NVFP4 projection).

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]^T`, all BF16.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_bf16_gemm_act_weight_t(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS bf16 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Native CUTLASS NVFP4 dense projection:
/// `out[M,N] = quant_nvfp4(act[M,K]) @ weight_t[N,K]^T -> BF16`.
///
/// `weight_packed_t` and `weight_scale_t` are Atlas's transposed NVFP4
/// prefill layout: packed data `[K/2,N]`, scales `[K/16,N]`. The wrapper
/// repacks activation and scale tensors into CUTLASS's SM120 blockscaled
/// layouts in the shared CUTLASS workspace before dispatch.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_bf16_act_weight_t(
    act: u64,
    weight_packed_t: u64,
    weight_scale_t: u64,
    weight_scale_2: f32,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
                act as *const c_void,
                weight_packed_t as *const c_void,
                weight_scale_t as *const c_void,
                weight_scale_2,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (
            act,
            weight_packed_t,
            weight_scale_t,
            weight_scale_2,
            out,
            m,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
