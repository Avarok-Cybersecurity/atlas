// SPDX-License-Identifier: AGPL-3.0-only
//! Grouped (per-expert) CUTLASS NVFP4 MoE GEMM host wrappers.

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// Grouped (per-expert) NVFP4 fused gate_up GEMM — Holo MoE Phase-1
/// escape-hatch path. Dispatches the proven Sm120 NVFP4 collective once per
/// active expert over its token slice; bit-faithful to
/// `nvfp4_gemm_bf16_act_weight_t` (it IS that collective), at one launch per
/// expert. Used to validate that the FP4 math integrates correctly in grouped
/// form before the hand-rolled block-scaled mma (Phase 2).
///
/// `a` is bf16 `[M_total, K]`; expert `e` owns rows
/// `[expert_offsets[e], expert_offsets[e+1])`. `*_packed_ptrs`/`*_scale_ptrs`
/// are device-pointer arrays (one per expert) in the
/// `pack_bf16_weight_to_nvfp4_t` layout (`[N,K/2]` + `[K/16,N]`); the
/// `*_scale2_vals` and `expert_offsets` slices are HOST arrays.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up(
    a: u64,
    gate_packed_ptrs: &[u64],
    gate_scale_ptrs: &[u64],
    gate_scale2_vals: &[f32],
    up_packed_ptrs: &[u64],
    up_scale_ptrs: &[u64],
    up_scale2_vals: &[f32],
    c_gate: u64,
    c_up: u64,
    expert_offsets: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let num_experts = gate_packed_ptrs.len();
        if expert_offsets.len() != num_experts + 1 {
            bail!(
                "nvfp4_grouped_gate_up: expert_offsets len {} != num_experts+1 {}",
                expert_offsets.len(),
                num_experts + 1
            );
        }
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_nvfp4_grouped_gate_up(
                a as *const c_void,
                gate_packed_ptrs.as_ptr(),
                gate_scale_ptrs.as_ptr(),
                gate_scale2_vals.as_ptr(),
                up_packed_ptrs.as_ptr(),
                up_scale_ptrs.as_ptr(),
                up_scale2_vals.as_ptr(),
                c_gate as *mut c_void,
                c_up as *mut c_void,
                expert_offsets.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped gate_up failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (
            a,
            gate_packed_ptrs,
            gate_scale_ptrs,
            gate_scale2_vals,
            up_packed_ptrs,
            up_scale_ptrs,
            up_scale2_vals,
            c_gate,
            c_up,
            expert_offsets,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Single-launch grouped (`GemmUniversalMode::kGrouped`) NVFP4 fused gate_up
/// GEMM — the Phase-2 successor to [`nvfp4_grouped_gate_up`]. Replaces the
/// per-expert collective loop with ONE grouped launch over all active experts,
/// eliminating the N-launch overhead.
///
/// `a` is bf16 `[M_total, K]`, expert-contiguous (caller permuted so expert `e`
/// owns rows `[expert_offsets_host[e], expert_offsets_host[e+1])`).
/// `*_packed_ptrs` are device-pointer arrays (one per expert) into the CUTLASS
/// `[N,K/2]` packed weight tables; `*_sfb_ptrs` are device-pointer arrays into
/// the swizzled SFB (ue4m3) scale tables (see `pack_weight_sfb`).
/// `*_scale2_vals` and `expert_offsets_host` are HOST arrays.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_fused(
    a: u64,
    sorted_token_ids: u64,
    gate_packed_ptrs: &[u64],
    gate_sfb_ptrs: &[u64],
    gate_scale2_vals: &[f32],
    up_packed_ptrs: &[u64],
    up_sfb_ptrs: &[u64],
    up_scale2_vals: &[f32],
    c_gate: u64,
    c_up: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let num_experts = gate_packed_ptrs.len();
        if expert_offsets_host.len() != num_experts + 1 {
            bail!(
                "nvfp4_grouped_gate_up_fused: expert_offsets len {} != num_experts+1 {}",
                expert_offsets_host.len(),
                num_experts + 1
            );
        }
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_nvfp4_grouped_gate_up_fused(
                a as *const c_void,
                sorted_token_ids as *const i32,
                gate_packed_ptrs.as_ptr(),
                gate_sfb_ptrs.as_ptr(),
                gate_scale2_vals.as_ptr(),
                up_packed_ptrs.as_ptr(),
                up_sfb_ptrs.as_ptr(),
                up_scale2_vals.as_ptr(),
                c_gate as *mut c_void,
                c_up as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped(fused) gate_up failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (
            a,
            sorted_token_ids,
            gate_packed_ptrs,
            gate_sfb_ptrs,
            gate_scale2_vals,
            up_packed_ptrs,
            up_sfb_ptrs,
            up_scale2_vals,
            c_gate,
            c_up,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Single-launch grouped NVFP4 DOWN projection (`atlas_cutlass_nvfp4_grouped_down`).
/// `a` is the post-SiLU bf16 intermediate `[M_total, K=inter]`, ALREADY
/// expert-contiguous (no gather). `packed_ptrs`/`sfb_ptrs` are device-pointer
/// arrays into the `[N=hidden,K/2]` packed + swizzled-SFB down tables; `scale2_vals`
/// and `expert_offsets_host` are HOST arrays. Writes `c` `[M_total, N=hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_down(
    a: u64,
    packed_ptrs: &[u64],
    sfb_ptrs: &[u64],
    scale2_vals: &[f32],
    c: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let num_experts = packed_ptrs.len();
        if expert_offsets_host.len() != num_experts + 1 {
            bail!(
                "nvfp4_grouped_down: expert_offsets len {} != num_experts+1 {}",
                expert_offsets_host.len(),
                num_experts + 1
            );
        }
        let ctx = ctx()?;
        let status = unsafe {
            atlas_cutlass_nvfp4_grouped_down(
                a as *const c_void,
                packed_ptrs.as_ptr(),
                sfb_ptrs.as_ptr(),
                scale2_vals.as_ptr(),
                c as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped down failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (
            a,
            packed_ptrs,
            sfb_ptrs,
            scale2_vals,
            c,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
