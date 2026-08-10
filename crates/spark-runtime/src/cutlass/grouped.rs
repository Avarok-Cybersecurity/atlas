// SPDX-License-Identifier: AGPL-3.0-only
//! Grouped (per-expert) CUTLASS NVFP4 MoE GEMM host wrappers.

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// Validate every host array a grouped launch hands to C++ as a bare pointer.
///
/// The C++ side reads `num_experts` elements from EACH per-expert host array and
/// `num_experts + 1` from `expert_offsets` — the lengths never cross the FFI
/// boundary, so a slice that is one entry short becomes a host heap
/// over-read whose result is then dereferenced as a DEVICE pointer. Each of the
/// three wrappers below used to check `expert_offsets` alone and pass the other
/// five or six arrays unchecked; the shape held only because the single caller
/// happened to build them all from one `num_experts`. This function is where
/// that rule lives now, so a wrapper cannot check some of its arrays.
///
/// Deliberately OUTSIDE the `#[cfg(atlas_cutlass)]` arms: the guard has to run
/// (and be testable) on a build without CUTLASS, which is what CI builds.
///
/// `offsets` is additionally checked for the one property that is complete
/// without more parameters — non-negative and non-decreasing, so no group gets a
/// negative row count. The UPPER bound (`offsets[num_experts] <= M_total`)
/// cannot be checked here: `M_total` is not a parameter of any of these
/// wrappers, only the device pointer `a` is. That check belongs to the caller
/// that owns the activation buffer.
fn ensure_group_arrays(
    who: &str,
    num_experts: usize,
    per_expert: &[(&str, usize)],
    offsets: &[i32],
) -> Result<()> {
    if offsets.len() != num_experts + 1 {
        bail!(
            "{who}: expert_offsets len {} != num_experts+1 {}",
            offsets.len(),
            num_experts + 1
        );
    }
    for (name, len) in per_expert {
        if *len != num_experts {
            bail!("{who}: {name} len {len} != num_experts {num_experts}");
        }
    }
    if offsets[0] < 0 {
        bail!("{who}: expert_offsets[0] = {} is negative", offsets[0]);
    }
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            bail!(
                "{who}: expert_offsets is not non-decreasing ({} then {}) — a group would \
                 have a negative row count",
                w[0],
                w[1]
            );
        }
    }
    Ok(())
}

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
    let num_experts = gate_packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_gate_up",
        num_experts,
        &[
            ("gate_scale_ptrs", gate_scale_ptrs.len()),
            ("gate_scale2_vals", gate_scale2_vals.len()),
            ("up_packed_ptrs", up_packed_ptrs.len()),
            ("up_scale_ptrs", up_scale_ptrs.len()),
            ("up_scale2_vals", up_scale2_vals.len()),
        ],
        expert_offsets,
    )?;
    #[cfg(atlas_cutlass)]
    {
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
    let num_experts = gate_packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_gate_up_fused",
        num_experts,
        &[
            ("gate_sfb_ptrs", gate_sfb_ptrs.len()),
            ("gate_scale2_vals", gate_scale2_vals.len()),
            ("up_packed_ptrs", up_packed_ptrs.len()),
            ("up_sfb_ptrs", up_sfb_ptrs.len()),
            ("up_scale2_vals", up_scale2_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(atlas_cutlass)]
    {
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
    let num_experts = packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_down",
        num_experts,
        &[
            ("sfb_ptrs", sfb_ptrs.len()),
            ("scale2_vals", scale2_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(atlas_cutlass)]
    {
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

#[cfg(test)]
mod tests {
    use super::ensure_group_arrays;

    /// The rule these wrappers exist to enforce: EVERY per-expert host array is
    /// read `num_experts` deep by C++, not just the one that used to be checked.
    #[test]
    fn rejects_any_short_per_expert_array() {
        let offsets = [0i32, 4, 9];
        // All lengths correct → accepted.
        ensure_group_arrays("t", 2, &[("a", 2), ("b", 2)], &offsets).unwrap();
        // Each array in turn, one entry short → rejected, and the message names
        // the array so the failure is actionable.
        let e = ensure_group_arrays("t", 2, &[("a", 1), ("b", 2)], &offsets).unwrap_err();
        assert!(e.to_string().contains("a len 1"), "{e}");
        let e = ensure_group_arrays("t", 2, &[("a", 2), ("b", 1)], &offsets).unwrap_err();
        assert!(e.to_string().contains("b len 1"), "{e}");
        // Over-long is rejected too: it means the caller disagrees with
        // `num_experts`, and C++ would silently use only a prefix.
        assert!(ensure_group_arrays("t", 2, &[("a", 3), ("b", 2)], &offsets).is_err());
    }

    #[test]
    fn rejects_bad_expert_offsets() {
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 4]).is_err(),
            "short"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 4, 9, 12]).is_err(),
            "long"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[-1i32, 4, 9]).is_err(),
            "negative base"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 9, 4]).is_err(),
            "decreasing => negative group row count"
        );
        // Empty groups (equal consecutive offsets) are legitimate: an expert
        // that no token routed to.
        ensure_group_arrays("t", 2, &[], &[0i32, 4, 4]).unwrap();
    }

    /// A zero-expert launch is a valid no-op shape, and the check must not
    /// index `offsets[0]` out of bounds when it happens.
    #[test]
    fn zero_experts_is_not_a_panic() {
        ensure_group_arrays("t", 0, &[], &[0i32]).unwrap();
    }
}
