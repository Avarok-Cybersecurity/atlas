// SPDX-License-Identifier: AGPL-3.0-only
//! Optional CUTLASS host-wrapper FFI for de-risking GB10 GEMM replacements.
//!
//! Split for the ≤500 LoC cap: this root holds the shared FFI `extern` block,
//! the workspace `Ctx`, and module wiring; the public wrappers live in the
//! `gemm` (dense BF16 + NVFP4), `grouped` (per-expert MoE), and `pack`
//! (weight pack / SFB swizzle / transpose) siblings. The public API
//! (`spark_runtime::cutlass::<fn>`) is preserved via the re-exports below.

#[cfg(atlas_cutlass)]
use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;
#[cfg(atlas_cutlass)]
use std::sync::OnceLock;

mod gemm;
mod grouped;
mod pack;

pub use gemm::{bf16_gemm_act_weight_t, nvfp4_gemm_bf16_act_weight_t};
pub use grouped::{nvfp4_grouped_down, nvfp4_grouped_gate_up, nvfp4_grouped_gate_up_fused};
pub use pack::{pack_bf16_weight_to_nvfp4_t, pack_weight_sfb, transpose_nvfp4_packed_kton};

#[cfg(all(test, atlas_cutlass))]
mod tests;

#[cfg(atlas_cutlass)]
unsafe extern "C" {
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
        act: *const c_void,
        weight_packed_t: *const c_void,
        weight_scale_t: *const c_void,
        weight_scale_2: f32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_pack_bf16_weight_to_nvfp4_t(
        weight_bf16: *const c_void,
        packed_t: *mut c_void,
        scale_t: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_gate_up(
        a_bf16: *const c_void,
        gate_packed_ptrs: *const u64,
        gate_scale_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_scale_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_gate_up_fused(
        a_bf16: *const c_void,
        sorted_token_ids: *const i32,
        gate_packed_ptrs: *const u64,
        gate_sfb_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_sfb_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_down(
        a_bf16: *const c_void,
        packed_ptrs: *const u64,
        sfb_ptrs: *const u64,
        scale2_vals: *const f32,
        c_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_pack_weight_sfb(
        scale_in: *const c_void,
        scale_out: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_transpose_nvfp4_packed_kton(
        src_packed_t: *const c_void,
        dst_packed: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_128x256(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_256x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_64x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_128x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_64x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cublaslt_bf16_gemm_act_weight_t_algo(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
        algo_index: i32,
        returned_count: *mut i32,
    ) -> i32;
    pub(crate) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
}

#[cfg(atlas_cutlass)]
pub(crate) struct Ctx {
    pub(crate) workspace: u64,
    pub(crate) ws_size: usize,
}

#[cfg(atlas_cutlass)]
unsafe impl Send for Ctx {}
#[cfg(atlas_cutlass)]
unsafe impl Sync for Ctx {}

#[cfg(atlas_cutlass)]
static CTX: OnceLock<Ctx> = OnceLock::new();

#[cfg(atlas_cutlass)]
pub(crate) fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    // Shared scratch for all CUTLASS host wrappers. The grouped NVFP4 MoE path
    // (single-launch kGrouped over up to 256 experts) stages packed-A + SFA +
    // per-group arrays + the gemm workspace here; at large prefill M the 256-group
    // gemm workspace alone exceeds the old 64 MB (-> status -2 + an OOB context
    // corruption). 512 MB by default; override with ATLAS_CUTLASS_WORKSPACE_MB.
    let ws_size = std::env::var("ATLAS_CUTLASS_WORKSPACE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        * 1024
        * 1024;
    let mut workspace = 0u64;
    let status = unsafe { cuMemAlloc_v2(&mut workspace, ws_size) };
    if status != 0 {
        bail!("cuMemAlloc CUTLASS workspace failed: {status}");
    }
    let _ = CTX.set(Ctx { workspace, ws_size });
    Ok(CTX.get().unwrap())
}
