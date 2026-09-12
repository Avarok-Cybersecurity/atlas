// SPDX-License-Identifier: AGPL-3.0-only

//! The Hopper GDN decode twin for the BATCHED-STRIDED arm: its launch
//! geometry, the width guard that selects it, and the one route line that
//! names it (#927).
//!
//! The kernel is `kernels/hopper/common/gdn_decode_strided_hopper.cu`, an
//! ADDITION declared in `kernels/hopper/HARDWARE.toml`'s `[kernels] overrides`
//! and present under no other target — not even `kernels/b200`, which links
//! Hopper's `common/` only for the sources IT declares. So its [`KernelHandle`]
//! is `KernelHandle(0)` everywhere else and the dispatch stays on the gb10
//! parent there without reading anything.
//!
//! ⚠️ **NOT the same kernel as `ssm_gdn_hopper`'s twins.** That module's
//! `gated_delta_rule_decode_f32_strided_hopper` re-partitions state COLUMNS to
//! fill a 132-SM device at n=1 and is DEFAULT OFF — H100 round 12 measured it
//! 0.83x at contiguous n=1 and a null (+0.19%) at n=16. This one re-partitions
//! nothing and attacks the other axis: the parent reads the f32 state TWICE
//! per step (once for `hk_dot`, once to apply the update), and this twin reads
//! 96 of its 128 rows once.
//!
//! WHY IT EXISTS, in one receipt (full derivation in
//! `GDN-DECODE-ATTRIBUTION.md`, "Round 17"). nsys `--cuda-graph-trace=node`,
//! 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8, round 13 cell V, median n=16 decode
//! step 19.887 ms busy: `gated_delta_rule_decode_f32_strided*` is 48 graph
//! nodes, **2 748.9 us = 13.82% of the step**, 57.27 us/launch, moving a
//! 50.33 MB live state. Compulsory traffic (one read, one write) is 100.66 MB
//! = 1 758 GB/s = 52.5% of HBM; ISSUED traffic is 150.99 MB = 2 637 GB/s =
//! 78.7%. The gap between those two numbers IS the lever, and this twin closes
//! three quarters of it.
//!
//! BIT-IDENTICAL, and that is the contract rather than an aspiration: the
//! twin keeps one thread per state COLUMN walking every `j` itself, because
//! f32 addition is not associative and any split of the `kd` reduction across
//! threads would re-bracket it. Only the STORAGE of the state between the two
//! passes changes. `native_gdn_decode_hopper_microtest` asserts
//! `state_diff == 0` / `out_diff == 0`, not a tolerance, and carries a
//! KNOWN_BAD control.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// The module `qwen3_ssm::init_kernels` probes and this file's route line
/// names. One string, two consumers — the arrangement
/// `ssm_gdn_tc_route::GDN_TC_SPINE_ENTRY` exists to force, after a serve log
/// named a kernel family while the binary launched a different member of it
/// (H100 round 12, stage 4b).
pub const GDN_STRIDED_SMEM_MODULE: &str = "gdn_decode_strided_hopper";
/// The entry point inside it.
pub const GDN_STRIDED_SMEM_ENTRY: &str = "gated_delta_rule_decode_f32_strided_hopper_smem";

/// `k_dim`, `v_dim` and `blockDim.x` the kernel is statically sized for.
///
/// Not a tunable: `smem_h` is `[72][128]` and the state-norm clamp reduces
/// across exactly the four warps the parent's `norm_sums[4]` and `tid / 32`
/// assume. The kernel re-checks this itself and returns rather than scribble
/// past its staging buffer; this constant is why it never has to.
pub const GDN_STRIDED_SMEM_DIM: u32 = 128;

/// Resident CTAs per SM the shipped kernel is compiled for.
///
/// ptxas, sm_90a, CUDA 13.0, `--fmad=false`, `--Werror all-warnings`:
/// **80 registers, 0 bytes of spill, 37 904 B of shared memory**, so on a
/// GH100 SM (65 536 registers, 233 472 B of shared memory) residency is
/// `min(65536 / (80 * 128), 233472 / 37904)` = `min(6, 6)` = 6, i.e. 24
/// warps/SM. `__launch_bounds__(128, 6)` in the kernel is the same number,
/// stated as a contract so a toolkit that cannot hold the split fails as a
/// visible spill instead of silently halving residency.
pub const GDN_STRIDED_SMEM_CTAS_PER_SM: u32 = 6;

/// Threads per CTA — the parent's block, and a contract: the reduction order
/// is a function of it.
pub const GDN_STRIDED_SMEM_BLOCK: u32 = GDN_STRIDED_SMEM_DIM;

/// Rows of one `[k_dim][v_dim]` state tile the kernel STAGES in shared memory.
///
/// The three constants below are the kernel's `GDN_STR_SMEM_ROWS`,
/// `GDN_STR_REG_ROWS` and the remainder, mirrored so the host can state the
/// launch's shared-memory footprint and so [`gdn_strided_smem_row_home`] can
/// be graded. They are a DESCRIPTION of the kernel, not an input to it: the
/// `.cu` spells its own literals and nothing passes these across the ABI.
pub const GDN_STRIDED_SMEM_STAGED_ROWS: u32 = 72;
/// Rows retained in registers across the two passes.
pub const GDN_STRIDED_SMEM_REG_ROWS: u32 = 24;
/// Rows re-read from global on pass 2, exactly as the parent does for all 128.
pub const GDN_STRIDED_SMEM_REREAD_ROWS: u32 =
    GDN_STRIDED_SMEM_DIM - GDN_STRIDED_SMEM_STAGED_ROWS - GDN_STRIDED_SMEM_REG_ROWS;
/// Bytes of static shared memory one CTA takes: the staging tile plus the `k`
/// and `q` rows plus the clamp's four-warp scratch. ptxas reports exactly this.
pub const GDN_STRIDED_SMEM_BYTES: u32 =
    GDN_STRIDED_SMEM_STAGED_ROWS * GDN_STRIDED_SMEM_DIM * 4 + 2 * GDN_STRIDED_SMEM_DIM * 4 + 16;

/// Where row `j` of a state tile lives between the kernel's two passes.
///
/// The tile map, as data. Its only job is to be checkable: the bit-identity
/// argument rests on the three segments covering `[0, 128)` exactly once each,
/// in ascending `j`, with the accumulator carried across the segment
/// boundaries — a gap would drop a term and an overlap would double one, and
/// neither is visible by reading three loops in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnStridedRowHome {
    /// Staged in shared memory on pass 1, read from SRAM on pass 2.
    Smem,
    /// Held in registers across both passes.
    Register,
    /// Re-read from global on pass 2 — the parent's own behaviour.
    Reread,
}

/// [`GdnStridedRowHome`] for one row, `None` outside the tile.
pub fn gdn_strided_smem_row_home(j: u32) -> Option<GdnStridedRowHome> {
    if j >= GDN_STRIDED_SMEM_DIM {
        None
    } else if j < GDN_STRIDED_SMEM_STAGED_ROWS {
        Some(GdnStridedRowHome::Smem)
    } else if j < GDN_STRIDED_SMEM_STAGED_ROWS + GDN_STRIDED_SMEM_REG_ROWS {
        Some(GdnStridedRowHome::Register)
    } else {
        Some(GdnStridedRowHome::Reread)
    }
}

/// The state element thread `tid` of CTA `(vh, b)` owns at row `j`, as an
/// offset into `h_state` in ELEMENTS — the parent's addressing, reproduced so
/// a test can assert the twin touches the same bytes.
///
/// One thread per state COLUMN is the bit-identity constraint itself: the
/// `kd` reduction is a serial f32 chain and only its owner may walk it.
pub fn gdn_strided_smem_elem(vh: u32, b: u32, num_v_heads: u32, j: u32, tid: u32) -> u64 {
    let tile = u64::from(b * num_v_heads + vh)
        * u64::from(GDN_STRIDED_SMEM_DIM)
        * u64::from(GDN_STRIDED_SMEM_DIM);
    tile + u64::from(j) * u64::from(GDN_STRIDED_SMEM_DIM) + u64::from(tid)
}

/// Fewest decode rows the twin will take. **The width guard.**
///
/// One CTA per (sequence, head) means the grid is `num_v_heads * batch_size`,
/// so the row count IS the occupancy, and this kernel buys its traffic saving
/// with residency: six resident CTAs per SM against the parent's twelve (40
/// registers, 1 040 B of smem). Where the grid already fails to fill the
/// device that difference is free — both kernels leave SMs idle — but the
/// twin's per-CTA shared-memory footprint means it cannot make up for a thin
/// grid the way the parent can, and its staging phase adds a barrier the
/// parent does not have. So below the threshold the PARENT runs.
///
/// 4, not 2, because the two conditions in [`gdn_decode_strided_smem_accept`]
/// have to hold at the SAME shape: at this model's `nv = 48` a batch of 4 is
/// 192 CTAs against 132 SMs — the first width at which every SM gets a CTA
/// *and* the second wave is small. At n=1 the grid is 48 CTAs on 132 SMs and
/// the thing to fix is underfill, not traffic — which is
/// `gdn_decode_hopper.cu`'s job, and H100 round 12 says it does not manage it
/// either. The n=16 step this twin was written for is 768 CTAs.
pub const GDN_STRIDED_SMEM_MIN_ROWS: u32 = 4;

/// Does this shape take the twin? — the width guard, as a pure function.
///
/// Two conditions, both necessary:
///  1. `batch_size >= `[`GDN_STRIDED_SMEM_MIN_ROWS`] — the decode arm this
///     kernel was measured for is the batched one;
///  2. the grid fills the device: `num_v_heads * batch_size >= sm_count`, so
///     no SM is idle while a CTA elsewhere pays for the staging barrier.
///
/// Pure, and `sm_count` is an argument rather than a read of
/// `atlas_kernels::TARGET_SM_COUNT`, so the rule is gradeable for any part
/// from a CPU test.
pub fn gdn_decode_strided_smem_accept(batch_size: u32, num_v_heads: u32, sm_count: u32) -> bool {
    batch_size >= GDN_STRIDED_SMEM_MIN_ROWS
        && num_v_heads.saturating_mul(batch_size) >= sm_count.max(1)
}

/// The kernel's own dimension contract, mirrored on the host.
pub fn gdn_strided_smem_dims_ok(k_dim: u32, v_dim: u32) -> bool {
    k_dim == GDN_STRIDED_SMEM_DIM && v_dim == GDN_STRIDED_SMEM_DIM
}

/// Is the one-read strided twin selected? — `[defaults]
/// gdn_decode_strided_hopper`, with `ATLAS_GDN_DECODE_STRIDED_HOPPER`
/// overriding it and `ATLAS_NO_GDN_HOPPER=1` outranking both
/// ([`super::target_defaults`]).
///
/// The kill switch is SHARED with `gdn_decode_hopper` on purpose: an operator
/// who sets `ATLAS_NO_GDN_HOPPER=1` to take a Hopper GDN decode kernel out of
/// a serve means every Hopper GDN decode kernel, and a second spelling is how
/// one of them comes to survive a bisect that thought it had disarmed them all.
pub fn gdn_decode_strided_smem_enabled() -> bool {
    super::target_defaults::resolved()
        .gdn_decode_strided_hopper
        .value
}

/// The lever, a resolved handle, the dimension contract and the width guard —
/// the whole decision, in one place, so no dispatch site spells any part of it.
pub fn gdn_decode_strided_smem_selected(
    twin: KernelHandle,
    batch_size: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    sm_count: u32,
) -> bool {
    twin.0 != 0
        && gdn_decode_strided_smem_enabled()
        && gdn_strided_smem_dims_ok(k_dim, v_dim)
        && gdn_decode_strided_smem_accept(batch_size, num_v_heads, sm_count)
}

/// `GDN state decode: …` — the line the batched decode prints ONCE PER PROCESS
/// for the arm it routed to, built from the constants above so it can only
/// ever name the kernel the probe bound.
///
/// Pure and returning a `String` rather than logging: a route line nothing can
/// grade is how a log comes to describe a kernel the binary does not run
/// (`ssm_gdn_tc_route`'s own lesson, twice).
pub fn gdn_decode_strided_smem_route_line(
    twin: bool,
    num_v_heads: u32,
    batch_size: u32,
    sm_count: u32,
) -> String {
    let ctas = num_v_heads.saturating_mul(batch_size);
    if twin {
        format!(
            "GDN state decode: {GDN_STRIDED_SMEM_ENTRY} ([defaults] \
             gdn_decode_strided_hopper; f32 state read once for 96 of 128 rows, \
             72 staged in smem + 24 retained in registers) \
             grid=[{num_v_heads},{batch_size}] block={GDN_STRIDED_SMEM_BLOCK} \
             smem={GDN_STRIDED_SMEM_BYTES}B ctas={ctas} sm_count={sm_count} \
             ctas_per_sm<={GDN_STRIDED_SMEM_CTAS_PER_SM}"
        )
    } else {
        format!(
            "GDN state decode: gated_delta_rule_decode_f32_strided (gb10 parent; \
             the Hopper one-read twin declines below \
             {GDN_STRIDED_SMEM_MIN_ROWS} rows or a grid under sm_count) \
             grid=[{num_v_heads},{batch_size}] ctas={ctas} sm_count={sm_count}"
        )
    }
}

/// Launch the one-read strided twin. Same arguments as
/// [`super::gdn_decode_f32_strided`], same bits out, same grid and block.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_hopper_smem(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        gdn_strided_smem_dims_ok(k_dim, v_dim),
        "{GDN_STRIDED_SMEM_ENTRY} needs k_dim == v_dim == {GDN_STRIDED_SMEM_DIM} \
         (its staging buffer is statically sized and its state-norm clamp reduces \
         across the whole head); got k_dim={k_dim} v_dim={v_dim}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([GDN_STRIDED_SMEM_BLOCK, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(out_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "ssm_gdn_strided_hopper_tests.rs"]
mod tests;
