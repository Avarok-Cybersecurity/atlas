// SPDX-License-Identifier: AGPL-3.0-only

//! The Hopper GDN chunked-prefill remnant twins: shared-memory SSOT and the
//! grammar of the lever that selects them (#928).
//!
//! The kernels are `kernels/hopper/common/gdn_fwd_o_hopper.cu` and
//! `..._recompute_wu_hopper.cu`, which exist only under `kernels/hopper`, so
//! their [`spark_runtime::gpu::KernelHandle`]s resolve on that target and are
//! `KernelHandle(0)` everywhere else. That — not an env var — is what keeps
//! gb10, b200 and strix on their parents.
//!
//! WHY THEY EXIST, in one receipt (full derivation in
//! `GDN-PREFILL-ATTRIBUTION.md`): on 1xH100 / Qwen3.8-27B-FP8, nsys round 9
//! (2026-09-11) put `gated_delta_rule_chunk_fwd_o` at 20.1 ms of a 368.3 ms
//! 1193-token prefill (5.5%; 71.9 ms at 4593 tokens) and
//! `gated_delta_rule_recompute_wu` at 14.9 ms (4.0%; 47.5 ms). Both already
//! run their BIG matmuls on `mma.sync` — 16-17 and 12-14.5 TFLOP/s, 4.4x the
//! scalar state spine — so what is left in them is the scalar remainder:
//! `fwd_o`'s triangular `tril(kq).uc`, 0.53 of its 3.68 MFLOP on 128 of its
//! 512 threads, and `wu`'s two forward substitutions, 79-85% of that kernel
//! per its own 2026-08-22 solve-removed probe.
//!
//! ONE LEVER FOR THE FAMILY. `ATLAS_GDN_PREFILL_TC` already selects the
//! tensor-core state spine; it now selects these two as well wherever the
//! image carries them, because the three kernels are one pipeline and an
//! operator who wants the TC prefill wants all of it. The A/B that needs them
//! apart gets `ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1`, which keeps the spine and
//! leaves `wu`/`fwd_o` on their parents — one variable, in the direction that
//! is safe to be wrong about.

use spark_runtime::gpu::KernelHandle;

/// Compile-time tile of both twins: `K_DIM == V_DIM` in
/// `kernels/hopper/common/gdn_prefill_hopper.cuh`.
pub(crate) const GDN_HOPPER_DIM: u32 = 128;
/// That header's `CHUNK`.
pub(crate) const GDN_HOPPER_CHUNK: u32 = 64;

/// SSOT mirror of `FOH_SMEM` in `gdn_fwd_o_hopper.cu`:
///
/// ```text
/// sq[64][136] + sk[64][136] + Sb[128][136] + ucT[128][72] + kqh[64][72]
///   + gc[64] f32 = 17408 + 17408 + 34816 + 18432 + 9216 + 256 = 97 536 B
/// ```
///
/// SMALLER than the parent's 98 816 B, because the `kq` lo limb aliases `sk`.
/// The padded 136/72 row strides are what make the MMA fragment reads
/// bank-conflict-free; under-sizing this reads a tile out of bounds, so the
/// launcher and the kernel must not be able to disagree about it. The kernel
/// carries the matching `static_assert`.
pub(crate) const GDN_FWD_O_HOPPER_SMEM: u32 = 2 * (GDN_HOPPER_CHUNK * 136 * 2)
    + GDN_HOPPER_DIM * 136 * 2
    + GDN_HOPPER_DIM * 72 * 2
    + GDN_HOPPER_CHUNK * 72 * 2
    + GDN_HOPPER_CHUNK * 4;

/// SSOT mirror of `WUH_SMEM` in `gdn_recompute_wu_hopper.cu`:
///
/// ```text
/// sk[64][136] + Ld/Tf[64][24] f32 + Lh/Ll[64][72] + Th/Tl[64][24]
///   + Xh/Xl[16][16][24] + gc[64] f32
///   = 17408 + 2*6144 + 2*9216 + 2*3072 + 2*12288 + 256 = 79 104 B
/// ```
pub(crate) const GDN_WU_HOPPER_SMEM: u32 = GDN_HOPPER_CHUNK * 136 * 2
    + 2 * (GDN_HOPPER_CHUNK * 24 * 4)
    + 2 * (GDN_HOPPER_CHUNK * 72 * 2)
    + 2 * (GDN_HOPPER_CHUNK * 24 * 2)
    + 2 * (16 * 16 * 24 * 2)
    + GDN_HOPPER_CHUNK * 4;

/// Both twins launch 512 threads — 16 warps, every one of which computes.
pub(crate) const GDN_HOPPER_REMNANT_BLOCK: u32 = 512;

/// Why a Hopper remnant twin is NOT running — `None` means it is.
///
/// Pure so the grammar is testable without a GPU or the process environment,
/// and it NAMES THE GUARD THAT REJECTED: a perf path that asks to be enabled
/// and silently is not measures as "no effect" (PR #296 shipped exactly that,
/// an ldmatrix GEMM that fell back with no error while both gates stayed
/// green).
///
/// The tile guards are not defensive padding. Both kernels' fragment maps,
/// padded smem strides and warp splits are compile-time 128/128/64; a narrower
/// head or a different chunk would not run slowly, it would read the wrong
/// columns.
pub(crate) fn gdn_hopper_remnant_reject(
    requested: bool,
    killed: bool,
    kernel_present: bool,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if killed {
        Some("ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1 pins wu/fwd_o to their parents")
    } else if !kernel_present {
        Some("kernel absent from this image (kernels/hopper only)")
    } else if k_dim != GDN_HOPPER_DIM || v_dim != GDN_HOPPER_DIM || chunk != GDN_HOPPER_CHUNK {
        Some("head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)")
    } else {
        None
    }
}

/// What the launcher needs to know about one remnant: which handle to launch,
/// how many threads, and how much shared memory.
///
/// Returned as a pair rather than branched at each launch site so the two
/// kernels cannot drift into different selection rules — the failure mode the
/// spine's own `tc_ok` block avoids by computing the verdict once.
pub(crate) struct RemnantPick {
    pub kernel: KernelHandle,
    pub block: u32,
    pub smem: u32,
    /// `None` when the twin runs; the named guard when the parent does.
    pub reject: Option<&'static str>,
}

/// Choose between a parent kernel and its Hopper twin.
///
/// `parent_smem`/`parent_block` are used verbatim when the twin is rejected,
/// so a rejection is exactly the pre-#928 launch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_hopper_remnant_pick(
    requested: bool,
    killed: bool,
    parent: KernelHandle,
    parent_block: u32,
    parent_smem: u32,
    twin: KernelHandle,
    twin_smem: u32,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> RemnantPick {
    let reject = gdn_hopper_remnant_reject(requested, killed, twin.0 != 0, k_dim, v_dim, chunk);
    match reject {
        None => RemnantPick {
            kernel: twin,
            block: GDN_HOPPER_REMNANT_BLOCK,
            smem: twin_smem,
            reject,
        },
        Some(_) => RemnantPick {
            kernel: parent,
            block: parent_block,
            smem: parent_smem,
            reject,
        },
    }
}

/// Say WHICH kernel ran and, when the lever asked for a twin it did not get,
/// WHICH guard refused. Lives here so a future third remnant cannot acquire a
/// differently-worded log line: a perf path that asks to be enabled and
/// silently is not measures as "no effect" (PR #296).
pub(crate) fn gdn_hopper_remnant_log(name: &str, pick: &RemnantPick, requested: bool) {
    match pick.reject {
        Some(why) if requested => {
            tracing::warn!("GDN {name}: the Hopper twin is NOT running: {why}");
        }
        None => tracing::info!(
            "GDN {name}: gated_delta_rule_{name}_hopper (ATLAS_GDN_PREFILL_TC) \
             block={} smem={}B",
            pick.block,
            pick.smem
        ),
        _ => {}
    }
}

/// Resolve BOTH remnants and log the verdicts, in one call.
///
/// Returns `ATLAS_GDN_PREFILL_TC`'s presence alongside the two picks because
/// the same lever also selects the tensor-core state spine and must be read
/// exactly once per prefill: two reads is two chances to disagree about what
/// the operator asked for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_hopper_remnants(
    k_wu: KernelHandle,
    smem_wu: u32,
    k_fo: KernelHandle,
    smem_fo: u32,
    k_wu_hopper: KernelHandle,
    k_fo_hopper: KernelHandle,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> (bool, RemnantPick, RemnantPick) {
    let requested = std::env::var("ATLAS_GDN_PREFILL_TC").is_ok();
    let killed = gdn_hopper_remnants_killed();
    let pick = |parent, pblock, psmem, twin, tsmem| {
        gdn_hopper_remnant_pick(
            requested, killed, parent, pblock, psmem, twin, tsmem, k_dim, v_dim, chunk,
        )
    };
    let wu = pick(k_wu, 256, smem_wu, k_wu_hopper, GDN_WU_HOPPER_SMEM);
    let fo = pick(k_fo, 512, smem_fo, k_fo_hopper, GDN_FWD_O_HOPPER_SMEM);
    // ONCE PER PROCESS, not once per layer. This runs inside the per-layer
    // prefill call, so on a 48-layer model an unconditional line is 96 of them
    // per request — and on every non-Hopper target with the lever set, the
    // refusal is the NORMAL state, so it would be 96 warnings per request for
    // a correct build. The verdict cannot change between calls: it is a
    // function of the environment and of compile-time tile constants.
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        gdn_hopper_remnant_log("recompute_wu", &wu, requested);
        gdn_hopper_remnant_log("chunk_fwd_o", &fo, requested);
    });
    (requested, wu, fo)
}

/// `ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1` — the one-variable A/B switch.
///
/// Read here rather than at the launch site so the two remnants cannot be
/// enabled by different spellings, and so the tests can pin the spelling
/// without touching the process environment.
pub(crate) fn gdn_hopper_remnants_killed() -> bool {
    std::env::var("ATLAS_NO_GDN_PREFILL_TC_REMNANTS")
        .ok()
        .as_deref()
        == Some("1")
}

#[cfg(test)]
#[path = "ssm_gdn_remnants_tests.rs"]
mod ssm_gdn_remnants_tests;

#[cfg(test)]
#[path = "ssm_gdn_remnants_numerics_tests.rs"]
mod ssm_gdn_remnants_numerics_tests;
