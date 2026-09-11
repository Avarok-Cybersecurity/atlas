// SPDX-License-Identifier: AGPL-3.0-only

//! The Hopper BA-gates twin: its launch geometry, and the grammar of the lever
//! and the guards that select it (#928).
//!
//! The kernel is `kernels/hopper/common/ssm_ba_gates_hopper.cu`, which exists
//! only under `kernels/hopper`, so its [`KernelHandle`] is `KernelHandle(0)`
//! on gb10, b200 and strix and the launcher stays on the gb10 parent there
//! without reading anything. That — not an env var — is what keeps the other
//! targets on `ssm_preprocess::dense_gemm_ba_gates_prefill`.
//!
//! WHY IT EXISTS, in one receipt (full derivation in
//! `SSM-BA-GATES-ATTRIBUTION.md`). The parent launches
//! `Grid: (ceil(N/4), M_tokens, 1) Block: (256,1,1)`: one CTA computes four of
//! the `N = ssm_ba_size = 2*nv` BA outputs for one token, and each of those
//! four walks the token's entire `K` activation row — the CTA's 256 threads are
//! four 64-lane groups, one per output. At the Qwen3.8-27B geometry `N=96`,
//! `K=5120` that is 24 CTAs per token and **96 reads of every activation row**,
//! one per BA output. nsys, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8 @
//! `815c9f160`, round 13 cell T1: the kernel is 26 881.8 us = 5.85% of the
//! 4593-token prefill and 6 896.3 us = 3.13% of the 1168-token forward, 555.04
//! us per launch at `M=4576`, observed grid `(24, 4576)`, 88 GB/s of compulsory
//! traffic — 2.6% of HBM, so this was never a bandwidth problem. The twin reads
//! the row 12 times and issues ~1.8x fewer instructions (SASS, sm_90a).
//!
//! BIT-IDENTICAL, and that is the contract, not an aspiration: the twin keeps
//! the parent's lane-strided `kv` sweep, its 5-step shuffle butterfly and its
//! `warp_even + warp_odd` cross-warp order, and only hoists A's (exact) bf16 ->
//! f32 widening out of the output loop. `native_ssm_ba_gates_hopper_microtest`
//! asserts byte equality of `gate` and `beta`, not a tolerance.
//!
//! ⚠️ **THE TOKEN-COUNT GUARD IS LOAD-BEARING.** One CTA per token means the
//! grid IS the token count, and this kernel is on the BATCHED DECODE path too
//! (`trait_decode_batched.rs`; nsys round 13 SS C.2 prices it at 227.5 us/step
//! over 48 launches at n=16). A 16-row step would put 16 CTAs on 132 SMs where
//! the parent puts 384. [`ssm_ba_gates_hopper_reject`] declines below
//! `MIN_CTAS_PER_SM * sm_count` tokens and the parent runs, so promoting this
//! lever cannot cost the decode step.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_map::DenseWeight;

/// Threads per CTA — the parent's block, and a contract: the reduction order
/// is a function of this, [`BA_GATES_LANES`] and [`BA_GATES_OUTS`].
pub const BA_GATES_BLOCK: u32 = 256;
/// Threads that cooperate on ONE BA output. The parent's `threads_per_out`.
pub const BA_GATES_LANES: u32 = 64;
/// BA outputs one CTA-width covers at once: `BA_GATES_BLOCK / BA_GATES_LANES`.
pub const BA_GATES_OUTS: u32 = BA_GATES_BLOCK / BA_GATES_LANES;
/// Warps per CTA, i.e. the width of the twin's cross-warp scratch row.
pub const BA_GATES_WARPS: u32 = BA_GATES_BLOCK / 32;
/// `BAH_GROUPS` in the kernel: output groups a thread accumulates at once, and
/// therefore how many BA outputs one fetch of the activation row serves. The
/// row is read `ceil(ceil(N/4)/8) * 4` times per token — 12 at N=96, against
/// the parent's 96, which reads it once per output. Chosen on ptxas: 64
/// registers and no spill at 8; 12 spills and 16 costs a quarter of the
/// occupancy for 3.3% of the instructions.
pub const BA_GATES_GROUPS: u32 = 8;

/// Tokens per SM the twin insists on before it will take the launch.
///
/// Two, not one: at exactly one CTA per SM a trailing partial wave is the whole
/// kernel, and the parent's `ceil(N/4)`-wide grid beats that comfortably. This
/// is the only number here that is a judgement rather than a contract — it is
/// deliberately conservative, because being wrong in this direction costs the
/// prefill lever nothing (a 1168- or 4576-token chunk clears it by 4-17x) and
/// being wrong in the other direction costs the decode step.
pub const MIN_CTAS_PER_SM: u32 = 2;

/// H100/H200 SXM5 SM count, used only when `GpuBackend::sm_count()` fails.
///
/// A wrong value here moves the guard's threshold, never an answer: both arms
/// compute the same bits.
pub const BA_GATES_FALLBACK_SM_COUNT: u32 = 132;

/// The smallest token count the twin accepts on a device with `sm_count` SMs.
pub fn ba_gates_min_tokens(sm_count: u32) -> u32 {
    MIN_CTAS_PER_SM.saturating_mul(sm_count.max(1))
}

/// Is the twin selected? — `[defaults] ssm_ba_gates_hopper`, with
/// `ATLAS_SSM_BA_GATES_HOPPER` overriding ([`super::target_defaults`]).
///
/// No `ATLAS_NO_*` rung, unlike the GDN families: this lever has no accuracy
/// question behind it, so there is nothing for a kill switch to outrank that
/// `ATLAS_SSM_BA_GATES_HOPPER=0` does not already say.
pub fn ssm_ba_gates_hopper_enabled() -> bool {
    super::target_defaults::resolved().ssm_ba_gates_hopper.value
}

/// Why the twin is NOT running — `None` means it is.
///
/// Pure, so the grammar is testable without a GPU or the process environment,
/// and it NAMES THE GUARD THAT REFUSED. A perf path that asks to be enabled and
/// silently is not measures as "no effect", which is how PR #296 shipped an
/// ldmatrix GEMM that fell back with both gates green.
///
/// The shape guards are not defensive padding:
///   * `K % 8 != 0` would leave a tail the uint4 sweep never reads — the parent
///     has the same limit and the same silence about it, so it is stated here;
///   * `K_stride < K` would read past the row;
///   * `N == 0` has nothing to compute;
///   * the token-count floor is the decode guard described in this module's
///     header.
pub fn ssm_ba_gates_hopper_reject(
    requested: bool,
    kernel_present: bool,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    sm_count: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if !kernel_present {
        Some("kernel absent from this image (kernels/hopper only)")
    } else if n == 0 || k == 0 {
        Some("empty BA projection")
    } else if !k.is_multiple_of(8) {
        Some("K is not a multiple of 8 (the uint4 K sweep would drop a tail)")
    } else if k_stride < k {
        Some("K_stride < K: the activation row is shorter than the reduction")
    } else if m < ba_gates_min_tokens(sm_count) {
        Some("too few tokens to fill the device at one CTA per token")
    } else {
        None
    }
}

/// Which kernel this launch runs, and why the other one did not.
pub struct BaGatesPick {
    pub kernel: KernelHandle,
    /// `true` when `kernel` is the Hopper twin.
    pub twin: bool,
    /// `None` when the twin runs; the named guard when the parent does.
    pub reject: Option<&'static str>,
}

/// Choose between the gb10 parent and the Hopper twin, once, here.
///
/// One entry point rather than an `if` at each of the five dispatch sites: the
/// decision is a lever, a resolved handle and four shape facts, and a call site
/// that spelled it itself would be a second copy of a rule that has already
/// been wrong once elsewhere in this file's family.
#[allow(clippy::too_many_arguments)]
pub fn ba_gates_pick(
    requested: bool,
    parent: KernelHandle,
    twin: KernelHandle,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    sm_count: u32,
) -> BaGatesPick {
    let reject = ssm_ba_gates_hopper_reject(requested, twin.0 != 0, m, n, k, k_stride, sm_count);
    match reject {
        None => BaGatesPick {
            kernel: twin,
            twin: true,
            reject,
        },
        Some(_) => BaGatesPick {
            kernel: parent,
            twin: false,
            reject,
        },
    }
}

/// Say WHICH kernel runs and, when the lever asked for the twin and did not get
/// it, WHICH guard refused — ONCE per process.
///
/// Once, not once per call: this runs inside the per-layer prefill step, so on
/// a 48-GDN-layer model an unconditional line is 48 of them per request. The
/// verdict CAN change between calls here (the token count is an argument), so
/// unlike the GDN remnants' `Once` this logs the first verdict and names the
/// shape it was reached at — a reader who sees "too few tokens" at M=16 has the
/// number that explains it.
pub fn ba_gates_log(pick: &BaGatesPick, requested: bool, m: u32) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| match pick.reject {
        Some(why) if requested => {
            tracing::info!(
                "SSM ba_gates: the Hopper twin is NOT running at M={m}: {why} \
                 (ATLAS_SSM_BA_GATES_HOPPER)"
            );
        }
        None => tracing::info!(
            "SSM ba_gates: dense_gemm_ba_gates_prefill_hopper \
             (ATLAS_SSM_BA_GATES_HOPPER) M={m} block={BA_GATES_BLOCK} grid=(M,1,1)"
        ),
        _ => {}
    });
}

/// The SM count the guard compares against.
pub fn ba_gates_sm_count(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count().unwrap_or(BA_GATES_FALLBACK_SM_COUNT).max(1)
}

/// Launch `dense_gemm_ba_gates_prefill_hopper` — one CTA per token.
///
/// Same ABI as the parent, deliberately: the two differ in `grid` and in
/// nothing else a caller can see, which is what lets `ba_gates_pick` return a
/// handle rather than a closure.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_ba_gates_prefill_hopper(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    ba_weight: &DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    gate_stride: u32,
    nv: u32,
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([BA_GATES_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_stride)
        .arg_u32(gate_stride)
        .arg_u32(nv)
        .arg_u32(vheads_per_group)
        .launch(stream)
}

// ── The index model, as pure functions the host tests grade ────────────────
//
// Both kernels' correctness rests on three mappings agreeing exactly, and none
// of the three is visible in a diff of the two `.cu` files side by side. They
// are spelled here so `ssm_ba_gates_hopper_tests` can grade them on a CPU:
// a mismatch is a wrong ANSWER, not a slow kernel, and the microtest that would
// otherwise be the only witness needs a GPU.

/// The `kv` indices lane `lane` accumulates, in order — the parent's
/// `for (kv = lane; kv < K_VEC; kv += threads_per_out)`, which the twin
/// reproduces verbatim. The ORDER of this sequence is the reduction order.
pub fn ba_gates_lane_kv(lane: u32, k_vec: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut kv = lane;
    while kv < k_vec {
        out.push(kv);
        kv += BA_GATES_LANES;
    }
    out
}

/// The BA output a thread owns.
///
/// PARENT: `n = blockIdx.x * BA_GATES_OUTS + local_out`.
/// TWIN:   `n = (g0 + g) * BA_GATES_OUTS + local_out`, where `g0 + g` walks the
/// same `0..ceil(N/4)` range in tiles of [`BA_GATES_GROUPS`].
/// The two must be the same function of (output group, local_out).
pub fn ba_gates_output(group: u32, local_out: u32) -> u32 {
    group * BA_GATES_OUTS + local_out
}

/// The block-wide warp index that owns lane `lane` of output `local_out`.
///
/// The parent writes its warp partial to `smem[local_out * 2 + (lane / 32)]`;
/// the twin writes `red[g * BA_GATES_WARPS + threadIdx.x / 32]`. Those are the
/// same slot only because `threadIdx.x / 32 == local_out * 2 + lane / 32`,
/// which is what this function and its test pin.
pub fn ba_gates_warp(local_out: u32, lane: u32) -> u32 {
    local_out * 2 + lane / 32
}

/// The two warp partials summed for output `local_out`, in order — the
/// parent's `smem[local_out * 2] + smem[local_out * 2 + 1]`.
pub fn ba_gates_cross_warp_pair(local_out: u32) -> (u32, u32) {
    (local_out * 2, local_out * 2 + 1)
}

/// Where `n` lands in the `[gate(nv), beta(nv)]` output row: `Ok(vh)` for a
/// gate (alpha) element, `Err(vh)` for a beta element — the parent's
/// `within_group < vheads_per_group` split, which the twin copies verbatim.
pub fn ba_gates_slot(n: u32, vheads_per_group: u32) -> Result<u32, u32> {
    let group_dim_ba = 2 * vheads_per_group;
    let within_group = n % group_dim_ba;
    let group = n / group_dim_ba;
    if within_group < vheads_per_group {
        Err(group * vheads_per_group + within_group)
    } else {
        Ok(group * vheads_per_group + (within_group - vheads_per_group))
    }
}

#[cfg(test)]
#[path = "ssm_ba_gates_hopper_tests.rs"]
mod ssm_ba_gates_hopper_tests;
