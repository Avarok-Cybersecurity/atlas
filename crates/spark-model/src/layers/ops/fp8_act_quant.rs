// SPDX-License-Identifier: AGPL-3.0-only

//! WHICH per-token FP8 activation quantizer this build launches, and on what
//! grid.
//!
//! Two kernels compute the same bytes. The SHARED one
//! (`kernels/gb10/common/per_token_group_quant_fp8.cu`) spends one 128-thread
//! CTA on one 128-element K-group — one bf16 element per thread, a 256-byte
//! load per CTA — and measures 633/641/636 GB/s on an H100, 18.9-19.1% of HBM,
//! for 10.36% of a 4593-token prefill (#928; round-13 nsys, cell T1N). The
//! HOPPER twin (`kernels/hopper/common/fp8_act_quant_hopper.cu`) gives each
//! group 16 threads of one `uint4` each and packs 8 groups into a CTA, so the
//! CTA's load is 2 KB. Its output is BIT-IDENTICAL by contract — see that file
//! and `native_fp8_act_quant_hopper_microtest`.
//!
//! Selected by PRESENCE **and by WIDTH**. The twin exists only under
//! `kernels/hopper` (`[kernels] overrides`), so on gb10/b200/strix
//! `try_kernel` misses and the shared kernel runs. On Hopper the choice is
//! then `[defaults] fp8_act_quant_hopper` plus a CTA-count floor, because
//! presence alone shipped a 0.76x-0.95x REGRESSION at every decode width
//! (round-16 receipt SS 2.1, Recommendation 2). The rule, the per-K
//! thresholds and the once-per-branch route line live in
//! `fp8_act_quant_floor.rs`; this file owns the PAIR and the two grids.
//!
//! There is still no numeric A/B to arm — the two kernels emit the same bytes
//! — so `ATLAS_FP8_ACT_QUANT_HOPPER=0` is a SPEED kill switch, and the control
//! for the GB/s claim remains a build without the file.
//!
//! [`Fp8ActQuant`] is a PAIR rather than a single resolved handle because the
//! two kernels need DIFFERENT grids, and a bare `KernelHandle` cannot say which
//! it is. Carrying the pair makes the mismatch unrepresentable: the grid is
//! computed from the same value that chose the entry point, in one place
//! ([`fp8_quant_grid`]), and every layer that holds a quantizer holds this.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::Fp8QuantPick;

/// The shared quantizer's module and entry point — the same string in both
/// slots, because the file and the kernel share a name.
pub const FP8_QUANT_MODULE: &str = "per_token_group_quant_fp8";
/// Entry point of the shared quantizer.
pub const FP8_QUANT_ENTRY: &str = "per_token_group_quant_fp8";
/// Module (file stem) of the Hopper twin.
pub const FP8_QUANT_HOPPER_MODULE: &str = "fp8_act_quant_hopper";
/// Entry point of the Hopper twin. A NEW name, not an override of the shared
/// one: both kernels must be in the Hopper image at once, because the gate for
/// the twin is byte equality against the shared kernel ON DEVICE.
pub const FP8_QUANT_HOPPER_ENTRY: &str = "per_token_group_quant_fp8_hopper";

/// K-groups one Hopper CTA covers: 128 threads / 16 threads per group. SSOT
/// with `HQ_GROUPS_PER_CTA` in the `.cu`; the kernel derives its own span from
/// `gridDim.y`, so this value only has to be the one that fills the block.
pub const FP8_QUANT_HOPPER_GROUPS_PER_CTA: u32 = 8;

/// The FP8 activation quantizer a layer will launch: the shared kernel, plus
/// the Hopper twin when this target has one.
#[derive(Clone, Copy, Debug)]
pub struct Fp8ActQuant {
    /// `per_token_group_quant_fp8`, present on every target.
    pub shared: KernelHandle,
    /// `per_token_group_quant_fp8_hopper`, or `KernelHandle(0)` off Hopper.
    pub hopper: KernelHandle,
}

impl Default for Fp8ActQuant {
    /// No quantizer at all — what a target with no FP8 path resolves, and
    /// the negative every W8A8 selector is tested against.
    fn default() -> Self {
        Self {
            shared: KernelHandle(0),
            hopper: KernelHandle(0),
        }
    }
}

impl Fp8ActQuant {
    /// Probe both. `try_kernel` for each: the shared one is optional on targets
    /// with no FP8 path at all (the callers already gate on
    /// [`Self::available`]), and the twin is optional everywhere but Hopper.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        Self {
            shared: crate::layers::try_kernel(gpu, FP8_QUANT_MODULE, FP8_QUANT_ENTRY),
            hopper: crate::layers::try_kernel(gpu, FP8_QUANT_HOPPER_MODULE, FP8_QUANT_HOPPER_ENTRY),
        }
    }

    /// A quantizer with only the shared kernel — the state every non-Hopper
    /// target resolves to, and the one unit tests construct.
    pub fn shared_only(shared: KernelHandle) -> Self {
        Self {
            shared,
            hopper: KernelHandle(0),
        }
    }

    /// Is there a quantizer to launch at all? The replacement for the
    /// `handle.0 != 0` test every W8A8 selector used to spell inline.
    ///
    /// EITHER handle, not [`Self::pick`]'s: the pick is width-dependent, and a
    /// caller asking "is there an FP8 path on this target" is not asking about
    /// one launch's M.
    pub fn available(&self) -> bool {
        self.shared.0 != 0 || self.hopper.0 != 0
    }

    /// Is the twin in this image? PRESENCE, which is a property of the build —
    /// not "will the next launch use it", which is [`Fp8QuantPick::twin`].
    pub fn twin_present(&self) -> bool {
        self.hopper.0 != 0
    }

    /// Which kernel this `(m, k)` launches, on which grid, and why not the
    /// other one — the process's resolved lever and the compiled target's SM
    /// count. See [`Self::pick_with`] for the pure form.
    pub fn pick(&self, m: u32, k: u32) -> Fp8QuantPick {
        self.pick_with(
            super::fp8_act_quant_hopper_enabled(),
            m,
            k,
            atlas_kernels::TARGET_SM_COUNT,
        )
    }

    /// [`Self::pick`] over an explicit lever and SM count — pure, so every
    /// (M, K) the attribution prices is gradeable from a CPU test.
    pub fn pick_with(&self, requested: bool, m: u32, k: u32, sm_count: u32) -> Fp8QuantPick {
        let reject = super::fp8_act_quant_hopper_reject(
            requested,
            self.twin_present(),
            self.shared.0 != 0,
            m,
            k,
            sm_count,
        );
        let twin = reject.is_none();
        Fp8QuantPick {
            kernel: if twin { self.hopper } else { self.shared },
            grid: fp8_quant_grid(twin, m, k),
            twin,
            reject,
            requested,
        }
    }

    /// The handle this `(m, k)` launches.
    pub fn kernel(&self, m: u32, k: u32) -> KernelHandle {
        self.pick(m, k).kernel
    }

    /// The grid for `(m, k)`, matching [`Self::kernel`] — from the SAME pick,
    /// so a twin handle can never reach the parent's grid.
    pub fn grid(&self, m: u32, k: u32) -> [u32; 3] {
        self.pick(m, k).grid
    }
}

/// Grid for one quantizer launch. PURE, so both arms are testable without a
/// GPU (`fp8_act_quant_tests.rs`).
///
/// `M` goes on grid X in both arms — its limit is 2^31-1, where grid Y stops at
/// 65535, and MoE `total_expanded` exceeds 65535.
///
/// Shared arm: `(M, K/128)`, one CTA per K-group, which is what the shared
/// kernel indexes with `blockIdx.y`.
///
/// Hopper arm: `(M, ceil(K/128 / 8))`. The kernel re-derives its own group span
/// as `ceil(L / gridDim.y)` rather than assuming 8, so this Y extent is a
/// PERFORMANCE choice and not a correctness contract — any Y in `1..=L` covers
/// the same groups exactly once.
pub fn fp8_quant_grid(hopper: bool, m: u32, k: u32) -> [u32; 3] {
    let groups = k / 128;
    if hopper {
        [
            m,
            groups.div_ceil(FP8_QUANT_HOPPER_GROUPS_PER_CTA).max(1),
            1,
        ]
    } else {
        [m, groups, 1]
    }
}

/// The K-group half-open range one Hopper CTA owns, mirroring the `.cu`'s
/// `gpc`/`g0`/`g_end`. Exists so `fp8_act_quant_tests.rs` can assert the
/// launcher's Y extent and the kernel's span agree — a partition of `0..L` —
/// without a device.
pub fn fp8_quant_hopper_span(groups: u32, grid_y: u32, block_y: u32) -> (u32, u32) {
    let gpc = groups.div_ceil(grid_y);
    let g0 = block_y * gpc;
    if g0 >= groups {
        return (groups, groups);
    }
    (g0, (g0 + gpc).min(groups))
}

/// Which 8 elements of a 128-element group thread `tid` of a Hopper CTA owns,
/// as a half-open `[start, end)` within the group, plus the group slot it is
/// working on. Mirrors `sub`/`lane` in the `.cu`. Returns `None` for a thread
/// whose group slot is past the CTA's span.
pub fn fp8_quant_hopper_lane(tid: u32, span: u32) -> Option<(u32, u32, u32)> {
    const LANES: u32 = 16;
    const ELEMS: u32 = 8;
    let sub = tid / LANES;
    let lane = tid % LANES;
    if sub >= span {
        return None;
    }
    Some((sub, lane * ELEMS, lane * ELEMS + ELEMS))
}

#[cfg(test)]
#[path = "fp8_act_quant_tests.rs"]
mod fp8_act_quant_tests;
