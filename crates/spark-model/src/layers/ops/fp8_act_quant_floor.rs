// SPDX-License-Identifier: AGPL-3.0-only

//! WHEN the Hopper FP8 activation-quant twin takes the launch: the CTA-count
//! floor, the lever that arms it, and the one line per branch that says which
//! arm ran (#928, #927).
//!
//! # The defect this closes
//!
//! Round-16 H100 receipt, `native_fp8_act_quant_hopper_microtest` on
//! `9919b1810` (§2.1, Anomaly 3, Recommendation 2). The twin is the largest
//! kernel win the campaign has measured AT PREFILL WIDTHS — 3.30-3.59x, 63.7
//! to 68.4% of HBM — and a REGRESSION at decode widths:
//!
//! | M | K | parent | twin | speed-up |
//! |---:|---:|---:|---:|---:|
//! | 16 | 5120 | 3.22 us | 3.84 us | **0.84x** |
//! | 17 | 5120 | 3.69 us | 3.89 us | **0.95x** |
//! | 25 | 5120 | 3.20 us | 4.21 us | **0.76x** |
//! | 16 | 6144 | 3.18 us | 3.88 us | **0.82x** |
//! | 17 | 6144 | 3.09 us | 3.87 us | **0.80x** |
//! | 25 | 6144 | 3.31 us | 3.83 us | **0.87x** |
//! | 16..=25 | 17408 | 3.95-4.67 us | 3.89-4.32 us | 1.02-1.08x |
//! | 1168..=4576 | 5120..=17408 | 30.8-377.8 us | 9.1-105.3 us | 3.30-3.59x |
//!
//! Six of fifteen arms were losses, and the twin was selected by kernel
//! PRESENCE — no lever, no floor — so EVERY decode-width W8A8 call took the
//! slower arm. The mechanism is the twin's own design: 8 K-groups per CTA is
//! 8x fewer CTAs, and at M <= 25 the parent's `M x K/128` grid is already under
//! one wave on 132 SMs, so dividing it by 8 removes parallelism that was doing
//! useful work.
//!
//! # The rule, and why it is CTAs rather than tokens
//!
//! The twin runs when its own grid is at least
//! [`FP8_QUANT_MIN_CTAS_PER_SM`] x `sm_count` CTAs — 264 on an H100 — which is
//! the `ssm_ba_gates_hopper` floor's number and its argument (a trailing
//! partial wave is the whole kernel at one CTA per SM). It is spelled in CTAs
//! and not in tokens because this kernel's grid is `M x ceil(K/128 / 8)`, so
//! the same M is a different amount of machine at a different K, and the
//! measurement says exactly that: at K=17408 the twin is already AHEAD at
//! M=16 while at K=5120 it is 0.76x at M=25. A token floor would have to be
//! set for the widest K and would then forfeit the K=17408 decode win, or set
//! for the narrowest and keep the K=5120 loss.
//!
//! The per-K thresholds it produces (`sm_count = 132`, floor 264 CTAs), and
//! the microtest arm each one is graded against:
//!
//! | K | K/128 | grid Y | min M | measured |
//! |---:|---:|---:|---:|---|
//! | 5120 | 40 | 5 | **53** | parent at 16/17/25 (0.76-0.95x), twin at 1168/4576 |
//! | 6144 | 48 | 6 | **44** | parent at 16/17/25 (0.80-0.87x), twin at 1168/4576 |
//! | 17408 | 136 | 17 | **16** | twin at 16/17/25 (1.02-1.08x) and above |
//!
//! That reproduces the SIGN of all fifteen measured arms, which a flat `M >=
//! 128` would not: it would put K=17408's M in {16,17,25} on the parent and
//! give back a measured 1.02-1.08x for nothing.
//!
//! ⚠️ The floor is a JUDGEMENT, like the BA-gates one, and deliberately
//! conservative in the same direction: being wrong high costs the prefill
//! lever nothing (M=1168 clears every threshold by 22-73x) and being wrong low
//! costs the decode step, which is where the loss was measured.

use spark_runtime::gpu::KernelHandle;

/// CTAs per SM the twin insists on before it will take the launch.
///
/// Two, not one, and the same value as `ssm_ba_gates_hopper`'s
/// `MIN_CTAS_PER_SM` for the same reason: at exactly one CTA per SM a trailing
/// partial wave IS the kernel.
pub const FP8_QUANT_MIN_CTAS_PER_SM: u32 = 2;

/// The smallest twin grid this device is worth launching.
pub fn fp8_quant_min_ctas(sm_count: u32) -> u32 {
    FP8_QUANT_MIN_CTAS_PER_SM.saturating_mul(sm_count.max(1))
}

/// CTAs the TWIN's grid would launch for `(m, k)` — the product of
/// [`super::fp8_quant_grid`]'s Hopper arm, read from that function rather than
/// restated, so the floor cannot come to disagree with the launch it guards.
pub fn fp8_quant_hopper_ctas(m: u32, k: u32) -> u32 {
    let [x, y, z] = super::fp8_quant_grid(true, m, k);
    x.saturating_mul(y).saturating_mul(z)
}

/// The smallest `M` the twin accepts at this `K` — the table in this module's
/// header, as a function. Exists for the tests and for anyone reading a serve
/// log's refusal and asking "at what width would it have run?".
pub fn fp8_quant_hopper_min_m(k: u32, sm_count: u32) -> u32 {
    let [_, y, _] = super::fp8_quant_grid(true, 1, k);
    fp8_quant_min_ctas(sm_count).div_ceil(y.max(1))
}

/// Is the twin selected? — `[defaults] fp8_act_quant_hopper`, with
/// `ATLAS_FP8_ACT_QUANT_HOPPER` overriding ([`super::target_defaults`]).
///
/// No `ATLAS_NO_*` rung: the two kernels emit the same bytes, so there is no
/// accuracy question for a kill switch to outrank what
/// `ATLAS_FP8_ACT_QUANT_HOPPER=0` already says.
pub fn fp8_act_quant_hopper_enabled() -> bool {
    super::target_defaults::resolved()
        .fp8_act_quant_hopper
        .value
}

/// The floor's guard string, named because it is the one a decode-width serve
/// log will print.
pub const FP8_QUANT_TOO_FEW_CTAS: &str = "too few CTAs to fill the device at 8 K-groups per CTA";

/// Every guard [`fp8_act_quant_hopper_reject`] can return, in the order it
/// tests them — and therefore the log's slot table. A list rather than bare
/// literals at the call site because [`fp8_quant_log`] gives each ONE its own
/// once-flag, and a reason with no slot would silently share another's.
pub const FP8_QUANT_REJECTS: [&str; 3] = [
    "not requested",
    "kernel absent from this image (kernels/hopper only)",
    FP8_QUANT_TOO_FEW_CTAS,
];

/// Why the twin is NOT running — `None` means it is.
///
/// Pure, so the grammar is gradeable without a GPU or the process environment,
/// and it NAMES the guard that refused: a perf path that asks to be enabled
/// and silently is not measures as "no effect".
///
/// `parent_present` is LOAD-BEARING and not defensive padding. A rejection
/// hands the launch to `Fp8ActQuant::shared`, so with no parent to hand it to
/// there is nothing to decline TO and the twin runs — which is exactly the
/// pair `native_fp8_act_quant_hopper_microtest` builds to force each arm at
/// every `M` it measures (`shared: KernelHandle(0)`). Without this branch the
/// microtest's Hopper arm would launch handle 0 at the six small-M arms whose
/// numbers opened this lever.
///
/// It is tested ABOVE the lever as well as above the floor, deliberately: an
/// operator's `ATLAS_FP8_ACT_QUANT_HOPPER=0` means "prefer the parent", and on
/// a pair that has none it must not come to mean "launch nothing".
pub fn fp8_act_quant_hopper_reject(
    requested: bool,
    twin_present: bool,
    parent_present: bool,
    m: u32,
    k: u32,
    sm_count: u32,
) -> Option<&'static str> {
    if !twin_present {
        Some(FP8_QUANT_REJECTS[1])
    } else if !parent_present {
        None
    } else if !requested {
        Some(FP8_QUANT_REJECTS[0])
    } else if fp8_quant_hopper_ctas(m, k) < fp8_quant_min_ctas(sm_count) {
        Some(FP8_QUANT_TOO_FEW_CTAS)
    } else {
        None
    }
}

/// Which kernel this launch runs, on what grid, and why the other one did not.
///
/// Entry point AND grid together, which is the invariant `Fp8ActQuant` was
/// introduced for: the two kernels need different grids, and a verdict that
/// handed back only a handle would let one kernel reach the other's grid.
///
/// No `PartialEq`: `KernelHandle` is a foreign newtype without one, and the
/// thing a test wants to compare is the VERDICT (`twin`, `reject`, `grid`),
/// not two opaque handles.
#[derive(Debug, Clone, Copy)]
pub struct Fp8QuantPick {
    pub kernel: KernelHandle,
    pub grid: [u32; 3],
    /// `true` when `kernel` is the Hopper twin.
    pub twin: bool,
    /// `None` when the twin runs; the named guard when the parent does.
    pub reject: Option<&'static str>,
    /// What the lever said, carried so the log does not read it a second time.
    pub requested: bool,
}

/// Which line [`fp8_quant_log`] would say for this verdict — `None` for
/// silence.
///
/// ONE slot per branch. The round-15 H100 serve logs carried the BA-gates
/// twin's refusal line for the life of the process while nsys showed the twin
/// running, because a single `Once` was tripped by a 27-token smoke request
/// before the first real prefill. The verdict here changes between calls for
/// the same reason — the width is an argument — so this lever gets the slot
/// table from the start: a serve log carries BOTH lines, the positive at the
/// first prefill width and the negative at the first decode width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8QuantLogSlot {
    /// The twin took the launch.
    Twin,
    /// The parent took it, for the guard at this index of
    /// [`FP8_QUANT_REJECTS`].
    Reject(usize),
}

/// Total once-flags [`fp8_quant_log`] keeps: one per guard, plus the twin's.
pub const FP8_QUANT_LOG_SLOTS: usize = FP8_QUANT_REJECTS.len() + 1;

/// The slot a verdict belongs to. Pure, so the once-set can be replayed on a
/// CPU against a real serve's call order.
pub fn fp8_quant_log_slot(pick: &Fp8QuantPick) -> Option<Fp8QuantLogSlot> {
    match pick.reject {
        None => Some(Fp8QuantLogSlot::Twin),
        // The lever is off, or this target has no twin at all: the parent is
        // the ANSWER, not a refusal, and a line per process saying so is noise
        // on gb10, b200 and strix, which is every target but one.
        Some(_) if !pick.requested => None,
        Some(why) => FP8_QUANT_REJECTS
            .iter()
            .position(|r| *r == why)
            .map(Fp8QuantLogSlot::Reject),
    }
}

/// Say WHICH quantizer runs and, when the lever asked for the twin and did not
/// get it, WHICH guard refused — once per process PER BRANCH.
///
/// Not once per call: this is reached per projection per layer per step, so an
/// unconditional line is dozens per request.
pub fn fp8_quant_log(pick: &Fp8QuantPick, m: u32, k: u32) {
    static SAID: [std::sync::Once; FP8_QUANT_LOG_SLOTS] =
        [const { std::sync::Once::new() }; FP8_QUANT_LOG_SLOTS];
    let Some(slot) = fp8_quant_log_slot(pick) else {
        return;
    };
    let idx = match slot {
        Fp8QuantLogSlot::Twin => FP8_QUANT_LOG_SLOTS - 1,
        Fp8QuantLogSlot::Reject(i) => i,
    };
    let [gx, gy, _] = pick.grid;
    SAID[idx].call_once(|| match pick.reject {
        Some(why) => tracing::info!(
            "FP8 act-quant: the Hopper twin is NOT running at M={m} K={k}: {why} \
             (ATLAS_FP8_ACT_QUANT_HOPPER)"
        ),
        None => tracing::info!(
            "FP8 act-quant: per_token_group_quant_fp8_hopper \
             (ATLAS_FP8_ACT_QUANT_HOPPER) M={m} K={k} grid=({gx},{gy},1) block=128"
        ),
    });
}
