// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DFlash GAMMA RESOLVER: the per-step verify width for a block-diffusion
//! drafter, chosen by CONCURRENCY and, single-stream, by WORKLOAD.
//!
//! Measured on Qwen3.8-27B NVFP4 + DFlash-2 (block 8), one Spark, 2026-09-03:
//!
//! | C     | prose (Volvo)        | code (MinHeap)        |
//! |-------|----------------------|-----------------------|
//! | 1     | γ5 27.0 / γ10 24.6   | γ10 66.8 / γ4 ~23     |
//! | 16    | γ4 187.7 / γ5 119.5  | γ4 269.7 / γ10 214.2  |
//!
//! Two facts fall out. (1) At C>=2 the K=4 write-on-accept verify wins BOTH
//! workloads: the state is read once and written once per step, and that
//! kernel exists only for K=4. (2) At C=1 the verify pass is a fixed cost, so
//! the width should track how many drafts the target will accept: code
//! accepts ~7.6 of 9 (p1 0.97) and wants the full block; prose accepts ~1.7
//! of 9 (p1 0.76) and wants a narrow block. `p1` (first-draft accept) is the
//! signal: it is bimodal, independent of the width, and free.
//!
//! The width is a DRAFT COUNT handed to the drafter per propose; the head runs
//! that many block rows (never more than it was sized for) and the verify
//! graphs on both sides are keyed by width, so after each width's first
//! capture a switch costs nothing. Sticky with hysteresis and a dwell, never
//! per step.
//!
//! Pinned (no resolver) when the operator passes `--dflash-gamma` explicitly
//! (a record run is a fixed configuration with a reproduction key) or sets
//! `ATLAS_DFLASH_STATIC_GAMMA` (presence). `ATLAS_DFLASH_GAMMA_RESOLVER`
//! (presence) turns it on even under an explicit flag, which then acts as the
//! cap. On the record: a resolved single-stream run is NOT reproducible by
//! any fixed-gamma run once it switches mid-stream; records stay per fixed
//! config.
//!
//! The trigger is an EXACT 64-step window, not a smoothed average: one bit per
//! verify step (first draft accepted or not) in a shift register, decided on
//! the popcount. A 10-step EWMA of a 0.76 process has sigma ~0.10 and brushed
//! the enter line about once per hundred steps (measured 2026-09-03: a flip
//! up and, one dwell later, a flip back, inside a single prose request).
//! Sized by simulation (300 seeds x 2000 steps per cell): at 60/55 of 64 with
//! a 64-step dwell, steady prose (0.76) flips 0.11 times per 2000 steps and
//! steady code (0.97) 0.01; a real transition flips once, ~65 steps after it
//! (~8 s at C=1). A 32-bit register at 29/27 was tried first and flipped 12.8
//! times per 2000 prose steps: the tails of Bin(32, 0.76) are fat.
//!
//! Overrides (tuning, all optional, read ONCE in [`configure`] and held in
//! atomics so the per-step path is plain loads): `ATLAS_DFLASH_RUNG_MULTI`
//! (K at C>=2, default 4), `ATLAS_DFLASH_RUNG_NARROW` (C=1 prose K, default
//! 5), `ATLAS_DFLASH_RUNG_WIDE` (C=1 code K, default = cap),
//! `ATLAS_DFLASH_RUNG_ENTER` (window hits at/above which C=1 goes wide, 60),
//! `ATLAS_DFLASH_RUNG_LEAVE` (hits at/below which it goes narrow, 55),
//! `ATLAS_DFLASH_RUNG_DWELL` (min steps between switches, 64 = one window).
//!
//! The C>=2 rung is K=4 BECAUSE of the write-on-accept kernel (#844). With
//! that kernel switched off (`ATLAS_NO_GDN_WOA=1`) the rung falls back to
//! the cap: a K=4 parent-kernel serve has no receipt in either PR, and every
//! width this resolver hands out by default must have one.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Verify width K at C>=2: the write-on-accept kernel's width.
const MULTI_K: usize = 4;
/// Single-stream narrow width (prose): K=5 measured 27.0 vs 24.6 at K=10.
const NARROW_K: usize = 5;
/// Window length in verify steps (bits of the shift register; the whole u64).
pub const WINDOW: u32 = 64;
/// Go wide when at least this many of the last `WINDOW` first drafts were
/// accepted (60/64 = 0.9375; prose sits ~0.76, code ~0.97).
const ENTER: u32 = 60;
/// Go narrow when at most this many were (55/64 = 0.859). The band between
/// holds state.
const LEAVE: u32 = 55;
/// Minimum verify steps between two switches: one full window, so every
/// decision is made on a fully refreshed register.
const DWELL: u64 = WINDOW as u64;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The resolved rungs and thresholds for one serve: every width already
/// clamped to `2..=cap`. Built once by [`configure`]; the pure rules take it
/// by reference so tests pass constants and never touch the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rungs {
    /// K at C>=2.
    pub multi: usize,
    /// C=1 narrow (prose) K.
    pub narrow: usize,
    /// C=1 wide (code) K.
    pub wide: usize,
    /// Window hits at/above which C=1 goes wide.
    pub enter: u32,
    /// Window hits at/below which C=1 goes narrow.
    pub leave: u32,
    /// Minimum verify steps between two switches.
    pub dwell: u64,
}

impl Rungs {
    /// The measured defaults for a head whose widest verify width is `cap`.
    pub fn defaults(cap: usize) -> Self {
        Self {
            multi: MULTI_K.clamp(2, cap),
            narrow: NARROW_K.clamp(2, cap),
            wide: cap.max(2),
            enter: ENTER,
            leave: LEAVE,
            dwell: DWELL,
        }
    }

    /// Defaults with the environment overrides applied, and the C>=2 rung
    /// pinned at the cap when the write-on-accept kernel is off.
    pub fn from_env(cap: usize, woa_available: bool) -> Self {
        let d = Self::defaults(cap);
        Self {
            multi: if woa_available {
                env_usize("ATLAS_DFLASH_RUNG_MULTI", d.multi).clamp(2, cap.max(2))
            } else {
                cap.max(2)
            },
            narrow: env_usize("ATLAS_DFLASH_RUNG_NARROW", d.narrow).clamp(2, cap.max(2)),
            wide: env_usize("ATLAS_DFLASH_RUNG_WIDE", d.wide).clamp(2, cap.max(2)),
            enter: env_u32("ATLAS_DFLASH_RUNG_ENTER", d.enter),
            leave: env_u32("ATLAS_DFLASH_RUNG_LEAVE", d.leave),
            dwell: env_usize("ATLAS_DFLASH_RUNG_DWELL", d.dwell as usize) as u64,
        }
    }
}

struct Ctl {
    /// Widest block the head was sized for (K). 0 = not configured (no DFlash).
    cap: AtomicUsize,
    /// Resolver active. False = every call returns `num_drafts` unchanged.
    armed: AtomicBool,
    /// The resolved [`Rungs`], one atomic per field (plain loads per step).
    multi: AtomicUsize,
    narrow: AtomicUsize,
    wide_k: AtomicUsize,
    enter: AtomicU32,
    leave: AtomicU32,
    dwell: AtomicU64,
    /// C=1 state: `true` = wide (code).
    wide: AtomicBool,
    /// Shift register: bit i = first draft accepted, i steps ago (`WINDOW`
    /// bits, the whole u64).
    hits: AtomicU64,
    /// Verify steps observed at C=1 (dwell clock; also fills the register).
    tick: AtomicU64,
    last_switch: AtomicU64,
    /// Width of the last `drafts_for` call: the observer only scores C=1 steps.
    last_n: AtomicUsize,
    flips: AtomicU64,
}

static CTL: Ctl = Ctl {
    cap: AtomicUsize::new(0),
    armed: AtomicBool::new(false),
    multi: AtomicUsize::new(MULTI_K),
    narrow: AtomicUsize::new(NARROW_K),
    wide_k: AtomicUsize::new(0),
    enter: AtomicU32::new(ENTER),
    leave: AtomicU32::new(LEAVE),
    dwell: AtomicU64::new(DWELL),
    wide: AtomicBool::new(true),
    hits: AtomicU64::new(0),
    tick: AtomicU64::new(0),
    last_switch: AtomicU64::new(0),
    last_n: AtomicUsize::new(0),
    flips: AtomicU64::new(0),
};

fn rungs() -> Rungs {
    Rungs {
        multi: CTL.multi.load(Ordering::Relaxed),
        narrow: CTL.narrow.load(Ordering::Relaxed),
        wide: CTL.wide_k.load(Ordering::Relaxed),
        enter: CTL.enter.load(Ordering::Relaxed),
        leave: CTL.leave.load(Ordering::Relaxed),
        dwell: CTL.dwell.load(Ordering::Relaxed),
    }
}

/// Serve-time configuration. `cap_k` is the head's widest verify width (its
/// gamma); `explicit_flag` says the operator pinned `--dflash-gamma`;
/// `woa_available` says the K=4 write-on-accept kernel is on (the C>=2 rung
/// falls back to the cap otherwise). Reads the environment overrides once.
pub fn configure(cap_k: usize, explicit_flag: bool, woa_available: bool) {
    let pinned = std::env::var_os("ATLAS_DFLASH_STATIC_GAMMA").is_some()
        || (explicit_flag && std::env::var_os("ATLAS_DFLASH_GAMMA_RESOLVER").is_none());
    configure_with(cap_k, pinned, Rungs::from_env(cap_k, woa_available));
    if !armed() {
        tracing::info!(
            "DFlash gamma PINNED at K={cap_k} ({})",
            if explicit_flag {
                "--dflash-gamma explicit"
            } else {
                "ATLAS_DFLASH_STATIC_GAMMA"
            }
        );
    } else if !woa_available {
        tracing::info!(
            "DFlash gamma resolver: ATLAS_NO_GDN_WOA set, C>=2 rung falls back to the cap K={cap_k}"
        );
    }
}

/// The controller reset with explicit rungs (no environment). `pinned` =
/// hand `num_drafts` back unchanged. Tests drive this directly.
pub fn configure_with(cap_k: usize, pinned: bool, r: Rungs) {
    let armed = !pinned && cap_k >= 3;
    CTL.cap.store(cap_k, Ordering::Relaxed);
    CTL.multi.store(r.multi, Ordering::Relaxed);
    CTL.narrow.store(r.narrow, Ordering::Relaxed);
    CTL.wide_k.store(r.wide, Ordering::Relaxed);
    CTL.enter.store(r.enter, Ordering::Relaxed);
    CTL.leave.store(r.leave, Ordering::Relaxed);
    CTL.dwell.store(r.dwell, Ordering::Relaxed);
    // Single-stream starts WIDE: code is the stronger single-stream use and
    // the prose penalty at the wide width (-13%) is the smaller mistake.
    CTL.wide.store(true, Ordering::Relaxed);
    CTL.hits.store(0, Ordering::Relaxed);
    CTL.tick.store(0, Ordering::Relaxed);
    CTL.last_switch.store(0, Ordering::Relaxed);
    CTL.last_n.store(0, Ordering::Relaxed);
    CTL.armed.store(armed, Ordering::Relaxed);
    if armed {
        tracing::info!(
            "DFlash GAMMA RESOLVER armed: cap K={cap_k}; C>=2 -> K={}; C=1 -> K={} wide / K={} \
             narrow on first-draft hits in the last {WINDOW}: enter>={} leave<={} dwell={}",
            r.multi,
            r.wide,
            r.narrow,
            r.enter,
            r.leave,
            r.dwell,
        );
    }
}

pub fn armed() -> bool {
    CTL.armed.load(Ordering::Relaxed)
}

/// Switches so far (C=1 wide <-> narrow), for tests and the summary log.
pub fn flips() -> u64 {
    CTL.flips.load(Ordering::Relaxed)
}

/// The width rule, pure: verify K for `n_active` sequences given the rungs
/// and the single-stream state. Unit-tested without the GPU.
pub fn k_for(n_active: usize, wide: bool, r: &Rungs) -> usize {
    if n_active >= 2 {
        r.multi
    } else if wide {
        r.wide
    } else {
        r.narrow
    }
}

/// Hysteresis, pure: next single-stream state from the number of first-draft
/// hits in the last `WINDOW` steps.
pub fn next_wide(wide: bool, hits: u32, r: &Rungs) -> bool {
    if wide {
        hits > r.leave
    } else {
        hits >= r.enter
    }
}

/// Shift one step into the register, pure: returns the new register.
#[inline]
pub fn shift_in(register: u64, hit: bool) -> u64 {
    ((register << 1) | u64::from(hit)) & (u64::MAX >> (64 - WINDOW))
}

/// The per-step draft count for `n_active` sequences. `num_drafts` is the
/// serve's configured count (cap - 1) and is returned unchanged when the
/// resolver is pinned or unconfigured.
pub fn drafts_for(n_active: usize, num_drafts: usize) -> usize {
    if !armed() {
        return num_drafts;
    }
    CTL.last_n.store(n_active, Ordering::Relaxed);
    let k = k_for(n_active, CTL.wide.load(Ordering::Relaxed), &rungs());
    (k - 1).min(num_drafts).max(1)
}

/// One single-stream verify step scored: `d1_match` = the first draft was
/// accepted. Called from the DFlash verify path every step; cheap (relaxed
/// atomics) and a no-op unless the resolver is armed and the last dispatch
/// was C=1. Switches are logged on change only.
pub fn observe_step(d1_match: bool) {
    if !armed() || CTL.last_n.load(Ordering::Relaxed) != 1 {
        return;
    }
    let register = shift_in(CTL.hits.load(Ordering::Relaxed), d1_match);
    CTL.hits.store(register, Ordering::Relaxed);
    let tick = CTL.tick.fetch_add(1, Ordering::Relaxed) + 1;
    // No decision until the register has seen a full window since the last
    // switch (or since arming): partial windows are not samples.
    let r = rungs();
    if tick.saturating_sub(CTL.last_switch.load(Ordering::Relaxed)) < r.dwell {
        return;
    }
    let hits = register.count_ones();
    let was = CTL.wide.load(Ordering::Relaxed);
    let now = next_wide(was, hits, &r);
    if now != was {
        CTL.wide.store(now, Ordering::Relaxed);
        CTL.last_switch.store(tick, Ordering::Relaxed);
        let flips = CTL.flips.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::info!(
            "DFlash gamma resolver C=1 -> K={} ({}; hits={hits}/{WINDOW} tick={tick} flips={flips})",
            k_for(1, now, &r),
            if now { "wide/code" } else { "narrow/prose" },
        );
    }
}

#[cfg(test)]
#[path = "dflash_rung_tests.rs"]
mod tests;
