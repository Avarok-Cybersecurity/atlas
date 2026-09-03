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
//! Overrides (tuning, all optional): `ATLAS_DFLASH_RUNG_MULTI` (K at C>=2,
//! default 4), `ATLAS_DFLASH_RUNG_NARROW` (C=1 prose K, default 5),
//! `ATLAS_DFLASH_RUNG_WIDE` (C=1 code K, default = cap), `ATLAS_DFLASH_RUNG_ENTER`
//! (window hits at/above which C=1 goes wide, 60), `ATLAS_DFLASH_RUNG_LEAVE`
//! (hits at/below which it goes narrow, 55), `ATLAS_DFLASH_RUNG_DWELL` (min
//! steps between switches, 64 = one full window).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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

struct Ctl {
    /// Widest block the head was sized for (K). 0 = not configured (no DFlash).
    cap: AtomicUsize,
    /// Resolver active. False = every call returns `num_drafts` unchanged.
    armed: AtomicBool,
    /// C=1 state: `true` = wide (code).
    wide: AtomicBool,
    /// Shift register: bit i = first draft accepted, i steps ago (low 32 bits).
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
    wide: AtomicBool::new(true),
    hits: AtomicU64::new(0),
    tick: AtomicU64::new(0),
    last_switch: AtomicU64::new(0),
    last_n: AtomicUsize::new(0),
    flips: AtomicU64::new(0),
};

/// Serve-time configuration. `cap_k` is the head's widest verify width (its
/// gamma); `explicit_flag` says the operator pinned `--dflash-gamma`.
pub fn configure(cap_k: usize, explicit_flag: bool) {
    let pinned = std::env::var_os("ATLAS_DFLASH_STATIC_GAMMA").is_some()
        || (explicit_flag && std::env::var_os("ATLAS_DFLASH_GAMMA_RESOLVER").is_none());
    let armed = !pinned && cap_k >= 3;
    CTL.cap.store(cap_k, Ordering::Relaxed);
    CTL.armed.store(armed, Ordering::Relaxed);
    // Single-stream starts WIDE: code is the stronger single-stream use and
    // the prose penalty at the wide width (-13%) is the smaller mistake.
    CTL.wide.store(true, Ordering::Relaxed);
    CTL.hits.store(0, Ordering::Relaxed);
    CTL.tick.store(0, Ordering::Relaxed);
    CTL.last_switch.store(0, Ordering::Relaxed);
    if armed {
        tracing::info!(
            "DFlash GAMMA RESOLVER armed: cap K={cap_k}; C>=2 -> K={} (write-on-accept); \
             C=1 -> K={} wide / K={} narrow on first-draft hits in the last {WINDOW}: \
             enter>={} leave<={} dwell={}",
            multi_k(cap_k),
            wide_k(cap_k),
            narrow_k(cap_k),
            env_u32("ATLAS_DFLASH_RUNG_ENTER", ENTER),
            env_u32("ATLAS_DFLASH_RUNG_LEAVE", LEAVE),
            env_usize("ATLAS_DFLASH_RUNG_DWELL", DWELL as usize),
        );
    } else {
        tracing::info!(
            "DFlash gamma PINNED at K={cap_k} ({})",
            if explicit_flag {
                "--dflash-gamma explicit"
            } else {
                "ATLAS_DFLASH_STATIC_GAMMA"
            }
        );
    }
}

pub fn armed() -> bool {
    CTL.armed.load(Ordering::Relaxed)
}

fn multi_k(cap: usize) -> usize {
    env_usize("ATLAS_DFLASH_RUNG_MULTI", MULTI_K).clamp(2, cap)
}
fn narrow_k(cap: usize) -> usize {
    env_usize("ATLAS_DFLASH_RUNG_NARROW", NARROW_K).clamp(2, cap)
}
fn wide_k(cap: usize) -> usize {
    env_usize("ATLAS_DFLASH_RUNG_WIDE", cap).clamp(2, cap)
}

/// The width rule, pure: verify K for `n_active` sequences given the cap and
/// the single-stream state. Unit-tested without the GPU.
pub fn k_for(n_active: usize, cap: usize, wide: bool) -> usize {
    if n_active >= 2 {
        multi_k(cap)
    } else if wide {
        wide_k(cap)
    } else {
        narrow_k(cap)
    }
}

/// Hysteresis, pure: next single-stream state from the number of first-draft
/// hits in the last `WINDOW` steps.
pub fn next_wide(wide: bool, hits: u32) -> bool {
    if wide {
        hits > env_u32("ATLAS_DFLASH_RUNG_LEAVE", LEAVE)
    } else {
        hits >= env_u32("ATLAS_DFLASH_RUNG_ENTER", ENTER)
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
    let cap = CTL.cap.load(Ordering::Relaxed);
    let k = k_for(n_active, cap, CTL.wide.load(Ordering::Relaxed));
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
    let dwell = env_usize("ATLAS_DFLASH_RUNG_DWELL", DWELL as usize) as u64;
    if tick.saturating_sub(CTL.last_switch.load(Ordering::Relaxed)) < dwell {
        return;
    }
    let hits = register.count_ones();
    let was = CTL.wide.load(Ordering::Relaxed);
    let now = next_wide(was, hits);
    if now != was {
        CTL.wide.store(now, Ordering::Relaxed);
        CTL.last_switch.store(tick, Ordering::Relaxed);
        let flips = CTL.flips.fetch_add(1, Ordering::Relaxed) + 1;
        let cap = CTL.cap.load(Ordering::Relaxed);
        tracing::info!(
            "DFlash gamma resolver C=1 -> K={} ({}; hits={hits}/{WINDOW} tick={tick} flips={flips})",
            k_for(1, cap, now),
            if now { "wide/code" } else { "narrow/prose" },
        );
    }
}

#[cfg(test)]
#[path = "dflash_rung_tests.rs"]
mod tests;
