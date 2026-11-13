// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the consumer-card cap on how many SSM-pool slots the MTP
//! verify state pools cover (port lane L24, the 27B pool sizing on a 16 GB
//! card), published by preflight and applied by [`super::mtp_state_slots`].
//!
//! # Why
//!
//! On a 16 GB card Qwen3.8-27B leaves about 1.7 GiB beside 12.5 GiB of 4-bit
//! weights (report 77 section 12.1). Its per-sequence GDN state is 144 MiB in
//! FP32, and one verify slot at `--num-drafts 3` keeps three more h snapshots,
//! four conv snapshots and a checkpoint blob: about 600 MiB. The default
//! `--max-batch-size 8` sizes eight such slots (4.7 GiB), so the serve either
//! refuses at preflight or starves the KV cache. "MTP and eight streams do not
//! coexist at 4.25 bits" (77 section 12.3): what does fit is speculation for
//! the first slot or two, with the other streams plain-decoding.
//!
//! The scheduler already gates every speculative step on all active slots
//! being covered (`slot_idx < mtp_state_slots(bs)`, `scheduler/mod.rs`), and
//! `SsmStatePool::new` and preflight size the pools from the same function,
//! so a smaller slot count is the whole mechanism: one number, published once
//! before the load, read by all three.
//!
//! # Who it applies to
//!
//! Only a target that declares `memory_fit_consumer` (ada) runs the fit, and
//! only when the verify pools would not fit their share of the memory beside
//! the weights; a fit that keeps every slot publishes nothing. GB10, Hopper,
//! B200 and B300 never publish, so [`super::mtp_state_slots`] returns exactly
//! the pre-sprint number there (pinned by `consumer_fit_tests.rs`). On the 4060
//! Ti the measured 4B serves (batch 1 and 8, BF16 and FP8 FFN, with and
//! without `--speculative`) keep every slot. `AVAROK_MTP_POOL_FULL_WIDTH`
//! ignores the cap with the rest of the pool diet, and
//! `AVAROK_MTP_CONSUMER_FIT_OFF` (presence) skips the fit alone.
//!
//! # The fit
//!
//! `budget = (free - resident weights - base state blobs - KV floor - CUDA
//! headroom) / VERIFY_BUDGET_DIVISOR`; the cap is the largest slot count whose
//! verify bytes (the SSOT [`super::ssm_pool_reserve_bytes`] at zero base
//! slots, so the tiered per-slot widths are priced exactly) fit the budget,
//! never below one: the first stream keeps its drafts, which is the
//! single-stream target G4 is about. A configuration where even one slot does
//! not fit is left to the preflight refusal, as before.

use std::sync::OnceLock;

/// The verify pools may take at most `1 / VERIFY_BUDGET_DIVISOR` of what is
/// left beside the weights, the base state, the KV floor and the headroom.
/// The other half is KV beyond the floor and prefill workspace.
pub const VERIFY_BUDGET_DIVISOR: usize = 2;

/// Largest slot count `n <= requested` with `verify_bytes(n) <= budget`,
/// floored at one slot when `requested > 0`. `verify_bytes` must be
/// non-decreasing in `n` (every slot adds its own snapshots).
pub fn fit_verify_slots(
    requested: usize,
    budget: usize,
    verify_bytes: impl Fn(usize) -> usize,
) -> usize {
    if requested == 0 {
        return 0;
    }
    let mut n = requested;
    while n > 1 && verify_bytes(n) > budget {
        n -= 1;
    }
    n
}

/// What is left for the verify pools: memory beside the resident weights,
/// minus the terms that are paid regardless (base per-sequence state, the KV
/// floor, the CUDA headroom), divided by [`VERIFY_BUDGET_DIVISOR`].
pub fn verify_budget_bytes(
    free_mem: usize,
    resident_weights: usize,
    base_state: usize,
    kv_floor: usize,
    headroom: usize,
) -> usize {
    free_mem
        .saturating_sub(resident_weights)
        .saturating_sub(base_state)
        .saturating_sub(kv_floor)
        .saturating_sub(headroom)
        / VERIFY_BUDGET_DIVISOR
}

/// `slots` under a published cap. `full_width` (`AVAROK_MTP_POOL_FULL_WIDTH`)
/// ignores the cap, as it ignores the rest of the pool diet.
pub fn apply_consumer_cap(slots: usize, cap: Option<usize>, full_width: bool) -> usize {
    match cap {
        Some(cap) if !full_width => slots.min(cap),
        _ => slots,
    }
}

/// The cap preflight decided. First write wins, like the decode ring's cell.
/// Absent is NOT a value: a process that never ran preflight (a bench, a unit
/// test) applies no cap.
static CONSUMER_VERIFY_SLOT_CAP: OnceLock<usize> = OnceLock::new();

/// Publish the cap. Returns the cap in force.
pub fn set_consumer_verify_slot_cap(cap: usize) -> usize {
    let _ = CONSUMER_VERIFY_SLOT_CAP.set(cap);
    *CONSUMER_VERIFY_SLOT_CAP.get().expect("just set")
}

/// The published cap, or `None` when preflight published none.
pub fn published_consumer_verify_slot_cap() -> Option<usize> {
    CONSUMER_VERIFY_SLOT_CAP.get().copied()
}

/// `AVAROK_MTP_CONSUMER_FIT_OFF` (presence): skip the fit.
pub fn consumer_fit_disabled() -> bool {
    std::env::var_os("AVAROK_MTP_CONSUMER_FIT_OFF").is_some()
}

#[cfg(test)]
#[path = "consumer_fit_tests.rs"]
mod consumer_fit_tests;
