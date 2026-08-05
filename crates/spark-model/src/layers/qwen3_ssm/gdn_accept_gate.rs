// SPDX-License-Identifier: AGPL-3.0-only

//! Accept-EWMA gate for the general-K fused GDN verify (wyk) dispatch.
//!
//! The wyk kernel family is workload-conditional by measurement (27B dense,
//! cap11/K=12, healthy clocks, length-matched pairs): +10% to +34% tok/s on
//! code/reasoning content (mean accept 4.4-8.4), but -12% on low-accept prose
//! (accept ~1.7-2.0, step 173 ms fused vs 144 ms sequential — at low accept
//! the full-K fused launch does more work per harvested token than the
//! per-token loop). No static default is right for both, and per project
//! direction there is no env switch: the engine picks the favorable path
//! itself from the recent accept history.
//!
//! Mechanism: the verify step feeds each step's accepted-row count into an
//! EWMA (`note_verify_accept`); the GDN dispatch consults `wide_fused_favored`
//! BEFORE the step's own accept is known, so the gate is strictly predictive.
//! Hysteresis (engage at 4.0, release at 2.5) plus a 16-step dwell prevents
//! flapping around the crossover; the band spans the measured dead zone
//! between the prose regime (EWMA settles ~2.2) and the code regime (4.4+),
//! where fused and sequential are near parity, so a misprediction inside the
//! band costs ~nothing.
//!
//! State is process-global (relaxed atomics): with single-stream serving this
//! is exact; with concurrent sequences it blends their accept histories, which
//! only ever selects between two correct paths. Per-sequence gating is the
//! natural follow-up when multi-stream DFlash lands.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// EWMA of accepted rows per verify step, stored as f32 bits.
static EWMA_BITS: AtomicU32 = AtomicU32::new(0);
/// Whether the fused wide path is currently engaged.
static ENGAGED: AtomicBool = AtomicBool::new(false);
/// Steps since the last transition (saturating), for the dwell requirement.
static STEPS_SINCE_SWITCH: AtomicU32 = AtomicU32::new(u32::MAX);

/// EWMA smoothing factor: ~16-step memory. Single steps swing 0..K, so the
/// average must move slower than the per-step noise.
const ALPHA: f32 = 0.0625;
/// Engage the fused path when recent accept clears this.
const ENGAGE_AT: f32 = 4.0;
/// Release back to the sequential path when recent accept falls below this.
/// Must sit ABOVE the measured prose regime (accept ~1.7-2.4 settles the EWMA
/// near 2.2) or prose never releases; the worst measured code rough-stretch
/// bottoms out near 3.7, leaving margin on both sides.
const RELEASE_AT: f32 = 2.5;
/// Minimum steps between transitions: a workload change is a sustained shift,
/// not a bad step. Prevents oscillation when the EWMA rides near a threshold.
const MIN_DWELL_STEPS: u32 = 16;

/// Feed one verify step's accepted-row count. Called by the scheduler's
/// DFlash verify step after the accept-prefix scan.
pub fn note_verify_accept(num_accepted: usize) {
    let prev = f32::from_bits(EWMA_BITS.load(Ordering::Relaxed));
    let next = prev + ALPHA * (num_accepted as f32 - prev);
    EWMA_BITS.store(next.to_bits(), Ordering::Relaxed);

    let dwell = STEPS_SINCE_SWITCH.load(Ordering::Relaxed).saturating_add(1);
    STEPS_SINCE_SWITCH.store(dwell, Ordering::Relaxed);

    let engaged = ENGAGED.load(Ordering::Relaxed);
    let engage_now = if engaged {
        next >= RELEASE_AT
    } else {
        next >= ENGAGE_AT
    };
    if engage_now != engaged && dwell >= MIN_DWELL_STEPS {
        ENGAGED.store(engage_now, Ordering::Relaxed);
        STEPS_SINCE_SWITCH.store(0, Ordering::Relaxed);
        tracing::info!(
            "GDN WYK gate: {} (accept EWMA {:.2})",
            if engage_now {
                "ENGAGED — fused general-K verify"
            } else {
                "RELEASED — sequential verify"
            },
            next,
        );
    }
}

/// Whether the fused general-K (wyk) path is currently favored. Consulted by
/// the GDN dispatch each verify step; starts disengaged (sequential) until
/// the accept history warms past `ENGAGE_AT` (~6-8 steps on code content).
pub fn wide_fused_favored() -> bool {
    ENGAGED.load(Ordering::Relaxed)
}
