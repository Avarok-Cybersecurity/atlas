// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2 — anti-degeneration (penalty ramp + watchdog→resample).
//!
//! Root cause (controlled greedy A/B, 2026-05-18): on long structured
//! code the model falls into a repetition loop; the loop-watchdog only
//! *truncated* (kill / force-`</think>`) in both the MTP and non-MTP
//! paths. This module holds the tuning constants + the combined
//! penalty multiplier; the watchdog→escalation wiring lives in
//! `decode_logits_step.rs` and the application in `decode_logits_seq.rs`.
//!
//! Reversibility: `RESAMPLE_RAMP_SLOPE = 0.0` ⇒ ramp is identity;
//! `RESAMPLE_MAX_ESC = 0` ⇒ the content-loop watchdog finishes on
//! first detection, i.e. exactly the pre-Phase-2 kill behaviour.

// 2a — per-output-length penalty ramp. Attractors lock in well before
// the watchdog can detect them, so scale presence/lz/DRY up linearly
// once output exceeds RAMP_ONSET (capped).
pub const RESAMPLE_RAMP_ONSET: usize = 512;
pub const RESAMPLE_RAMP_SLOPE: f32 = 0.5;
pub const RESAMPLE_RAMP_PER_TOKENS: f32 = 1024.0;
pub const RESAMPLE_RAMP_CAP: f32 = 1.4;
// Absolute ceiling on effective presence_penalty — stays below the
// ~1.5 premature-EOS cliff documented in qwen3.6-27b MODEL.toml
// (6c69d3f bisect). lz/DRY have no such cliff so are uncapped here.
pub const RESAMPLE_PRESENCE_MAX: f32 = 1.45;
// 2b — convert the content-loop watchdog from kill-switch to escalating
// resample: on detection bump `resample_escalation` (compounding every
// CONTENT_LOOP_CHECK_STRIDE re-fire) so the model is steered out of the
// loop instead of the response being truncated; only after
// RESAMPLE_MAX_ESC un-cleared escalations fall back to the original
// hard finish (never regress safety).
pub const RESAMPLE_MAX_ESC: u8 = 4;
pub const RESAMPLE_ESC_STEP: f32 = 0.6;

/// Combined Phase-2 anti-repetition penalty multiplier for a sequence
/// at `n_out` output tokens with escalation level `esc`. Returns 1.0
/// (identity) at slope 0 / esc 0. The ramp component is capped at
/// `RESAMPLE_RAMP_CAP`; the escalation component compounds on top.
pub fn resample_penalty_factor(n_out: usize, esc: u8) -> f32 {
    let ramp = if n_out > RESAMPLE_RAMP_ONSET {
        1.0 + RESAMPLE_RAMP_SLOPE
            * ((n_out - RESAMPLE_RAMP_ONSET) as f32 / RESAMPLE_RAMP_PER_TOKENS)
    } else {
        1.0
    };
    ramp.min(RESAMPLE_RAMP_CAP) * (1.0 + RESAMPLE_ESC_STEP * esc as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_below_onset_and_zero_esc() {
        assert_eq!(resample_penalty_factor(0, 0), 1.0);
        assert_eq!(resample_penalty_factor(RESAMPLE_RAMP_ONSET, 0), 1.0);
    }

    #[test]
    fn ramp_increases_then_caps() {
        let mid = resample_penalty_factor(RESAMPLE_RAMP_ONSET + 1024, 0);
        assert!(mid > 1.0 && mid <= RESAMPLE_RAMP_CAP);
        let far = resample_penalty_factor(RESAMPLE_RAMP_ONSET + 1_000_000, 0);
        assert!((far - RESAMPLE_RAMP_CAP).abs() < 1e-4);
    }

    #[test]
    fn escalation_compounds_on_top_of_cap() {
        let base = resample_penalty_factor(10_000_000, 0);
        let esc2 = resample_penalty_factor(10_000_000, 2);
        assert!((esc2 - base * (1.0 + RESAMPLE_ESC_STEP * 2.0)).abs() < 1e-4);
    }
}
