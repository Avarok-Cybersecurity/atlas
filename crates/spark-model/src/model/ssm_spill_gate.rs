// SPDX-License-Identifier: AGPL-3.0-only

//! The SPILL-side cost gate (`ATLAS_SSM_SPILL_MIN_TOKENS`) — the missing half
//! of the tier's cost model.
//!
//! The fault-in side has had a depth gate since task #5
//! (`ATLAS_SSM_FAULT_MIN_TOKENS`, `trait_impl/ssm_fault_in.rs`). The spill side
//! had none, and that asymmetry is the design gap: a spill is charged in full
//! to whichever request happened to trigger the eviction, while the benefit
//! accrues to a different, later request — so an unbounded spill rate taxes the
//! live request to bank state nobody may ever fault back (measured this
//! session: 8 spills, 0 fault-ins).

use std::sync::atomic::{AtomicBool, Ordering};

/// Budgeted cost of one spill, milliseconds. Kept adjacent to
/// [`DEFAULT_SPILL_MIN_TOKENS`] so the constant and the threshold derived from
/// it cannot drift apart.
///
/// 45 ms is the POST-FIX target (60 async D2H chunks + one stream sync into a
/// reusable pinned buffer ≈ 20-30 ms gather + the unchanged 19 ms `store.put`
/// host memcpy), not the ~400 ms the blocking-copy shape measured. Re-derive
/// this from a fresh `ATLAS_SSM_TIER_TIMING` line if the gather changes again.
const SPILL_COST_MS: usize = 45;

/// Minimum victim depth (tokens) worth spilling rather than dropping.
/// `ATLAS_SSM_SPILL_MIN_TOKENS` overrides; `0` disables the gate.
///
/// Derivation: `spill_min ≈ R × (C_s / p_target + C_f)` where `R` ≈ 6500 tok/s
/// is measured SSM prefill throughput, `C_s` = [`SPILL_COST_MS`] = 45 ms,
/// `C_f` ≈ 50 ms is the fault-in cost, and `p_target` = 0.3 is the fault-back
/// probability we require a spill to be worth its price:
/// `6500 × (0.045/0.3 + 0.05) ≈ 1300` → 1024 (a round block-aligned value just
/// under it, biased toward spilling).
///
/// NOTE the coupling: at the PRE-fix 400 ms spill cost the same formula gives
/// ~9000 tokens. This default is only defensible once the gather fix has
/// landed — which is why the two ship together.
const DEFAULT_SPILL_MIN_TOKENS: usize = 1024;

/// One-shot latch for the clamp warning: an operator needs to learn once that
/// their config is self-defeating, not once per eviction.
static CLAMP_WARNED: AtomicBool = AtomicBool::new(false);

/// The effective spill gate, already clamped to the fault-in gate.
pub(in crate::model) fn spill_min_tokens() -> usize {
    let raw = parse_spill_min_tokens(std::env::var("ATLAS_SSM_SPILL_MIN_TOKENS").ok());
    let fault = super::trait_impl::ssm_fault_in::fault_in_min_tokens();
    let eff = clamp_spill_to_fault(raw, fault);
    if eff != raw && !CLAMP_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "ATLAS_SSM_SPILL_MIN_TOKENS={raw} is below ATLAS_SSM_FAULT_MIN_TOKENS={fault}; \
             clamping the spill gate to {eff}. Spilling a snapshot the fault-in gate would \
             then REFUSE to read back is a guaranteed pure loss — the spill cost is paid and \
             the benefit can never be claimed."
        );
    }
    eff
}

/// Pure parse of `ATLAS_SSM_SPILL_MIN_TOKENS`: unset or unparseable falls back
/// to [`DEFAULT_SPILL_MIN_TOKENS`]; `0` disables the gate. Same lenient idiom
/// as `parse_fault_min_tokens` — a typo here only mis-tunes a heuristic.
pub(in crate::model) fn parse_spill_min_tokens(raw: Option<String>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SPILL_MIN_TOKENS)
}

/// INVARIANT: `spill_min >= fault_min`. Clamp, never bail — a bad gate value
/// must not take the server down. `0` (gate explicitly disabled) is preserved:
/// that is an operator asking for the pre-gate behaviour, not a mis-ordering.
pub(in crate::model) fn clamp_spill_to_fault(spill: usize, fault: usize) -> usize {
    if spill == 0 { 0 } else { spill.max(fault) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spill_min_defaults_when_unset_or_garbage() {
        assert_eq!(parse_spill_min_tokens(None), DEFAULT_SPILL_MIN_TOKENS);
        assert_eq!(
            parse_spill_min_tokens(Some("twelve".into())),
            DEFAULT_SPILL_MIN_TOKENS
        );
        assert_eq!(
            parse_spill_min_tokens(Some("".into())),
            DEFAULT_SPILL_MIN_TOKENS
        );
    }

    #[test]
    fn parse_spill_min_parses_explicit_values() {
        assert_eq!(parse_spill_min_tokens(Some("0".into())), 0);
        assert_eq!(parse_spill_min_tokens(Some("2048".into())), 2048);
    }

    /// The invariant: never spill something the fault-in gate would refuse to
    /// read back. Today's implicit config (spill 0 = ungated, fault 256)
    /// violated it in exactly this direction.
    #[test]
    fn spill_min_clamped_to_fault_min() {
        assert_eq!(clamp_spill_to_fault(64, 256), 256);
        assert_eq!(clamp_spill_to_fault(256, 256), 256);
        assert_eq!(clamp_spill_to_fault(1024, 256), 1024);
        // Explicitly disabled stays disabled.
        assert_eq!(clamp_spill_to_fault(0, 256), 0);
    }

    /// The shipped default must satisfy the invariant against the shipped
    /// fault-in default — a tripwire if either constant moves.
    #[test]
    fn shipped_defaults_satisfy_the_invariant() {
        let fault = super::super::trait_impl::ssm_fault_in::DEFAULT_FAULT_MIN_TOKENS;
        assert_eq!(
            clamp_spill_to_fault(DEFAULT_SPILL_MIN_TOKENS, fault),
            DEFAULT_SPILL_MIN_TOKENS,
            "DEFAULT_SPILL_MIN_TOKENS must already be >= the fault-in gate"
        );
        // `SPILL_COST_MS > 0` is a compile-time property of a const, so a
        // runtime assert on it is vacuous (clippy::assertions_on_constants).
        // Enforce it where it actually binds — at compile time.
        const _: () = assert!(SPILL_COST_MS > 0);
    }
}
