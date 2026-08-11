// SPDX-License-Identifier: AGPL-3.0-only

//! The exit status of a faulted process is a COMPOSITION property (issue #429).
//!
//! `atlas_core::fault::exit_code` is a pure function with its own positive and
//! negative unit tests. Those prove the mapping is right; they cannot prove
//! `main` calls it. That gap is not theoretical — this campaign has already
//! shipped two fixes that were provably correct and entirely INERT because a
//! later writer clobbered the field they set (`StreamState.guard_stop`), and
//! the unit tests stayed green throughout.
//!
//! So this reads the source, like `cancel_guard_tests` and `coverage_map_tests`
//! do for the same reason: the property is structural, so the test is too.
//!
//! `main` has TWO exits — the startup escape hatch and the normal tail — and a
//! fault can latch on either side of the point that separates them (weight
//! upload and warmup both run before the server accepts). Covering one and not
//! the other is the shape of the original bug, so both are pinned.

const MAIN_RS: &str = include_str!("main.rs");

/// POSITIVE: `main` derives its exit status from the fault latch at all.
///
/// PROVEN BY: removing BOTH call sites turns this red. Measured, not assumed —
/// reverting only the tail to a bare `result` leaves this test GREEN, because
/// the startup escape still names the latch. That is the whole reason the
/// count assertion below exists: this test alone would pass over a half-fix,
/// which is the exact state the original bug shipped in.
#[test]
fn main_consults_the_fault_latch_before_exiting() {
    assert!(
        MAIN_RS.contains("fault::global().fault()"),
        "main does not consult the fault latch on the way out — a poisoned \
         context will exit 0 and `restart: on-failure` will not restart it"
    );
}

/// POSITIVE: BOTH exit paths map through `exit_code`, not just the tail.
///
/// PROVEN BY: deleting either call site turns this red with the observed count
/// (`1`), which is why the assert compares a count rather than a bool — a
/// `contains` check passes with one site covered and one bare, which is
/// precisely the half-fixed state this guards against.
#[test]
fn both_of_mains_exit_paths_map_the_fault_onto_the_status() {
    let sites = MAIN_RS.matches("fault::exit_code(").count();
    assert_eq!(
        sites, 2,
        "expected both of main's exits (startup escape + normal tail) to map \
         through fault::exit_code, found {sites}"
    );
}

/// NEGATIVE: the healthy path still returns `result` untouched.
///
/// Without this, "always exit nonzero" would satisfy the positives above while
/// restart-looping every server that was cleanly asked to stop — a worse
/// outage than the one being fixed. Pins that a `None` arm exists and yields
/// the run's own status.
///
/// PROVEN BY: replacing the `None => result` arm with an unconditional
/// `std::process::exit(EXIT_GPU_FAULT)` turns this red.
#[test]
fn a_healthy_run_still_returns_its_own_result() {
    assert!(
        MAIN_RS.contains("None => result"),
        "main no longer has a healthy arm that returns the run's own status; a \
         clean shutdown would then exit nonzero and restart-loop"
    );
}
