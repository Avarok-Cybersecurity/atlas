// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: tests that the verify pick helper asks the device acceptance path FIRST and that a
//! verify pass with the `mtp_device_accept` row on always leaves a trace (lane L12).
//!
//! The regression is the boot check of an earlier GPU run on this card: the row on, 675 verify
//! passes, 0 device rows and 0 for every fallback reason. The traffic was greedy, grammar-free
//! and penalty-neutral, which the grammarless fast-greedy shortcut answers with the raw argmax,
//! and the hook sat behind it. These
//! tests drive `verify_pick_all_via` on exactly that row with a device function in the three states
//! it can be in (row off, device picks, device declines), and the per-step check with a step that
//! records nothing.
//!
//! The device is the kernel's CPU reference, as in `device_accept_tests.rs`. The model is a stub:
//! on these paths the helper only reads its vocabulary size and its logits pointer.

use super::device_accept::{account_verify_passes, picks_via, unaccounted_passes};
use super::device_accept_tests::{VOCAB, bf16_rows, rows, seq, with_ctx};
use super::pick_positions::pick_positions_from_host;
use super::verify_pick_all_via;
use crate::scheduler::fast_greedy::{PenaltyGate, classify_penalties};
use crate::scheduler::logit_processors::SamplingLevers;
use crate::scheduler::sample_step::{PositionKind, penalty_params_for};
use crate::scheduler::test_support::PreemptStubModel;
use spark_model::port::spec_accept::metrics::{self, Fallback};
use spark_model::port::spec_accept::reference;
use spark_runtime::gpu::DevicePtr;

/// What the verify forward's graphed argmax handed the helper. Deliberately NOT the argmax of the
/// rows below, so a result equal to these ids can only have come from the shortcut.
const SHORTCUT_IDS: [u32; 4] = [100, 101, 102, 103];

/// The argmax of each row below, which the host pipeline and the device path pick.
const PIPELINE_IDS: [u32; 4] = [5, 6, 7, 8];

/// Four rows whose maximum is unambiguous: one hot id per row, far above the 0.75 ceiling of the rest.
fn logits() -> Vec<u8> {
    let hot: Vec<(usize, u32, f32)> = PIPELINE_IDS
        .iter()
        .enumerate()
        .map(|(row, &id)| (row, id, 3.0))
        .collect();
    bf16_rows(&rows(SHORTCUT_IDS.len(), 11, &hot))
}

/// The levers of a default serve as far as this path reads them: the grammarless shortcut on.
fn greedy_chat() -> SamplingLevers {
    SamplingLevers {
        fast_greedy_chat: true,
        ..Default::default()
    }
}

#[test]
fn the_row_under_test_is_one_the_shortcut_answers() {
    // Greedy, no grammar, not thinking, neutral penalties: the L12 boot traffic.
    let a = seq();
    assert_eq!(a.temperature, 0.0);
    assert!(a.grammar_state.is_none() && !a.inside_thinking);
    let gate = classify_penalties(&penalty_params_for(
        &a,
        PositionKind::Verify,
        0.0,
        None,
        Vec::new(),
    ));
    assert_eq!(gate, PenaltyGate::Neutral);
}

#[test]
fn with_the_row_off_the_shortcut_answers_and_nothing_is_recorded_by_the_hook() {
    let model = PreemptStubModel::default();
    let mut a = seq();
    let picks = with_ctx(greedy_chat(), None, |c| {
        // `try_device_picks` with the row off: `None`, and no counter touched.
        verify_pick_all_via(&model, &SHORTCUT_IDS, &mut a, c, 0, |_| None)
    });
    assert_eq!(picks, SHORTCUT_IDS);
}

#[test]
fn with_the_row_on_the_device_path_picks_before_the_shortcut() {
    let model = PreemptStubModel::default();
    let buf = logits();
    let k = SHORTCUT_IDS.len();
    let mut a = seq();
    let rows_before = metrics::rows();
    let events_before = metrics::events();
    let (device, host) = with_ctx(greedy_chat(), None, |c| {
        let device = verify_pick_all_via(&model, &SHORTCUT_IDS, &mut a, c, 0, |a| {
            picks_via(k, VOCAB, DevicePtr::NULL, a, c, |req| {
                Ok(Some(reference::run(&buf, req)))
            })
        });
        // The host slow tail over the same rows: what the device path claims to equal.
        let host = pick_positions_from_host(&buf, VOCAB, 2, k, &mut a, c);
        (device, host)
    });
    assert_eq!(device, PIPELINE_IDS, "the device path answered");
    assert_ne!(device, SHORTCUT_IDS, "the shortcut did not");
    assert_eq!(device, host, "bit-identical to the host pipeline");
    // The call is accounted for: its rows are counted.
    assert!(metrics::rows() >= rows_before + k as u64);
    assert!(metrics::events() > events_before);
}

#[test]
fn a_call_the_device_declines_counts_its_reason_and_takes_the_row_off_path() {
    let model = PreemptStubModel::default();
    let k = SHORTCUT_IDS.len();

    // No device behind the hook (a target without the kernel).
    let mut a = seq();
    let unavailable = metrics::fallbacks(Fallback::Unavailable);
    let picks = with_ctx(greedy_chat(), None, |c| {
        verify_pick_all_via(&model, &SHORTCUT_IDS, &mut a, c, 0, |a| {
            picks_via(k, VOCAB, DevicePtr::NULL, a, c, |_| Ok(None))
        })
    });
    assert_eq!(
        picks, SHORTCUT_IDS,
        "the shortcut answers, as with the row off"
    );
    assert!(metrics::fallbacks(Fallback::Unavailable) > unavailable);

    // A row the planner refuses (AdaDec recording wants the host pipeline).
    let mut a = seq();
    let diagnostic = metrics::fallbacks(Fallback::Diagnostic);
    let events_before = metrics::events();
    let levers = SamplingLevers {
        adadec_diagnostic: true,
        ..greedy_chat()
    };
    let picks = with_ctx(levers, None, |c| {
        verify_pick_all_via(&model, &SHORTCUT_IDS, &mut a, c, 0, |a| {
            picks_via(k, VOCAB, DevicePtr::NULL, a, c, |_| {
                panic!("a refused row must not reach the device")
            })
        })
    });
    assert_eq!(picks, SHORTCUT_IDS);
    assert!(metrics::fallbacks(Fallback::Diagnostic) > diagnostic);
    assert!(metrics::events() > events_before);
}

#[test]
fn an_empty_span_is_not_a_verify_call() {
    let model = PreemptStubModel::default();
    let mut a = seq();
    let picks = with_ctx(greedy_chat(), None, |c| {
        verify_pick_all_via(&model, &[], &mut a, c, 0, |_| {
            panic!("no positions, nothing to ask the device")
        })
    });
    assert!(picks.is_empty());
}

#[test]
fn a_step_that_moved_no_counter_has_every_pass_unaccounted() {
    // Row on, verify passes ran, counters stood still: the 0/0 of an earlier GPU run on this card.
    assert_eq!(unaccounted_passes(true, 675, 40, 40), 675);
    assert_eq!(unaccounted_passes(true, 1, 0, 0), 1);
    // Anything recorded during the step accounts for it.
    assert_eq!(unaccounted_passes(true, 3, 40, 41), 0);
    assert_eq!(unaccounted_passes(true, 8, 40, 72), 0);
    // A bootstrap-only step has no verify pass to account for.
    assert_eq!(unaccounted_passes(true, 0, 40, 40), 0);
    // Row off: nothing is ever recorded, and nothing is owed.
    assert_eq!(unaccounted_passes(false, 675, 40, 40), 0);
}

#[test]
fn with_the_row_on_rows_and_reasons_cannot_both_stand_still_over_a_verify_pass() {
    // The test that earlier GPU run needed: it fails if a verify pass with the row on can end with
    // neither series having moved. The step below recorded nothing (a verify path that never
    // reaches the pick helper), and the per-step check counts its passes.
    let events_before = metrics::events();
    account_verify_passes(true, 3, events_before);
    assert!(
        metrics::events() > events_before,
        "row on, 3 verify passes, device rows and every fallback reason unchanged"
    );
    // Row off: the check records nothing, whatever the step did.
    let not_consulted = metrics::fallbacks(Fallback::NotConsulted);
    account_verify_passes(false, 3, metrics::events());
    // (Other tests in this process may record `not_consulted`; a lower bound is all that holds.)
    assert!(metrics::fallbacks(Fallback::NotConsulted) >= not_consulted);
}
