// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the admission-time swap-out path (`lifecycle::swap_out_sequence`).
//!
//! The bug class under test is a request that vanishes. `swap_out_sequence`
//! begins by `swap_remove`ing the victim out of `active`, so from that line on
//! it is the ONLY owner of the request. An early return before the victim is
//! handed to `spill_out_sequence` therefore drops it — and a dropped
//! `ResponseSink` is not a quiet failure but a FALSE one:
//!
//!   * blocking clients see the oneshot close, which `api::chat_blocking`
//!     renders as `500 "Inference cancelled"` — the exact wording the server
//!     uses when the CLIENT aborted, so the log and the client agree on a lie;
//!   * streaming clients see the SSE channel end under an HTTP 200 that was
//!     committed with the first token — a short but complete-looking response.
//!
//! The assertions are therefore on the transport seam the API layer actually
//! consumes, not on the return value: `Err` alone was always there.

use super::lifecycle::{resume_swapped_seq, swap_out_sequence};
use super::test_support::{PreemptStubModel, active_seq, streaming_seq};
use super::types::ActiveSeq;
use spark_runtime::kv_spill::KvSpillManager;
use std::sync::atomic::Ordering;

/// A spill pool in a per-test directory. The swap-out failures under test all
/// fire BEFORE any file is created, so nothing is written here; the manager
/// exists only because the signature demands one.
fn spill(name: &str) -> KvSpillManager {
    let dir = std::env::temp_dir().join(format!(
        "atlas-swap-out-tests-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    KvSpillManager::new(dir, 1 << 20).expect("spill dir")
}

/// `active` laid out so `swap_remove(1)` migrates a non-contiguous survivor
/// into the hole, which is the only condition under which `swap_out_sequence`
/// calls `compact_sequence` at all: slot 2 lands at index 1.
fn three_actives() -> (ActiveSeq, ActiveSeq) {
    let (a0, _rx0) = active_seq(0, 5);
    let (a2, _rx2) = active_seq(2, 9);
    (a0, a2)
}

#[test]
fn compaction_failure_reaches_the_blocking_client_instead_of_dropping_it() {
    let model = PreemptStubModel::failing_compact("compact_sequence: slot 1 still owned");
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = active_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    let r = swap_out_sequence(&model, &mut active, 1, &mut spill("blocking"));
    assert!(r.is_err(), "compaction failed, so the swap-out must fail");

    // THE assertion. A dropped sink closes the oneshot, and
    // `api::chat_blocking` turns `Err(RecvError)` into
    // `500 "Inference cancelled"` — indistinguishable from a client abort.
    // The client must instead receive a real, server-authored error.
    let sent = victim_rx
        .try_recv()
        .expect("victim's sink was DROPPED — the client cannot tell this from its own abort");
    let Err(err) = sent else {
        panic!("a failed swap-out must not report success to the client");
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("swap-out failed") && msg.contains("compact_sequence"),
        "the client must be told what actually failed, got: {msg}"
    );

    // And the victim's GPU state must be reclaimed on the way out — the
    // silent drop leaked its KV blocks and SSM slot for the life of the serve.
    assert!(
        model.freed_slots.lock().unwrap().contains(&1),
        "victim slot never freed: {:?}",
        model.freed_slots.lock().unwrap()
    );
}

#[test]
fn compaction_failure_sends_a_terminal_frame_to_a_streaming_client() {
    let model = PreemptStubModel::failing_compact("compact_sequence: slot 1 still owned");
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    assert!(swap_out_sequence(&model, &mut active, 1, &mut spill("streaming")).is_err());

    // A streaming victim is already on a committed HTTP 200. Ending the
    // channel with no frame truncates the body under a success status; the
    // SDK sees a short-but-complete response. An Error frame is the only
    // thing that makes the failure visible.
    match victim_rx.try_recv() {
        Ok(crate::api::StreamEvent::Error(msg)) => assert!(
            msg.contains("swap-out failed"),
            "terminal frame must name the failure, got: {msg}"
        ),
        Ok(_) => panic!("expected an Error frame, got a different StreamEvent"),
        Err(e) => panic!(
            "no terminal frame on the victim's stream ({e:?}) — the body is \
             truncated under an already-committed HTTP 200"
        ),
    }
}

#[test]
fn a_successful_swap_out_still_hands_back_the_sink_untouched() {
    // Guard against "fix" by always erroring: on the happy path the sink must
    // travel into the `SwappedSeq` with nothing sent on it, because the
    // request is still alive and will resume later.
    let model = PreemptStubModel::default();
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = active_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    let s = swap_out_sequence(&model, &mut active, 1, &mut spill("ok")).expect("swap-out");
    assert_eq!(s.output_tokens.len(), 2);
    assert_eq!(
        model.compact_calls.load(Ordering::SeqCst),
        1,
        "the migrated survivor must still be compacted"
    );
    assert!(
        matches!(
            victim_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ),
        "a parked request's client must hear nothing yet"
    );
}

// ── swap-IN (`lifecycle::resume_swapped_seq`) ────────────────────────────
//
// The mirror image of the above. The scheduler's swap-in loop does
// `swapped.remove(idx)` and hands the `SwappedSeq` to `resume_swapped_seq`
// BY VALUE, and its only `Err` handling is `tracing::error!("Swap-in
// failed")`. So every fallible step inside that function was the last owner
// of a live request, and returning early erased it.

/// Park `victim` on disk so it can be handed back to the swap-in path, and
/// return the spill pool holding its image. Uses the real spill+restore
/// round trip rather than a hand-built `SwappedSeq` so the test cannot drift
/// from the shape the scheduler actually produces.
fn park(
    model: &PreemptStubModel,
    victim: ActiveSeq,
    name: &str,
) -> (super::types::SwappedSeq, KvSpillManager) {
    let mut sp = spill(name);
    let (a0, a2) = three_actives();
    let mut active = vec![a0, victim, a2];
    let s = swap_out_sequence(model, &mut active, 1, &mut sp).expect("park");
    (s, sp)
}

#[test]
fn swap_in_failure_reaches_the_blocking_client_instead_of_dropping_it() {
    let model = PreemptStubModel::default();
    let (victim, mut victim_rx) = active_seq(1, 2);
    let (s, mut sp) = park(&model, victim, "swapin-blocking");

    // `restore_sequence_state` is the stub's trait default, which bails —
    // the same shape as a real restore failing on a short or corrupt image.
    let r = resume_swapped_seq(None, None, &model, s, &mut sp);
    assert!(r.is_err(), "restore failed, so the resume must fail");

    let sent = victim_rx.try_recv().expect(
        "the parked request's sink was DROPPED — its client reads a server-side \
         swap-in failure as its own abort",
    );
    let Err(err) = sent else {
        panic!("a failed swap-in must not report success to the client");
    };
    assert!(
        format!("{err:#}").contains("swap-in failed"),
        "got: {err:#}"
    );
}

#[test]
fn swap_in_failure_sends_a_terminal_frame_to_a_streaming_client() {
    let model = PreemptStubModel::default();
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let (s, mut sp) = park(&model, victim, "swapin-streaming");

    assert!(resume_swapped_seq(None, None, &model, s, &mut sp).is_err());

    match victim_rx.try_recv() {
        Ok(crate::api::StreamEvent::Error(msg)) => assert!(
            msg.contains("swap-in failed"),
            "terminal frame must name the failure, got: {msg}"
        ),
        Ok(_) => panic!("expected an Error frame, got a different StreamEvent"),
        Err(e) => panic!(
            "no terminal frame on the parked request's stream ({e:?}) — the body \
             is truncated under an already-committed HTTP 200"
        ),
    }
}
