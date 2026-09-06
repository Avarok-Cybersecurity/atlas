// SPDX-License-Identifier: AGPL-3.0-only

//! What a `SIGTERM` does to requests the scheduler has parked.
//!
//! Shutdown finds work in three states — mid-prefill, swapped out to
//! disk, preempted awaiting resume — and the drain used to tell only the
//! third. The other two had their sinks dropped, which is not a quiet
//! failure: a blocking client gets the generic "Inference cancelled" and
//! a streaming client gets a response that stops mid-token with no error
//! frame and no finish chunk, under the HTTP 200 the stream already sent.
//! A rolling restart lands on exactly these states.

use super::lifecycle::swap_out_sequence;
// PreemptStubModel, not lifecycle_tests::StubModel: this test drives the
// swap-out path, and only the shared fixture implements `save_sequence_state`.
// #911's richer StubModel would also have served, but that commit is
// superseded here by 963e22f684, and test_support is the declared home for
// scheduler fixtures anyway.
use super::shutdown_drain::abort_in_flight_on_shutdown;
use super::test_support::PreemptStubModel;
use super::test_support::{RespRx, test_prefill, test_seq};
use super::types::PreemptedSeq;
use spark_runtime::kv_spill::KvSpillManager;

fn message(label: &str, mut rx: RespRx) -> String {
    match rx.try_recv() {
        Ok(Err(e)) => format!("{e:#}"),
        Ok(Ok(_)) => panic!("{label}: an abandoned request must not report success"),
        Err(e) => panic!("{label}: the client was dropped instead of told ({e})"),
    }
}

#[test]
fn shutdown_tells_every_parked_request_which_state_it_died_in() {
    let model = PreemptStubModel::default();
    let dir = std::env::temp_dir().join(format!("atlas-shutdown-drain-{}", std::process::id()));
    let mut spill = KvSpillManager::new(dir, 8 * 1024 * 1024).expect("spill manager");

    let (prefill, prefill_rx) = test_prefill(vec![1, 2, 3]);

    let (victim, swapped_rx) = test_seq(vec![4, 5], 6, None, 2);
    let mut active = vec![victim];
    let swapped = swap_out_sequence(&model, &mut active, 0, &mut spill).expect("swap-out writes");

    let (parked, preempted_rx) = test_seq(vec![7], 9, None, 1);
    let preempted = PreemptedSeq {
        a: parked,
        tokens: vec![7],
    };

    abort_in_flight_on_shutdown(
        &model,
        vec![prefill],
        vec![swapped],
        vec![preempted],
        Some(&mut spill),
    );

    // Each reason must name the state, not just "shutting down": a client
    // deciding whether to retry treats "your prompt never finished
    // loading" and "you were paged out to disk" differently.
    let m = message("prefilling", prefill_rx);
    assert!(m.contains("prefill"), "prefilling: got {m:?}");
    let m = message("swapped", swapped_rx);
    assert!(m.contains("swapped out"), "swapped: got {m:?}");
    let m = message("preempted", preempted_rx);
    assert!(m.contains("preempted"), "preempted: got {m:?}");
}
