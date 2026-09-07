// SPDX-License-Identifier: AGPL-3.0-only

//! The last thing the scheduler does: fail what it cannot finish.
//!
//! Split from `lifecycle.rs` for the =<500-line cap; the drain is one
//! self-contained step of shutdown, and its tests live beside it in
//! `shutdown_drain_tests.rs`.

use super::*;

/// Fail every request the scheduler is abandoning as it exits.
///
/// Shutdown reaches the scheduler with work in three parked states, and
/// only ONE of them used to be told: `preempted` got an error frame,
/// while `prefilling` was freed and `swapped` had its disk image deleted
/// with their sinks dropped on the floor. A dropped sink is not a quiet
/// success — it closes the client's channel, which `chat_blocking` and
/// `completions_exec` render as the generic "Inference cancelled", and
/// which a STREAMING client sees as a response that stops mid-token with
/// no error frame and no finish chunk, under an HTTP 200 that already
/// promised a complete answer. A `SIGTERM` during a rolling restart hits
/// exactly these states, so the honest report is the whole point.
///
/// Every request gets a reason that names WHICH state it died in: "your
/// prompt never finished loading" and "you were paged out to disk" are
/// different things to a client deciding whether to retry.
pub(super) fn abort_in_flight_on_shutdown(
    model: &dyn Model,
    prefilling: Vec<PrefillInProgress>,
    swapped: Vec<SwappedSeq>,
    preempted: Vec<PreemptedSeq>,
    mut spill: Option<&mut KvSpillManager>,
) {
    for mut p in preempted {
        send_error_to_sink(
            &mut p.a.sink,
            "server shutting down before preempted resume",
        );
    }
    for mut p in prefilling {
        send_error_to_sink(&mut p.sink, "server shut down during prefill");
        let seq = &mut p.seq;
        let _ = model.free_sequence(seq);
        let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
    }
    for mut s in swapped {
        send_error_to_sink(
            &mut s.sink,
            "server shut down while this request was swapped out to disk",
        );
        if let Some(spill) = spill.as_deref_mut() {
            let _ = spill.remove_file(s.swap_id);
        }
    }
}
