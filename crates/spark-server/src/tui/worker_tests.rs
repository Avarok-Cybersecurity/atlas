// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_result_arrives_on_the_channel() {
    let rx = spawn("atlas-test", || 42u32, |_| 0);
    assert_eq!(rx.recv().expect("a result"), 42);
}

#[test]
fn the_thread_is_named_so_it_is_identifiable_in_a_dump() {
    // Every worker this dashboard starts is named; an unnamed thread in a
    // stack dump during an incident is a thread nobody can attribute.
    let rx = spawn(
        "atlas-named",
        || std::thread::current().name().map(str::to_string),
        |_| None,
    );
    assert_eq!(rx.recv().unwrap().as_deref(), Some("atlas-named"));
}

#[test]
fn a_dropped_receiver_does_not_panic_the_worker() {
    // The UI moves on all the time — navigating away, quitting mid-fetch. A
    // send to nobody must be a no-op, not a panic on a background thread.
    let rx = spawn("atlas-dropped", || 1u8, |_| 0);
    drop(rx);
    // Give the worker a moment to run and attempt its send.
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[test]
fn work_that_panics_disconnects_rather_than_hanging() {
    // `Disconnected` is the codebase's "the producer is gone" idiom, and every
    // poller already handles it by clearing its spinner. What must NOT happen
    // is a receiver that stays Empty forever.
    let rx = spawn("atlas-panicky", || -> u8 { panic!("boom") }, |_| 0);
    match rx.recv() {
        Err(_) => {} // disconnected: correct
        Ok(v) => panic!("a panicking worker must not produce a value: {v}"),
    }
}
