// SPDX-License-Identifier: AGPL-3.0-only

use super::quiesce_streams;
use std::cell::RefCell;

/// EVERY stream is waited on, in order.
///
/// The bug this pins let `teardown` free pools while kernels were still
/// running, because the scheduler reasoned about HOST-side quiescence ("every
/// sequence freed, no request can arrive") and never waited on the device.
#[test]
fn every_stream_is_waited_on() {
    let synced = RefCell::new(Vec::<u64>::new());
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2)], |s| {
        synced.borrow_mut().push(s);
        Ok(())
    });
    assert!(failed.is_empty());
    assert_eq!(synced.into_inner(), [1, 2]);
}

/// ★ Both streams, not just the default one. Decode runs on the default stream
/// and prefill has its own for compute/copy overlap; work outstanding on either
/// can still be reading the pools. Waiting on only one is the same race with a
/// smaller window — the kind of "fix" that makes a fault rare enough to get
/// blamed on hardware.
#[test]
fn the_prefill_stream_is_not_forgotten() {
    let synced = RefCell::new(Vec::<u64>::new());
    quiesce_streams(&[("default", 7), ("prefill", 9)], |s| {
        synced.borrow_mut().push(s);
        Ok(())
    });
    assert_eq!(
        synced.into_inner(),
        [7, 9],
        "the prefill stream must be waited on too"
    );
}

/// A stream that will not synchronise is reported BY NAME, and the sweep keeps
/// going so the remaining streams are still waited on.
#[test]
fn a_failed_sync_is_named_and_does_not_stop_the_sweep() {
    let synced = RefCell::new(Vec::<u64>::new());
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2), ("extra", 3)], |s| {
        synced.borrow_mut().push(s);
        if s == 2 {
            anyhow::bail!("stream 2 is wedged")
        } else {
            Ok(())
        }
    });
    assert_eq!(failed, ["prefill"], "the failing stream is named");
    assert_eq!(
        synced.into_inner(),
        [1, 2, 3],
        "a failure must not skip the streams after it"
    );
}

/// Every stream failing names every stream — the caller logs one line each, and
/// still releases.
#[test]
fn all_failures_are_reported() {
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2)], |_| {
        anyhow::bail!("device is gone")
    });
    assert_eq!(failed, ["default", "prefill"]);
}

/// No streams is not an error — mock and translation models own no pooled
/// device memory, and their release must still run.
#[test]
fn no_streams_is_not_a_failure() {
    let called = RefCell::new(false);
    let failed = quiesce_streams(&[], |_| {
        *called.borrow_mut() = true;
        Ok(())
    });
    assert!(failed.is_empty());
    assert!(!called.into_inner(), "nothing to wait on, nothing called");
}
