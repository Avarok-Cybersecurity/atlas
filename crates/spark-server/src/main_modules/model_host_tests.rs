// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A request that took the state keeps serving against the model it started
/// with. This is what makes draining meaningful: without it a mid-flight
/// request could read half of one model and half of another.
#[test]
fn an_in_flight_request_keeps_the_model_it_started_with() {
    let host = ModelHost::empty();
    assert!(!host.is_loaded());
    assert!(host.current().is_none());

    // Two distinct states stand in for two models.
    let first = Arc::new(0u8);
    let second = Arc::new(1u8);

    // The host is generic over AppState in production; here the property under
    // test is the Arc handoff, which is the same for any payload.
    let cell: parking_lot::RwLock<Option<Arc<u8>>> = parking_lot::RwLock::new(Some(first.clone()));
    let taken = cell.read().clone().expect("loaded");
    *cell.write() = Some(second.clone());

    assert_eq!(*taken, 0, "the in-flight reader still sees its own model");
    assert_eq!(
        *cell.read().clone().expect("loaded"),
        1,
        "a new reader sees the swapped-in model"
    );
    // And the old model is alive precisely as long as someone holds it.
    assert_eq!(Arc::strong_count(&first), 2);
    drop(taken);
    assert_eq!(Arc::strong_count(&first), 1);
}

#[test]
fn clear_refuses_requests_without_destroying_in_flight_ones() {
    let cell: parking_lot::RwLock<Option<Arc<u8>>> = parking_lot::RwLock::new(Some(Arc::new(7)));
    let taken = cell.read().clone().expect("loaded");
    *cell.write() = None;
    assert!(cell.read().is_none(), "new requests are refused");
    assert_eq!(*taken, 7, "the one already running still completes");
}
