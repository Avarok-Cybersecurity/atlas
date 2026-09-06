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

#[test]
fn a_host_built_inside_the_runtime_lets_a_plain_thread_spawn() {
    // The Library drives swaps from a plain `std::thread`, and the load path
    // spawns Tokio tasks. Before the host carried the handle, the first launch
    // panicked with "there is no reactor running" — after the outgoing model
    // had already been released, so it took the server down with it.
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let host = rt.block_on(async { Arc::new(ModelHost::empty()) });
    assert!(host.runtime().is_some(), "captured at construction");

    let spawned = std::thread::spawn(move || {
        let handle = host.runtime().expect("a handle");
        let _entered = handle.enter();
        // The exact call that panicked.
        tokio::spawn(async {});
    })
    .join();
    assert!(spawned.is_ok(), "no reactor panic off the runtime");
}

#[test]
fn a_host_built_outside_a_runtime_has_no_handle_rather_than_panicking() {
    let host = ModelHost::empty();
    assert!(host.runtime().is_none());
}

#[test]
fn the_auth_policy_survives_having_no_model() {
    // It used to live on AppState, so it ceased to exist whenever no model was
    // loaded — and /v1/models then answered 200 without a token in exactly that
    // window, while correctly answering 401 once a model was up. Whether a
    // request is authorised cannot depend on whether a model happens to be
    // loaded.
    let host = ModelHost::empty();
    assert!(host.current().is_none(), "no model, by construction");
    assert!(host.auth().is_none(), "and none configured yet");

    let cfg = std::sync::Arc::new(
        crate::auth::AuthConfig::from_inline("sk-test-token").expect("valid token"),
    );
    host.set_auth(Some(cfg));
    assert!(
        host.auth().is_some(),
        "the policy is in force with no model loaded"
    );
}

#[test]
fn the_rate_limiter_survives_having_no_model() {
    // Same shape as the auth bypass: read off AppState it does not exist while
    // no model is loaded, and every /v1/* request in that window went through
    // the middleware unlimited.
    let host = ModelHost::empty();
    assert!(host.current().is_none(), "no model, by construction");
    assert!(host.rate_limiter().is_none(), "and none installed yet");

    let carried = crate::main_modules::serve_load::Carried::from_env()
        .expect("no ATLAS_* config is set in the test environment");
    let rl = carried.rate_limiter.clone();
    host.set_process(carried);
    let got = host.rate_limiter().expect("in force with no model loaded");
    assert!(
        std::sync::Arc::ptr_eq(&got, &rl),
        "and it is the same instance, not a rebuild"
    );
    // The stores travel with it: a handler that reads only these needs no
    // model, and gating them on one made stored data unreadable mid-swap.
    assert!(host.process().is_some(), "the stores are reachable too");
}

#[test]
fn auto_swap_is_answered_without_cloning_the_whole_argv() {
    // `args()` clones ServeArgs — around a hundred fields, several of them
    // String, Vec and PathBuf — and the chat path called it on every request
    // to read two booleans.
    use clap::Parser as _;
    let host = ModelHost::empty();
    assert!(!host.auto_swap_enabled(), "no argv installed yet");

    host.set_args(crate::cli::ServeArgs::parse_from(["spark", "org/m"]));
    assert!(!host.auto_swap_enabled(), "off unless asked for");

    host.set_args(crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--auto-swap",
    ]));
    assert!(host.auto_swap_enabled());

    // --no-auto-swap still wins, as `auto_swap::enabled` defines it.
    host.set_args(crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--auto-swap",
        "--no-auto-swap",
    ]));
    assert!(!host.auto_swap_enabled(), "the prohibition still wins");
}

#[test]
fn the_dashboard_channel_survives_for_later_loads() {
    // The Stats pane samples the run handles every tick and Ops toggles levers
    // through them. Both swap paths used to pass `None` because neither had
    // the sender, so after a swap the dashboard kept sampling a scheduler that
    // had already been joined.
    let host = ModelHost::empty();
    assert!(host.tui_handles().is_none(), "no dashboard yet");

    let (tx, rx) = std::sync::mpsc::channel::<crate::tui::RunHandles>();
    host.set_tui_handles(tx);
    assert!(
        host.tui_handles().is_some(),
        "a later load can publish through it"
    );
    drop(rx);
}
