// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A swap must CARRY the process-scoped stores, never rebuild them.
///
/// Asserted by POINTER identity, not equality: two freshly-built empty stores
/// compare equal, so an equality check would pass on exactly the bug this
/// guards — a swap that silently drops every stored conversation and resets
/// every rate-limit bucket while looking fine.
#[test]
fn carried_state_is_the_same_allocation_not_an_equal_one() {
    let first = Carried::from_env().expect("no ATLAS_* config is set in the test environment");
    let cloned = first.clone();

    assert!(
        std::sync::Arc::ptr_eq(&first.response_store, &cloned.response_store),
        "responses must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.rate_limiter, &cloned.rate_limiter),
        "rate-limit buckets must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.conversation_store, &cloned.conversation_store),
        "stored conversations must survive a swap"
    );
}

/// Two independent `from_env()` calls are DIFFERENT allocations — which is why
/// `load_model` takes `Carried` rather than building its own.
#[test]
fn building_from_env_twice_would_lose_the_stores() {
    let first = Carried::from_env().expect("no ATLAS_* config is set in the test environment");
    let second = Carried::from_env().expect("no ATLAS_* config is set in the test environment");
    assert!(
        !std::sync::Arc::ptr_eq(&first.conversation_store, &second.conversation_store),
        "if this ever passes, from_env has become a singleton and the carried \
         parameter is no longer what protects the stores — re-check the swap"
    );
}

#[test]
fn carried_uses_the_process_limiter_rather_than_minting_its_own() {
    // Handlers refund through `AppState.rate_limiter` and the middleware debits
    // through the host's. If those are two instances, refunds credit buckets
    // the middleware never debited and the accounting silently drifts.
    let host = crate::main_modules::model_host::ModelHost::empty();
    let carried = Carried::from_env().expect("no ATLAS_* config is set in the test environment");
    let process = carried.rate_limiter.clone();
    host.set_process(carried);
    assert!(
        std::sync::Arc::ptr_eq(&host.rate_limiter().expect("installed"), &process),
        "the host's limiter IS the one the model's AppState will hold"
    );
}
