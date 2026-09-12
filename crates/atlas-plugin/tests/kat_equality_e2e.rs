// SPDX-License-Identifier: AGPL-3.0-only

//! The KAT equality gate, driven through the real executor against a real
//! socket — because unit tests cover the arithmetic and this covers the thing
//! that actually breaks: whether the driver issues the draw in the orders it
//! says it does and forms the verdict it says it forms.
//!
//! ## Why these are `#[ignore]`
//!
//! `load()` provisions the BFCL dataset, which needs python and the network.
//! A test that did that unconditionally would make `cargo test` depend on
//! pypi. They are ignored rather than silently skipped on a missing artifact,
//! because cargo prints `... ignored` distinctly from `... ok` — a skip that
//! prints like a pass is how a suite comes to certify nothing.
//!
//! Run them with an artifact store you can write to:
//!
//! ```text
//! HOME=/path/to/writable cargo test -p atlas-plugin --test kat_equality_e2e -- --ignored
//! ```

// Each test binary uses a different subset of the shared mock's helpers, so
// what one does not call is dead code in that binary. Same reason, same
// attribute, as `coherence.rs`.
#[allow(dead_code)]
mod mock_endpoint;

use std::time::Duration;

use atlas_plugin::{ArtifactStore, ParamValues, TargetEndpoint, VerdictKind};

/// How many samples each order issues. Small on purpose: these tests prove the
/// MECHANISM, and the statistical power to see a 12-in-995 effect is a
/// property of a real run's sample count, not of this test.
const CAP: &str = "3";

async fn run_against(port: u16) -> atlas_plugin::RunRecord {
    run_with_orders(port, "2").await
}

async fn run_with_orders(port: u16, orders: &str) -> atlas_plugin::RunRecord {
    let store = ArtifactStore::discover().expect("an artifact store");
    let executor =
        atlas_plugin::BenchmarkExecutor::new(tokio::runtime::Handle::current(), store.clone());
    let descriptor = atlas_plugin::registry::find("kat-equality-gate").expect("registered");
    let specs = descriptor.build().parameters();
    let values = ParamValues::from_overrides(
        &specs,
        [
            ("sample_cap", CAP),
            ("orders", orders),
            ("max_new_tokens", "32"),
        ],
    )
    .expect("overrides parse");
    let target = TargetEndpoint::local(port, "mock");
    tokio::task::spawn_blocking(move || {
        atlas_plugin::headless::run_blocking(
            &executor,
            atlas_plugin::headless::RunRequest {
                descriptor,
                values,
                target,
                serve_overrides: Default::default(),
                options: atlas_plugin::headless::HeadlessOptions::cli("test"),
            },
            &mut atlas_plugin::headless::SilentReporter,
            &|| false,
        )
    })
    .await
    .expect("join")
    .expect("drives")
    .record
}

/// A server that answers the same thing regardless of what it has served
/// before is order-independent, and must pass.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn a_server_that_ignores_history_passes() {
    let mock = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record = run_against(mock.port).await;
    let frame = &record.frame;
    let verdict = frame.verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Pass,
        "an order-independent server must pass: {}",
        verdict.reason
    );
    assert_eq!(frame.metrics["samples"], 3.0);
    assert_eq!(frame.metrics["orders"], 2.0);
    assert_eq!(frame.metrics["diverged"], 0.0);
    assert_eq!(frame.metrics["unmeasured"], 0.0);
    // At LEAST the 3x2 generations. Not an equality: the harness also issues
    // an endpoint probe and a coherence probe against the same path, and
    // pinning their number here would make this test fail whenever the
    // harness gained or dropped one — a test about the driver failing for a
    // reason that has nothing to do with the driver.
    // `each_extra_order_issues_exactly_one_more_pass_over_the_draw` is the
    // one that pins the per-order cost, by a difference in which the probes
    // cancel.
    assert!(
        mock.requests.load(std::sync::atomic::Ordering::Relaxed) >= 6,
        "3 samples x 2 orders is the floor"
    );
}

/// What "issue the draw once per order" actually means, measured as a
/// DIFFERENCE so the harness's own probe requests cancel out.
///
/// This is the assertion that catches a driver which issued one order twice,
/// or skipped an order, or re-issued the whole draw per order per sample. An
/// absolute count cannot: it would be pinning the harness's probe count, and
/// the first version of this test did exactly that and failed at 8 vs 6.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn each_extra_order_issues_exactly_one_more_pass_over_the_draw() {
    let two = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record_two = run_with_orders(two.port, "2").await;
    let three = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record_three = run_with_orders(three.port, "3").await;

    let n2 = two.requests.load(std::sync::atomic::Ordering::Relaxed);
    let n3 = three.requests.load(std::sync::atomic::Ordering::Relaxed);
    let cap: usize = CAP.parse().expect("CAP is a number");
    assert_eq!(
        n3 - n2,
        cap,
        "one more order must cost exactly one more pass over the draw ({n2} -> {n3})"
    );
    assert_eq!(record_two.frame.metrics["orders"], 2.0);
    assert_eq!(record_three.frame.metrics["orders"], 3.0);
    // And the third order must be COMPARED, not merely issued.
    assert_eq!(record_three.frame.metrics["identical"], cap as f64);
}

/// ★ THE CONTROL FOR THE WHOLE DRIVER.
///
/// A gate proven only against a well-behaved server has been shown to say
/// "pass", not to work. This server's reply depends on HOW MANY requests it
/// has already served and on nothing else — the purest possible form of the
/// order-dependence a KAT must not have. Every sample lands at a different
/// position in the reversed order, so every sample must be reported.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn a_server_whose_reply_depends_on_what_it_served_before_is_caught() {
    let mock =
        mock_endpoint::start_indexed(Duration::from_millis(1), Duration::from_millis(1)).await;
    let record = run_against(mock.port).await;
    let frame = &record.frame;
    let verdict = frame.verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Fail,
        "a server that answers by request count is order-dependent by construction"
    );
    assert!(
        verdict.reason.contains("ORDER-DEPENDENT"),
        "the verdict must name what it found: {}",
        verdict.reason
    );
    assert_eq!(
        frame.metrics["diverged"], 3.0,
        "with 3 samples reversed, no sample keeps its position — all three differ"
    );
    assert_eq!(frame.metrics["unmeasured"], 0.0, "every request succeeded");
}
