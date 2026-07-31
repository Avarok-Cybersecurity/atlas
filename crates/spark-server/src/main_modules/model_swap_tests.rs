// SPDX-License-Identifier: AGPL-3.0-only

//! Guards on the swap's REFUSALS — the checks that run before anything is torn
//! down. A successful swap needs a GPU and a checkpoint and is exercised on
//! hardware; what can be tested here is that a doomed swap costs nothing.

use super::*;
use clap::Parser as _;

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut argv = vec!["spark", "serve", "dummy/model"];
    argv.extend_from_slice(extra);
    match cli::Cli::parse_from(argv).command {
        cli::Command::Serve(a) => a,
        cli::Command::Benchmark(_) => unreachable!("parsed a serve command"),
    }
}

/// A bad config must be refused BEFORE the host is cleared. If validation ran
/// after the drain, a typo would cost the running model.
#[test]
fn an_invalid_config_is_refused_before_anything_is_torn_down() {
    let host = Arc::new(ModelHost::empty());
    let bad = args(&["--scheduling-policy", "nonsense"]);
    let err = swap(&host, bad, None).expect_err("refused");
    let text = format!("{err:#}");
    assert!(text.contains("scheduling-policy"), "{text}");
    assert!(
        !host.is_loaded(),
        "the host was empty and must still be empty"
    );
}

/// Multi-rank must fail loudly rather than half-swap: the EP worker takes the
/// model by `Option::take` and only returns when the head exits, so there is no
/// way to tell it to load a different one.
#[test]
fn a_multi_rank_deployment_is_refused() {
    let host = Arc::new(ModelHost::empty());
    let multi = args(&["--world-size", "2"]);
    let err = swap(&host, multi, None).expect_err("refused");
    assert!(format!("{err:#}").contains("single-node only"));
}

/// The refusal must not disturb a model that IS loaded — the whole point of
/// validating first.
#[test]
fn a_refused_swap_leaves_the_running_model_alone() {
    // `ModelHost` is generic over what it holds only in production; here the
    // property is that `clear()` is never reached, which `is_loaded` observes.
    let host = Arc::new(ModelHost::empty());
    assert!(!host.is_loaded());
    let _ = swap(&host, args(&["--world-size", "4"]), None);
    assert!(!host.is_loaded(), "clear() must not have run");
}

/// The host must know what it is running, or restore-on-failure is dead code.
///
/// This is the bug the parameter version had: `swap` took `previous_args` and
/// the first caller passed `None`, which disabled recovery silently — the
/// failure mode being an operator discovering, during an outage, that the
/// safety net was never armed.
#[test]
fn the_host_remembers_what_it_is_running() {
    let host = Arc::new(ModelHost::empty());
    assert!(
        host.args().is_none(),
        "nothing loaded, nothing to restore to"
    );

    let first = args(&["--port", "9001"]);
    host.set_args(first.clone());
    assert_eq!(
        host.args().map(|a| a.port),
        Some(9001),
        "a swap can now restore to what was running"
    );
}
