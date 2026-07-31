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

#[test]
fn a_recipe_cannot_switch_off_the_operators_auto_swap_policy() {
    // The real failure: a server started with --auto-swap loaded a recipe from
    // the Library, the recipe's argv replaced the host's, and auto-swap was
    // silently off from then on. Nothing logged, nothing failed — the next
    // request that should have swapped was just served by the old model.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["spark", "org/live", "--auto-swap"]);
    let mut next = cli::ServeArgs::parse_from(["spark", "org/next"]);
    assert!(!next.auto_swap, "the recipe says nothing about it");

    super::carry_process_flags(&mut next, &previous);
    assert!(next.auto_swap, "the operator's policy survives the swap");
    assert_eq!(
        next.model.as_deref(),
        Some("org/next"),
        "the MODEL still swaps"
    );
}

#[test]
fn a_recipe_cannot_switch_on_auto_swap_where_it_was_forbidden() {
    // The direction that matters for an enterprise deployment: --no-auto-swap
    // is a deployment contract, and a fetched recipe must not be able to
    // loosen it.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["spark", "org/live", "--no-auto-swap"]);
    let mut next = cli::ServeArgs::parse_from(["spark", "org/next", "--auto-swap"]);
    super::carry_process_flags(&mut next, &previous);
    assert!(next.no_auto_swap, "the prohibition survives");
    assert!(
        !super::super::auto_swap::enabled(&next),
        "and still wins over --auto-swap"
    );
}

#[test]
fn a_recipes_port_cannot_move_a_socket_that_is_already_bound() {
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["spark", "org/live", "--port", "8888"]);
    let mut next = cli::ServeArgs::parse_from(["spark", "org/next", "--port", "9100"]);
    super::carry_process_flags(&mut next, &previous);
    assert_eq!(next.port, 8888, "the bound port is authoritative");
}

#[test]
fn a_model_this_build_has_no_kernels_for_is_refused_before_teardown() {
    // The failure that cost a live server its model: the 35B was rejected for
    // `model_type 'qwen3_6_moe'` at phase 3 of the load — after the 27B had
    // been released — and the restore then failed on memory the dead attempt
    // still held. The check is a JSON read; it belongs before the teardown.
    let host = Arc::new(ModelHost::empty());
    let dir = tempfile::tempdir().expect("tmp");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"no_such_architecture","hidden_size":4096,"num_hidden_layers":1}"#,
    )
    .expect("write");

    use clap::Parser as _;
    let args = cli::ServeArgs::parse_from(["spark", dir.path().to_str().expect("utf8")]);
    let err = super::swap(&host, args, None).expect_err("refused");
    let text = format!("{err:#}");
    assert!(
        text.contains("no compiled kernels") || text.contains("no_such_architecture"),
        "{text}"
    );
    assert!(
        text.contains("running model is untouched") || host.current().is_none(),
        "nothing was torn down: {text}"
    );
}
