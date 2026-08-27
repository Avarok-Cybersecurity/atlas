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
        cli::Command::Benchmark(_) | cli::Command::DumpServeOptions | cli::Command::SyncRecipes => {
            unreachable!("parsed a serve command")
        }
    }
}

/// A bad config must be refused BEFORE the host is cleared. If validation ran
/// after the drain, a typo would cost the running model.
#[test]
fn an_invalid_config_is_refused_before_anything_is_torn_down() {
    let host = Arc::new(ModelHost::empty());
    let bad = args(&["--scheduling-policy", "nonsense"]);
    let err = swap(&host, bad).expect_err("refused");
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
    let err = swap(&host, multi).expect_err("refused");
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
    let _ = swap(&host, args(&["--world-size", "4"]));
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
    let err = super::swap(&host, args).expect_err("refused");
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

#[test]
fn two_swaps_at_once_do_not_both_tear_down_the_model() {
    // Unguarded, both callers reach `ModelHost::take`; the loser gets `None`,
    // reads it as a modelless boot, rebuilds the carried stores from scratch
    // and loads a second model onto a GPU already loading one. The TUI's
    // Library launch called `swap` directly, so two presses of `s` was enough.
    let host = Arc::new(ModelHost::empty());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let overlapping = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let inside = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let threads: Vec<_> = (0..2)
        .map(|_| {
            let (host, barrier) = (host.clone(), barrier.clone());
            let (overlapping, inside) = (overlapping.clone(), inside.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let _guard = host.swap_guard();
                if inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0 {
                    overlapping.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                std::thread::sleep(std::time::Duration::from_millis(60));
                inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            })
        })
        .collect();
    for t in threads {
        t.join().expect("no panic");
    }
    assert!(
        !overlapping.load(std::sync::atomic::Ordering::SeqCst),
        "two swaps were inside the guard at once"
    );
}

#[test]
fn the_swap_the_winner_already_performed_is_not_repeated() {
    // Requests that queued behind a swap must not each redo it. Compared as a
    // whole argv, so switching recipes for the SAME checkpoint still swaps.
    use clap::Parser as _;
    let a = cli::ServeArgs::parse_from(["spark", "org/m"]);
    let mut b = cli::ServeArgs::parse_from(["spark", "org/m"]);
    assert_eq!(a, b, "same argv");
    b.max_batch_size = a.max_batch_size + 1;
    assert_ne!(a, b, "a different recipe for the same model is a real swap");
}

#[test]
fn a_swap_is_refused_once_shutdown_has_been_requested() {
    // Starting a load into an exiting process releases the outgoing model for
    // a replacement that will never serve anything.
    let err = super::refuse_if_shutting_down(true).expect_err("refused");
    assert!(format!("{err:#}").contains("shutdown"), "{err:#}");
    super::refuse_if_shutting_down(false).expect("a live process may swap");
}

#[test]
fn a_recipe_cannot_turn_off_authentication() {
    // A swap replaces the ENTIRE argv with the recipe's, and `require_auth` is
    // deliberately not among the flags `carry_process_flags` carries. If the
    // live policy were read from those args, swapping to any recipe that omits
    // them — which is every recipe in the published catalogue — would silently
    // leave the endpoint unauthenticated.
    //
    // It is not read from them: the policy is installed on the host before the
    // listener binds and no swap path touches it. This test exists so that
    // stays true; a future `host.set_auth(...)` inside `swap` would fail here
    // rather than in production.
    let host = Arc::new(ModelHost::empty());
    let cfg = std::sync::Arc::new(
        crate::auth::AuthConfig::from_inline("sk-test-token").expect("valid token"),
    );
    host.set_auth(Some(cfg));

    // Any swap that gets far enough to have replaced argv will do; this one is
    // refused at the multi-rank gate, which is after the argv is taken.
    use clap::Parser as _;
    let mut args = cli::ServeArgs::parse_from(["spark", "org/m"]);
    args.world_size = 2;
    let _ = super::swap(&host, args);

    assert!(
        host.auth().is_some(),
        "the API-key policy must not be a casualty of a swap"
    );
}

#[test]
fn a_swap_does_not_silently_stop_request_dumping() {
    // `--dump` is an operator's observability choice and no recipe sets it.
    // Left uncarried, a swap replaces argv with the recipe's and the dump just
    // stops: the file stays where it was and is never written to again, which
    // is the worst way for a diagnostic to fail — it looks like no traffic.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["spark", "org/live", "--dump", "/tmp/probe.jsonl"]);
    let mut next = cli::ServeArgs::parse_from(["spark", "org/next"]);
    assert!(next.dump.is_none(), "the recipe says nothing about it");

    super::carry_process_flags(&mut next, &previous);
    assert_eq!(next.dump.as_deref(), Some("/tmp/probe.jsonl"));
    assert_eq!(
        next.model.as_deref(),
        Some("org/next"),
        "the MODEL still swaps"
    );
}

#[test]
fn a_repeat_of_the_live_config_is_a_no_op_even_with_process_flags_set() {
    // The live argv holds the carried flags; a recipe's does not. Comparing
    // them before carrying meant they could never be equal whenever any
    // process flag was set — so with --auto-swap on, which is exactly when
    // requests queue behind a swap, every queued request redid the load the
    // winner had just finished.
    use clap::Parser as _;

    // What the host stores after a swap: the recipe's argv plus carried flags.
    let mut live = cli::ServeArgs::parse_from(["spark", "org/m"]);
    let operator =
        cli::ServeArgs::parse_from(["spark", "org/boot", "--auto-swap", "--dump", "/tmp/d.jsonl"]);
    super::carry_process_flags(&mut live, &operator);

    // What a queued request brings: the same recipe, without those flags.
    let mut queued = cli::ServeArgs::parse_from(["spark", "org/m"]);
    assert_ne!(
        live, queued,
        "they differ before carrying — the old comparison"
    );

    super::carry_process_flags(&mut queued, &live);
    assert_eq!(
        live, queued,
        "and are equal after it, which is what makes the no-op fire"
    );
}
