// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `spark benchmark` argument parsing. Moved verbatim from
//! `bench_args.rs` at the 500-line boundary (exact piecewise copy);
//! everything here drives the REAL `Cli` parser.

use super::*;
use clap::Parser as _;

use crate::cli::Cli;

fn run_args(argv: &[&str]) -> RunArgs {
    let cli = Cli::try_parse_from(argv).expect("parses");
    match cli.command {
        crate::cli::Command::Benchmark(b) => match b.command {
            Some(BenchmarkCommand::Run(r)) => r,
            other => panic!("wanted run, got {other:?}"),
        },
        other => panic!("wanted benchmark, got {other:?}"),
    }
}

#[test]
fn a_run_takes_repeated_param_overrides() {
    let a = run_args(&[
        "spark",
        "benchmark",
        "run",
        "concurrency-sweep",
        "--model",
        "m",
        "--param",
        "osl=8",
        "--param",
        "isls=128,512",
    ]);
    assert_eq!(a.id, "concurrency-sweep");
    assert_eq!(a.model.as_deref(), Some("m"));
    assert_eq!(
        a.params,
        vec![
            ("osl".to_string(), "8".to_string()),
            ("isls".to_string(), "128,512".to_string()),
        ]
    );
    assert_eq!(
        a.url, "http://127.0.0.1:8888",
        "defaults to the local serve"
    );
}

#[test]
fn a_value_may_contain_an_equals_sign() {
    // Split on the FIRST `=` only: a Text parameter can legitimately hold
    // one, and splitting on every separator would truncate it.
    let (k, v) = parse_kv("prompt=a=b").expect("parses");
    assert_eq!((k.as_str(), v.as_str()), ("prompt", "a=b"));
}

#[test]
fn a_param_without_a_separator_is_rejected_with_an_example() {
    let err = parse_kv("osl8").expect_err("rejected");
    assert!(err.contains("KEY=VALUE"), "{err}");
    assert!(err.contains("--param osl=8"), "shows the shape: {err}");
    assert!(parse_kv("=8").is_err(), "an empty key is not a key");
}

#[test]
fn the_model_is_required_unless_the_gate_supplies_it() {
    // A run whose record cannot say what it measured is not worth keeping,
    // so this is a parse error rather than a silent default.
    assert!(
        Cli::try_parse_from(["spark", "benchmark", "run", "concurrency-sweep"]).is_err(),
        "--model must be supplied when driving an existing endpoint"
    );
    // Under the gate the recipe supplies it, so demanding it here would be
    // demanding a value the caller has no say over.
    assert!(
        Cli::try_parse_from([
            "spark",
            "benchmark",
            "run",
            "bfcl-subset",
            "--pull-request-gate",
        ])
        .is_ok(),
        "the gate resolves the model from the benchmark's recipe"
    );
}

#[test]
fn the_gate_refuses_a_hand_picked_endpoint() {
    // Silently ignoring these would leave the operator believing they
    // selected the target when the recipe did — the precise confusion this
    // mode exists to remove. Reject instead.
    for extra in [["--model", "m"], ["--url", "http://127.0.0.1:9999"]] {
        let mut argv = vec!["spark", "benchmark", "run", "bfcl-subset"];
        argv.extend_from_slice(&extra);
        argv.push("--pull-request-gate");
        assert!(
            Cli::try_parse_from(&argv).is_err(),
            "{extra:?} must conflict with --pull-request-gate"
        );
    }
}

fn bench_args(argv: &[&str]) -> BenchmarkArgs {
    let cli = Cli::try_parse_from(argv).expect("parses");
    match cli.command {
        crate::cli::Command::Benchmark(b) => b,
        other => panic!("wanted benchmark, got {other:?}"),
    }
}

#[test]
fn a_pr_without_the_gate_check_is_refused() {
    // clap's `requires = "pull_request_gate_check"` never fired here: the
    // SetTrue target's implicit `false` default satisfies clap 4's
    // requirement check, so BOTH of these parse clean (asserted via the
    // expect in bench_args) and enforcement lives in reject_orphan_pr.
    // Without it, the no-subcommand form below reached the
    // `expect("clap enforces a subcommand here")` panic in dispatch.
    for argv in [
        &["spark", "benchmark", "--pr", "5", "list"][..],
        &["spark", "benchmark", "--pr", "5"][..],
    ] {
        let a = bench_args(argv);
        let err = a.reject_orphan_pr().expect_err("refused");
        assert!(err.contains("--pull-request-gate-check"), "{err}");
        assert!(err.contains("--pr"), "names the orphan flag: {err}");
    }
}

#[test]
fn a_pr_with_the_gate_check_is_accepted() {
    // The acceptance direction: the legitimate pairing must keep parsing
    // AND pass the explicit check — this is what catches an over-eager
    // rejection breaking the CI invocation.
    let a = bench_args(&[
        "spark",
        "benchmark",
        "--pull-request-gate-check",
        "--pr",
        "513",
    ]);
    assert!(a.pull_request_gate_check);
    assert_eq!(a.pr, Some(513));
    a.reject_orphan_pr().expect("valid pairing accepted");
    // And the flag stays optional: the gate check without --pr is the
    // normal local/push-build invocation.
    let a = bench_args(&["spark", "benchmark", "--pull-request-gate-check"]);
    assert_eq!(a.pr, None);
    a.reject_orphan_pr().expect("absent --pr is fine");
}

#[test]
fn list_and_history_take_an_optional_id() {
    assert!(Cli::try_parse_from(["spark", "benchmark", "list"]).is_ok());
    assert!(Cli::try_parse_from(["spark", "benchmark", "list", "concurrency-sweep"]).is_ok());
    assert!(Cli::try_parse_from(["spark", "benchmark", "history"]).is_ok());
    assert!(Cli::try_parse_from(["spark", "benchmark", "history", "--run", "run-1"]).is_ok());
}

#[test]
fn a_run_takes_the_pull_request_gate_flag() {
    // No --model: the gate resolves it from the recipe, and passing one is
    // a conflict (see the_gate_refuses_a_hand_picked_endpoint).
    let a = run_args(&[
        "spark",
        "benchmark",
        "run",
        "agentic-webserver",
        "--yes",
        "--pull-request-gate",
    ]);
    assert!(a.pull_request_gate);
    assert!(a.yes);
    assert!(a.model.is_none());
    assert!(a.hardware.is_none(), "inferred when the baseline has one");
}

#[test]
fn the_gate_takes_an_explicit_checkpoint_variant() {
    let a = run_args(&[
        "spark",
        "benchmark",
        "run",
        "agentic-webserver",
        "--checkpoint",
        "unsloth/Qwen3.8-27B-NVFP4",
        "--yes",
        "--pull-request-gate",
    ]);
    assert_eq!(a.checkpoint.as_deref(), Some("unsloth/Qwen3.8-27B-NVFP4"));
}

#[test]
fn a_checkpoint_without_the_gate_is_refused() {
    // Outside the gate the serve config is whatever the operator started,
    // so a variant selector would be a flag that visibly does nothing —
    // the same confusion --model/--url vs the gate exists to remove.
    // Enforced by the explicit check, not clap `requires` — the SetTrue
    // flag's implicit default satisfies clap's version silently.
    let a = run_args(&[
        "spark",
        "benchmark",
        "run",
        "agentic-webserver",
        "--model",
        "m",
        "--checkpoint",
        "unsloth/Qwen3.8-27B-NVFP4",
    ]);
    let err = a.reject_orphan_checkpoint().expect_err("refused");
    assert!(err.contains("--pull-request-gate"), "{err}");
    assert!(err.contains("--model"), "names the alternative: {err}");

    let gated = run_args(&[
        "spark",
        "benchmark",
        "run",
        "agentic-webserver",
        "--checkpoint",
        "unsloth/Qwen3.8-27B-NVFP4",
        "--yes",
        "--pull-request-gate",
    ]);
    assert!(gated.reject_orphan_checkpoint().is_ok());
}

#[test]
fn the_gate_takes_an_explicit_hardware_class() {
    let a = run_args(&[
        "spark",
        "benchmark",
        "run",
        "ttft-warm-gate",
        "--hardware",
        "gb10",
        "--pull-request-gate",
    ]);
    assert_eq!(a.hardware.as_deref(), Some("gb10"));
}

#[test]
fn gate_check_runs_without_a_subcommand() {
    let cli =
        Cli::try_parse_from(["spark", "benchmark", "--pull-request-gate-check"]).expect("parses");
    match cli.command {
        crate::cli::Command::Benchmark(b) => {
            assert!(b.pull_request_gate_check);
            assert!(b.command.is_none());
        }
        other => panic!("wanted benchmark, got {other:?}"),
    }
}

#[test]
fn bare_benchmark_without_gate_check_still_needs_a_subcommand() {
    // `spark benchmark` alone must not silently do nothing.
    assert!(Cli::try_parse_from(["spark", "benchmark"]).is_err());
}
