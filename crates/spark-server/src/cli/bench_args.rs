// SPDX-License-Identifier: AGPL-3.0-only

//! Arguments for `spark benchmark`.
//!
//! The same suite the dashboard runs, without a terminal — so a benchmark can
//! be scripted, run in CI, or driven over SSH on a headless box.

/// `spark benchmark <list|run|history>` — or `--pull-request-gate-check` on
/// its own, which needs no subcommand.
#[derive(clap::Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct BenchmarkArgs {
    /// Check the committed `.benchmarks/` records for THIS commit: every
    /// required gate must have a passing record. Prints what is missing or
    /// failing and exits non-zero when the branch is not fully gated.
    /// Runs without a subcommand (and without an endpoint).
    #[arg(long = "pull-request-gate-check")]
    pub pull_request_gate_check: bool,
    #[command(subcommand)]
    pub command: Option<BenchmarkCommand>,
}

#[derive(clap::Subcommand, Debug)]
pub enum BenchmarkCommand {
    /// List the suite, or one benchmark's parameter schema.
    List(ListArgs),
    /// Run one benchmark against a served endpoint.
    Run(RunArgs),
    /// Past runs, from `~/.atlas/runs`.
    History(HistoryArgs),
}

#[derive(clap::Args, Debug)]
pub struct ListArgs {
    /// Benchmark id. Omit for the whole suite.
    pub id: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    /// Benchmark id — `spark benchmark list` prints them.
    pub id: String,
    /// The endpoint to drive.
    ///
    /// This does NOT start a server — EXCEPT under `--pull-request-gate`, which
    /// serves the benchmark's own recipe on a free port and tears it down
    /// after. The two are mutually exclusive: a gate run has nowhere to send
    /// this, so passing it is rejected rather than quietly overridden.
    #[arg(
        long,
        default_value = "http://127.0.0.1:8888",
        conflicts_with = "pull_request_gate"
    )]
    pub url: String,
    /// The `model` field sent in every request.
    ///
    /// Required rather than defaulted: it is recorded with the run, and a
    /// result that cannot say what it measured is not worth keeping. Under
    /// `--pull-request-gate` it is supplied by the benchmark's recipe instead,
    /// and passing it is rejected rather than silently ignored — a flag that
    /// looks like it selects the model while the recipe actually does is the
    /// confusion this mode exists to remove.
    #[arg(
        long,
        required_unless_present = "pull_request_gate",
        conflicts_with = "pull_request_gate"
    )]
    pub model: Option<String>,

    /// Which box class the run is for, e.g. `gb10`.
    ///
    /// Only consulted under `--pull-request-gate`, to pick the baseline entry
    /// when a benchmark has thresholds for more than one box class. With a
    /// single entry it is inferred; with several, omitting it is an error
    /// rather than a guess.
    #[arg(long)]
    pub hardware: Option<String>,
    /// Override one parameter, e.g. `--param osl=8`. Repeatable.
    ///
    /// Anything not overridden takes the schema default and is still recorded.
    #[arg(long = "param", value_name = "KEY=VALUE", value_parser = parse_kv)]
    pub params: Vec<(String, String)>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
    /// How often to drain the run's channels, in milliseconds.
    #[arg(long, default_value_t = 250)]
    pub poll_ms: u64,
    /// Do not write the run to `~/.atlas/runs`.
    #[arg(long)]
    pub no_save: bool,
    /// Confirm a benchmark with side effects beyond load on the endpoint.
    ///
    /// Required for `agentic-webserver`, which executes model-authored shell.
    #[arg(long)]
    pub yes: bool,
    /// Print only the final report, not per-phase progress.
    #[arg(long)]
    pub quiet: bool,
    /// Exit 0 even when the gate verdict is FAIL.
    #[arg(long)]
    pub no_fail_on_verdict: bool,
    /// Do not ask the endpoint two known-answer questions before measuring.
    ///
    /// The probe only WARNS — it never refuses to start — so this is for
    /// skipping the two extra completions, not for silencing a veto.
    #[arg(long)]
    pub skip_coherence_probe: bool,
    /// Commit this run as a gate record under the repo's `.benchmarks/<id>/`.
    ///
    /// The record carries the metrics, verdict, hardware fingerprint, the
    /// exact command and the current commit sha, so the branch itself can
    /// answer "did this pass" — no `~/.atlas` state required.
    #[arg(long)]
    pub pull_request_gate: bool,
    /// Override one SERVE key from the benchmark's recipe, e.g.
    /// `--serve-override kv_cache_dtype=fp8`. Repeatable.
    ///
    /// Distinct from `--param`, which sets the BENCHMARK's own knobs
    /// (iterations, max_tokens). This one reaches the recipe that starts the
    /// server, so it is how you exercise a code path the recipe's pinned
    /// config never reaches — the case that motivated it: every gate recipe
    /// pins `kv_cache_dtype: bf16`, so a change to the fp8-KV attention
    /// kernel could not be measured by any gate at all. Five greens that
    /// never executed the changed code are worse than no run, because they
    /// read as evidence.
    ///
    /// Keys are recipe `defaults` keys, and `Recipe::argv` REFUSES one that
    /// is absent there — a typo fails loudly instead of silently measuring
    /// the unmodified config. `port` is rejected: the gate picks a free port
    /// and a second opinion about it would race the listener.
    ///
    /// ★ Every override is written into the gate record. A record whose
    /// numbers came from a config other than its recipe's must say so, or it
    /// is a plausible number attached to the wrong provenance — which is the
    /// exact failure this whole record format exists to prevent.
    #[arg(long = "serve-override", value_name = "KEY=VALUE")]
    pub serve_override: Vec<String>,
}

#[derive(clap::Args, Debug)]
pub struct HistoryArgs {
    /// Restrict to one benchmark id.
    #[arg(long)]
    pub id: Option<String>,
    /// Print the whole record for one run id.
    #[arg(long)]
    pub run: Option<String>,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Split `KEY=VALUE` on the **first** `=` only.
///
/// An `IntList` value is `isls=128,512` and a `Text` value may legitimately
/// contain `=`, so splitting on every separator would corrupt both.
fn parse_kv(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!(
            "expected KEY=VALUE, got {s:?} — e.g. --param osl=8 or --param isls=128,512"
        )),
    }
}

#[cfg(test)]
mod tests {
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
        let cli = Cli::try_parse_from(["spark", "benchmark", "--pull-request-gate-check"])
            .expect("parses");
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
}
