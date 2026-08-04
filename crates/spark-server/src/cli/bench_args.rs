// SPDX-License-Identifier: AGPL-3.0-only

//! Arguments for `spark benchmark`.
//!
//! The same suite the dashboard runs, without a terminal — so a benchmark can
//! be scripted, run in CI, or driven over SSH on a headless box.

/// `spark benchmark <list|run|history>`
#[derive(clap::Args, Debug)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub command: BenchmarkCommand,
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
    /// The endpoint to drive. This does NOT start a server.
    #[arg(long, default_value = "http://127.0.0.1:8888")]
    pub url: String,
    /// The `model` field sent in every request.
    ///
    /// Required rather than defaulted: it is recorded with the run, and a
    /// result that cannot say what it measured is not worth keeping.
    #[arg(long)]
    pub model: String,
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
                BenchmarkCommand::Run(r) => r,
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
        assert_eq!(a.model, "m");
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
    fn the_model_is_required() {
        // A run whose record cannot say what it measured is not worth keeping,
        // so this is a parse error rather than a silent default.
        assert!(
            Cli::try_parse_from(["spark", "benchmark", "run", "concurrency-sweep"]).is_err(),
            "--model must be supplied"
        );
    }

    #[test]
    fn list_and_history_take_an_optional_id() {
        assert!(Cli::try_parse_from(["spark", "benchmark", "list"]).is_ok());
        assert!(Cli::try_parse_from(["spark", "benchmark", "list", "concurrency-sweep"]).is_ok());
        assert!(Cli::try_parse_from(["spark", "benchmark", "history"]).is_ok());
        assert!(Cli::try_parse_from(["spark", "benchmark", "history", "--run", "run-1"]).is_ok());
    }
}
