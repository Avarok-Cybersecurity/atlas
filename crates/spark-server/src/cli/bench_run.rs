// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch for `spark benchmark`.
//!
//! This subcommand drives an endpoint that is already serving; it never starts
//! a model and never touches the GPU. Everything below is a thin shell around
//! `atlas_plugin::headless`, which the dashboard shares.

use anyhow::{Context, Result, bail};
use atlas_plugin::headless::{HeadlessOptions, RunRequest, SilentReporter, run_blocking};
use atlas_plugin::{
    ArtifactStore, BenchmarkDescriptor, BenchmarkExecutor, ParamValues, TargetEndpoint, gate,
    history, registry,
};

use super::bench_args::{BenchmarkArgs, BenchmarkCommand, HistoryArgs, OutputFormat, RunArgs};
use super::bench_print;

/// Look up a benchmark, naming the alternatives when it is not one.
pub fn find(id: &str) -> Result<&'static BenchmarkDescriptor> {
    registry::find(id).ok_or_else(|| {
        let known: Vec<&str> = registry::all().iter().map(|d| d.id).collect();
        anyhow::anyhow!(
            "unknown benchmark {id:?} — the suite is: {}",
            known.join(", ")
        )
    })
}

pub async fn dispatch(args: BenchmarkArgs) -> Result<()> {
    if args.pull_request_gate_check {
        let code = gate_check_cmd()?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    let command = args.command.expect("clap enforces a subcommand here");
    match command {
        BenchmarkCommand::List(a) => match a.id {
            Some(id) => bench_print::print_schema(&id, a.format),
            None => bench_print::print_suite(a.format),
        },
        BenchmarkCommand::History(a) => history_cmd(a),
        BenchmarkCommand::Run(a) => {
            let code = run(a).await?;
            // `run` reports its own outcome; the exit code is the machine-
            // readable half and must survive returning through main.
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
    }
}

/// The repo root this checkout lives in — where `.benchmarks/` sits.
pub(super) fn repo_root() -> Result<std::path::PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse")?;
    if !out.status.success() {
        bail!("not inside a git checkout — the gate records live in the repo's .benchmarks/");
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        bail!("git rev-parse --show-toplevel printed nothing");
    }
    Ok(std::path::PathBuf::from(path))
}

/// `--pull-request-gate-check`: does THIS commit have a passing record for
/// every required gate? Prints a line per bench and exits 1 until they all
/// pass. Pure filesystem reads — fast enough to run on every PR in CI.
fn gate_check_cmd() -> Result<i32> {
    let root = repo_root()?;
    let sha = gate::git_sha(&root)?;
    let gates = gate::check_gates(&root, &sha);
    println!("gate check for {sha} ({})", root.display());
    let mut open = Vec::new();
    for id in gate::REQUIRED_GATES {
        let status = &gates[id];
        match status {
            gate::GateStatus::Pass => println!("  PASS  {id}"),
            gate::GateStatus::Fail(reasons) => {
                println!("  FAIL  {id}");
                for reason in reasons {
                    println!("        - {reason}");
                }
                open.push(id);
            }
            gate::GateStatus::Missing(reason) => {
                println!("  NONE  {id} — {reason}");
                open.push(id);
            }
        }
    }
    if open.is_empty() {
        println!("all {} required gates pass", gate::REQUIRED_GATES.len());
        Ok(0)
    } else {
        println!(
            "{} bench(es) still need a passing gate record: {}",
            open.len(),
            open.join(", ")
        );
        Ok(1)
    }
}

/// Commit this run as a gate record under the repo's `.benchmarks/<id>/`.
///
/// The hardware fingerprint is fetched from the endpoint that did the work —
/// not probed locally — so the record describes the box that actually served
/// the model. A write failure aborts the command with a clear error: the
/// point of the flag is the record, so a run that did not produce one must
/// not report success.
async fn write_gate_record(
    record: &atlas_plugin::RunRecord,
    url: &str,
    model: &str,
    recipe: Option<String>,
) -> Result<()> {
    let root = repo_root()?;
    let sha = gate::git_sha(&root)?;
    let target = TargetEndpoint::new(url, model);
    let hardware = atlas_plugin::http::fetch_hardware(&target, gate::HARDWARE_TIMEOUT).await;
    let gate_record = gate::GateRecord::from_run(record, hardware, sha, recipe)?;
    let path = gate::write_record(&root, &gate_record)?;
    eprintln!("gate record written as {}", path.display());
    Ok(())
}

fn store() -> Result<ArtifactStore> {
    ArtifactStore::discover()
}

fn history_cmd(args: HistoryArgs) -> Result<()> {
    let store = store()?;
    if let Some(run_id) = &args.run {
        let Some(record) = history::find(&store, run_id) else {
            bail!("no run {run_id:?} under {}", store.root().display());
        };
        return bench_print::print_record(&record, args.format);
    }
    let mut records = match &args.id {
        Some(id) => {
            find(id)?; // reject a typo rather than reporting an empty history
            history::load(&store, id)
        }
        None => history::load_all(&store),
    };
    records.truncate(args.limit);
    bench_print::print_history(&records, args.format)
}

async fn run(args: RunArgs) -> Result<i32> {
    let descriptor = find(&args.id)?;
    if descriptor.needs_confirmation && !args.yes {
        bail!(
            "{} has side effects beyond load on the endpoint — it executes \
             model-authored shell in a sandbox. Pass --yes to accept that.",
            descriptor.id
        );
    }

    let specs = descriptor.build().parameters();
    let pairs = args
        .params
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect::<Vec<_>>();
    let values = ParamValues::from_overrides(&specs, pairs)?;

    // With --pull-request-gate the suite provisions its own server from the
    // benchmark's recipe, so the record describes a config nobody typed. Without
    // it, --url/--model drive an existing endpoint exactly as before.
    let served = if args.pull_request_gate {
        Some(super::bench_selfstart::serve_for(&args.id, args.hardware.as_deref()).await?)
    } else {
        None
    };
    let target = match &served {
        Some(s) => s.target.clone(),
        None => TargetEndpoint::new(&args.url, args.model.as_deref().unwrap_or_default()),
    };

    let store = store()?;
    let executor = BenchmarkExecutor::new(tokio::runtime::Handle::current(), store);
    let request = RunRequest {
        descriptor,
        values,
        target: target.clone(),
        options: HeadlessOptions {
            poll: std::time::Duration::from_millis(args.poll_ms),
            save: !args.no_save,
            source: atlas_plugin::RunSource::Cli,
            atlas_version: super::ATLAS_VERSION.to_string(),
            coherence: if args.skip_coherence_probe {
                atlas_plugin::CoherencePolicy::Skip
            } else {
                atlas_plugin::CoherencePolicy::Probe
            },
        },
    };

    // Ctrl-C reuses the server's signal handling rather than installing a
    // second one. Disarm the startup escape first: that hatch exists so a
    // Ctrl-C during a multi-minute model load still exits, and this is not a
    // model load.
    crate::tui::shutdown::disarm_startup_escape();
    crate::tui::shutdown::install_signal_listeners();

    let quiet = args.quiet;
    let format = args.format;
    // `run_blocking` sleeps its thread, so it must not hold a runtime worker.
    let outcome = tokio::task::spawn_blocking(move || {
        let mut reporter = bench_print::StdoutReporter::new(quiet);
        let mut silent = SilentReporter;
        let reporter: &mut dyn atlas_plugin::headless::RunReporter = if format == OutputFormat::Json
        {
            &mut silent // JSON on stdout must not be interleaved with progress
        } else {
            &mut reporter
        };
        run_blocking(
            &executor,
            request,
            reporter,
            &crate::tui::shutdown::requested,
        )
    })
    .await??;

    if args.pull_request_gate {
        // Order is load-bearing. `write_gate_record` fetches the hardware
        // fingerprint FROM the endpoint, and that fetch degrades to
        // `Hardware::unknown()` on every failure path without returning an
        // error — so tearing the server down first would commit a record that
        // names no box and still exit 0. Write first, tear down second, and
        // tear down even when the write fails.
        let recipe = served.as_ref().map(|s| s.recipe_id.clone());
        let written =
            write_gate_record(&outcome.record, &target.base_url, &target.model, recipe).await;
        if let Some(s) = served {
            s.shutdown().await;
        }
        written?;
    } else if let Some(s) = served {
        s.shutdown().await;
    }

    match args.format {
        OutputFormat::Json => bench_print::print_record(&outcome.record, OutputFormat::Json)?,
        OutputFormat::Text => {
            println!();
            bench_print::print_frame(&outcome.record.frame);
            if let Some(path) = &outcome.saved_to {
                eprintln!("\nrecorded as {}", path.display());
            }
        }
    }

    let code = outcome.exit_code();
    // A failed gate is still a completed measurement; a caller collecting
    // numbers may not want it to fail the script.
    if code == 2 && args.no_fail_on_verdict {
        return Ok(0);
    }
    Ok(code)
}
