// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch for `spark benchmark`.
//!
//! This subcommand drives an endpoint that is already serving; it never starts
//! a model and never touches the GPU. Everything below is a thin shell around
//! `atlas_plugin::headless`, which the dashboard shares.

use anyhow::{Result, bail};
use atlas_plugin::headless::{HeadlessOptions, RunRequest, SilentReporter, run_blocking};
use atlas_plugin::{
    ArtifactStore, BenchmarkDescriptor, BenchmarkExecutor, ParamValues, TargetEndpoint, history,
    registry,
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
    match args.command {
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

    let store = store()?;
    let executor = BenchmarkExecutor::new(tokio::runtime::Handle::current(), store);
    let request = RunRequest {
        descriptor,
        values,
        target: TargetEndpoint::new(&args.url, &args.model),
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
