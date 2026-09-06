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
    if let Err(msg) = args.reject_orphan_pr() {
        bail!("{msg}");
    }
    if args.pull_request_gate_check {
        let code = super::bench_gate_check::gate_check_cmd(args.pr)?;
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
        BenchmarkCommand::Card(a) => super::bench_card::card_cmd(a),
        BenchmarkCommand::Aggregate(a) => {
            // Exits with the code so a script can gate on "is this group
            // complete", the same shape `Run` uses below.
            let code = super::bench_aggregate::aggregate_cmd(a)?;
            std::process::exit(code);
        }
        BenchmarkCommand::Run(a) => {
            let code = run(a).await?;
            // `run` reports its own outcome; the exit code is the machine-
            // readable half and must survive returning through main.
            //
            // ★ Nothing that must happen may be placed AFTER this call. Both
            // exits here — the `exit` below and the `?` above — skip whatever
            // follows, `exit` skipping destructors too. Teardown of a
            // self-started server therefore lives inside `run` (and, for the
            // paths that miss it, in `SelfServed::drop`), never here.
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
    }
}

/// The repo root this checkout lives in — where `.benchmarks/` sits.
pub(crate) fn repo_root() -> Result<std::path::PathBuf> {
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

/// The commit and the uncommitted invalidation-set files, both read BEFORE the
/// run — the two halves of "which sources produced this binary".
///
/// ★ The dirty list is captured HERE, at the start, and warned about HERE,
/// because that is the only moment the warning can still save anything: a
/// `bfcl-subset` gate takes ~3.5 hours, and an operator told at the end that
/// the binary never matched the commit has already spent the afternoon. A
/// A failure to read the dirt aborts before the model is loaded. A record with
/// an empty dirty list asserts that the tree was clean; it must not also mean
/// that git could not answer the question.
fn capture_provenance() -> Result<(String, Vec<String>)> {
    let root = repo_root()?;
    capture_provenance_at(&root)
}

fn capture_provenance_at(root: &std::path::Path) -> Result<(String, Vec<String>)> {
    let sha = gate::git_sha(root)?;
    let dirty = gate::dirty_perf_paths(root)
        .context("reading the working tree state before the gate run")?;
    if !dirty.is_empty() {
        eprintln!(
            "gate: WARNING — {} uncommitted file(s) that change what a gate \
             measures are in this tree, so the record will be stamped {sha} \
             but the binary is not {sha}:",
            dirty.len()
        );
        for path in &dirty {
            eprintln!("gate:   {path}");
        }
        eprintln!(
            "gate: the record will disclose this and the gate check will \
             reject it. Commit (or stash) and rebuild first."
        );
    }
    warn_if_signer_is_not_committed(root);
    Ok((sha, dirty))
}

/// Say, BEFORE the GPU-hours are spent, which identity this box will sign with
/// and whether that identity is committed.
///
/// `signing::register` writes `<fp>.pub` into `.github/record-signers/` on
/// first use and `bench_record` prints a one-time notice — but both happen
/// AFTER the run, into whatever log the operator redirected it to. On
/// 2026-09-05 a campaign was split across three boxes to save wall-clock;
/// each box minted its own identity (the key is per-ATLAS_HOME, not per
/// machine — one box here holds two), and the notice scrolled past in three
/// separate log files. The mistake only surfaced at CI, where
/// `.github/workflows/ci.yml`'s "One PR, one commit, one signer" step rejects
/// a record set spanning fingerprints outright. Seven gates had to be
/// re-measured.
///
/// So this warns at the point the operator can still act on it. It never
/// fails the run: a first record from a genuinely new box is legitimate, and
/// refusing it would make bringing up a box impossible.
fn warn_if_signer_is_not_committed(root: &std::path::Path) {
    let Ok(store) = ArtifactStore::discover() else {
        return;
    };
    let Ok(identity) = gate::signing::load_or_create(store.root()) else {
        return;
    };
    let fp = identity.fingerprint();
    match gate::signing::committed_signers(root) {
        Ok(committed) => {
            if let Some(msg) = gate::signing::signer_notice(&committed, fp) {
                eprintln!("{msg}");
            }
        }
        // Cannot answer: say so rather than imply the signer is fine.
        Err(e) => eprintln!("gate: NOTE — could not read .github/record-signers/: {e:#}"),
    }
}

#[cfg(test)]
#[path = "bench_provenance_tests.rs"]
mod provenance_tests;

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
    if let Err(msg) = args.reject_orphan_checkpoint() {
        bail!("{msg}");
    }
    if let Err(msg) = args.reject_orphan_image_args() {
        bail!("{msg}");
    }
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
    let mut values = ParamValues::from_overrides(&specs, pairs)?;

    // With --pull-request-gate the suite provisions its own server from the
    // benchmark's recipe, so the record describes a config nobody typed. Without
    // it, --url/--model drive an existing endpoint exactly as before.
    // Captured BEFORE the run so a multi-hour benchmark records the commit it
    // actually measured rather than whatever lands on HEAD meanwhile — and,
    // with it, the uncommitted files that make that commit an incomplete
    // answer, warned about now while aborting is still cheap.
    let provenance = if args.pull_request_gate {
        Some(capture_provenance()?)
    } else {
        None
    };
    // Discovered BEFORE the server exists. It is fallible, and every fallible
    // step that happens with a model loaded is one more path that has to tear
    // it down; the ones that can happen first, should. (`SelfServed::drop`
    // covers the ones that cannot.)
    let store = store()?;
    let served = if args.pull_request_gate {
        Some(
            super::bench_selfstart::serve_for(
                &args.id,
                args.hardware.as_deref(),
                args.checkpoint.as_deref(),
                super::bench_resolve::parse_serve_overrides(&args.serve_override)?,
            )
            .await?,
        )
    } else {
        None
    };
    let target = match &served {
        Some(s) => s.target.clone(),
        None => TargetEndpoint::new(&args.url, args.model.as_deref().unwrap_or_default()),
    };
    // The served VARIANT defines any baseline-coupled parameters the run's own
    // verdict reads (an explicit --param still wins) — see
    // `apply_threshold_params` for the precedence and why. Its
    // `[benchmarks.param_overrides]` pins go first: they shape the INSTRUMENT
    // (which ladder, which budget), the threshold params shape the VERDICT,
    // and the two are refused from naming the same key.
    if let Some(s) = &served {
        for (param, value) in super::bench_resolve::apply_param_overrides(
            descriptor,
            &specs,
            &mut values,
            &s.baseline_entry,
            &args.params,
        )? {
            eprintln!(
                "gate: {param} = {value} pinned by the {} variant's baseline \
                 [benchmarks.param_overrides] (not the schema default)",
                target.model
            );
        }
        for (param, bound) in super::bench_resolve::apply_threshold_params(
            descriptor,
            &specs,
            &mut values,
            &s.baseline_entry,
            &args.params,
        )? {
            eprintln!(
                "gate: {param} = {bound} from the {} variant's baseline (not the schema default)",
                target.model
            );
        }
    }

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
    .await;

    // Tear the self-started server down before propagating a failure.
    //
    // ★ This used to be `.await??`, which returns early — so a benchmark that
    // ERRORED skipped teardown entirely and left the model resident on the GPU.
    // The process happens to exit soon after today, which is what hid it, but
    // "the OS cleans up" is not a teardown: it depends on the caller exiting,
    // and `--no-fail-on-verdict` and any future in-process caller break that
    // assumption. A failed run must not leave a 100 GB model loaded.
    //
    // Success still writes the gate record FIRST and tears down second (see
    // below); only the failure paths shut down here.
    let outcome = match outcome {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            if let Some(s) = served {
                s.shutdown().await;
            }
            return Err(e);
        }
        Err(join) => {
            if let Some(s) = served {
                s.shutdown().await;
            }
            return Err(join.into());
        }
    };

    if args.pull_request_gate {
        // Order is load-bearing. `write_gate_record` fetches the hardware
        // fingerprint FROM the endpoint, and that fetch degrades to
        // `Hardware::unknown()` on every failure path without returning an
        // error — so tearing the server down first would commit a record that
        // names no box and still exit 0. Write first, tear down second, and
        // tear down even when the write fails.
        let recipe = served.as_ref().map(|s| s.recipe_id.clone());
        let serve_overrides = served
            .as_ref()
            .map(|s| s.overrides.clone())
            .unwrap_or_default();
        let (sha_at_start, dirty_at_start) = provenance.unwrap_or_default();
        let written = super::bench_record::write_gate_record(
            &outcome.record,
            &target.base_url,
            &target.model,
            recipe,
            serve_overrides,
            sha_at_start,
            dirty_at_start,
            match &args.output_image {
                Some(target) => Some((
                    target.clone(),
                    args.output_image_args
                        .as_deref()
                        .map(atlas_plugin::gate::card::parse_args)
                        .transpose()
                        .map_err(|e| anyhow::anyhow!("--output-image-args: {e}"))?
                        .unwrap_or_default(),
                )),
                None => None,
            },
        )
        .await;
        if let Some(s) = served {
            s.shutdown().await;
        }
        written?;
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
