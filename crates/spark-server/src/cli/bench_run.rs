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
use std::collections::BTreeMap;

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
        BenchmarkCommand::Card(a) => card_cmd(a),
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
    Ok((sha, dirty))
}

#[cfg(test)]
#[path = "bench_provenance_tests.rs"]
mod provenance_tests;

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
    serve_overrides: BTreeMap<String, String>,
    sha_at_start: String,
    dirty_at_start: Vec<String>,
    // `--output-image` target plus its parsed `--output-image-args`.
    //
    // Threaded in rather than read from `RunArgs` here: this function is also
    // the gate path, and giving it the whole args struct would let a future edit
    // reach for a flag that has nothing to do with writing a record.
    card: Option<(String, BTreeMap<String, String>)>,
) -> Result<()> {
    // ★ An INCOMPLETE run must not become a gate record.
    //
    // A cancelled or failed run still produces a RunRecord -- it just has no
    // measurements in it. Committing that gives the branch a file that looks
    // like evidence and contains none; `check_record` then reports every
    // threshold as "missing from the record", blaming the baseline rather than
    // the run that never finished. Observed for real: a BFCL run killed at
    // 972/1004 left a committed record whose metrics were `{}`.
    if record.frame.status != atlas_plugin::RunStatus::Completed {
        bail!(
            "the run ended as {:?}, not Completed -- no gate record was written. \
             A record is evidence that a benchmark RAN; an interrupted one is not.",
            record.frame.status
        );
    }
    if record.frame.metrics.is_empty() {
        bail!(
            "the run produced no metrics -- no gate record was written. Every \
             threshold would read as \"missing from the record\", which blames the \
             baseline for a run that measured nothing."
        );
    }
    let root = repo_root()?;
    // ★ The sha is the one captured BEFORE the run, not the one HEAD happens to
    // point at now. A record exists to say "these numbers came from this
    // commit", and `bfcl-subset` takes ~3.5 hours: reading HEAD at write time
    // stamps whatever was committed while the benchmark was running. Observed
    // in practice -- a 4-hour run recorded a sha that was 14 commits newer than
    // the binary that produced it.
    let sha = sha_at_start;
    if let Ok(now) = gate::git_sha(&root)
        && now != sha
    {
        // Not fatal: the measurement is real and belongs to `sha`. But the
        // tree moved underneath it, so whoever reads this record needs to know
        // the working copy is no longer what was measured.
        eprintln!(
            "gate: HEAD moved during the run ({sha} -> {now}); the record is \
             stamped {sha}, the commit that was actually measured"
        );
    }
    let target = TargetEndpoint::new(url, model);
    let hardware = atlas_plugin::http::fetch_hardware(&target, gate::HARDWARE_TIMEOUT).await;
    let dirty = dirty_at_start;
    let gate_record =
        gate::GateRecord::from_run(record, hardware, sha, dirty, recipe, serve_overrides)?
            // What THIS binary's kernels were compiled from. Baked at build
            // time, so it describes the code that actually ran rather than the
            // tree as it stands now.
            .with_closure(atlas_kernels::TARGET_CLOSURES);
    let path = gate::write_record(&root, &gate_record)?;

    // Sign it, and say BOTH filenames. The operator commits what the terminal
    // names; if this printed only the .json they would leave the .sig untracked
    // and the gate would hard-fail on a record that is perfectly good.
    //
    // Signing lives here rather than inside `write_record` so the writer stays a
    // pure function of (root, record) for the ~7 unit tests that call it — none
    // of which should be minting keys in a real ~/.atlas.
    let store = atlas_plugin::artifacts::ArtifactStore::discover()?;
    let identity = gate::signing::load_or_create(store.root())?;
    let sig = gate::signing::sign_record(&identity, &path, &gate_record.git_sha)?;
    let fresh_signer = gate::signing::register(&root, &identity)?;
    eprintln!(
        "gate record written as {}\n                  and {}",
        path.display(),
        sig.display()
    );
    if let Some((target, card_args)) = &card {
        // After the record, deliberately: the card is rendered FROM the record,
        // so a card can never show a number the record does not.
        match write_card(&root, &gate_record, target, card_args) {
            Ok(card) => eprintln!("result card written as {}", card.display()),
            // A card is a nice-to-have. Failing the whole run because a template
            // was missing would throw away a benchmark that already succeeded,
            // and the record — the thing that matters — is already on disk.
            Err(e) => {
                eprintln!("gate: the run succeeded but the result card did not render: {e:#}")
            }
        }
    }
    if fresh_signer {
        // Once per machine, ever. The first record from a new box carries its
        // public key into the diff, which is where a human decides whether this
        // signer is one of ours. Every run after this is silent.
        eprintln!(
            "gate: this machine signed a record for the first time. Commit \
             {}/{}.pub alongside the record — it is how the gate learns to trust \
             records from this box.",
            gate::signing::REGISTRY_DIR,
            identity.fingerprint()
        );
    }
    // Loud, and at the point the operator is about to commit the file. The
    // record itself carries the verdict (`hardware_state.postcheck`), but a
    // number is quoted from a terminal long before anyone opens the JSON, and
    // the 2026-08-15 retraction happened because nothing said this out loud.
    if let Some(hw) = &gate_record.hardware_state
        && hw.invalidated()
    {
        eprintln!(
            "gate: ★ that record is marked INVALID — the box throttled while it was \
             measuring, so its SPEED numbers are not comparable and must not be quoted. \
             Concerns: {}",
            hw.concerns().join("; ")
        );
    }
    // Repeated at the end as well as the start: the start-of-run warning has
    // scrolled hours off the top of the terminal by now, and this one names the
    // file the reader is about to commit.
    if !gate_record.dirty_paths.is_empty() {
        eprintln!(
            "gate: that record is stamped {} but was measured from a tree with \
             {} uncommitted invalidation-set file(s); it records them, and \
             --pull-request-gate-check will reject it. Re-run from a clean tree.",
            gate_record.git_sha,
            gate_record.dirty_paths.len()
        );
    }
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
        let written = write_gate_record(
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

/// Where `--output-image` writes.
///
/// A bare NAME becomes `./<name>.svg`. Anything carrying a path separator or an
/// extension is taken literally. The rule is stated in the flag's help so a user
/// never has to discover it by experiment, and `.svg` is appended rather than
/// substituted — a card named `qwen3.8-27b` must not become `qwen3.svg`, the
/// same trap the record sidecars have.
fn card_output_path(target: &str) -> std::path::PathBuf {
    let looks_like_a_path = target.contains(std::path::MAIN_SEPARATOR)
        || target.contains('/')
        || std::path::Path::new(target)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("svg"));
    if looks_like_a_path {
        std::path::PathBuf::from(target)
    } else {
        std::path::PathBuf::from(format!("{target}.svg"))
    }
}

/// Render the shareable card for a finished run.
fn write_card(
    root: &std::path::Path,
    record: &atlas_plugin::gate::GateRecord,
    target: &str,
    card_args: &std::collections::BTreeMap<String, String>,
) -> Result<std::path::PathBuf> {
    let template_path = root.join("assets/cards/result-card.svg");
    let template = std::fs::read_to_string(&template_path)
        .with_context(|| format!("reading the card template at {}", template_path.display()))?;
    let svg = atlas_plugin::gate::card::render(&template, record, card_args);
    let out = card_output_path(target);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&out, svg).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

/// `spark benchmark card <record>` — a card from an already-measured result.
fn card_cmd(args: crate::cli::bench_args::CardArgs) -> Result<()> {
    let record_path = resolve_record(&args.record)?;
    let record = atlas_plugin::gate::read_record(&record_path)
        .with_context(|| format!("reading the record at {}", record_path.display()))?;
    let card_args = args
        .output_image_args
        .as_deref()
        .map(atlas_plugin::gate::card::parse_args)
        .transpose()
        .map_err(|e| anyhow::anyhow!("--output-image-args: {e}"))?
        .unwrap_or_default();
    // Default the name off the record so `benchmark card <path>` alone does
    // something useful: `<gate>-<sha>.svg`, which sorts and is unambiguous.
    let target = args
        .output_image
        .unwrap_or_else(|| format!("{}-{}", record.benchmark_id, record.git_sha));
    // Find the template by walking UP FROM THE RECORD, not from the cwd. A card
    // is regenerated from a path, often from outside the checkout, and the
    // template that belongs to a record is the one in the repo that holds it.
    // Falling back to `repo_root()` keeps `benchmark card x.json` working from
    // inside the tree when the record was handed in by a relative path.
    let root = template_root_for(&record_path).or_else(|_| repo_root())?;
    let out = write_card(&root, &record, &target, &card_args)?;
    println!("{}", out.display());
    Ok(())
}

/// The repo root that owns `record`, found by walking up to the card template.
fn template_root_for(record: &std::path::Path) -> Result<std::path::PathBuf> {
    let start = record
        .canonicalize()
        .unwrap_or_else(|_| record.to_path_buf());
    for dir in start.ancestors().skip(1) {
        if dir.join("assets/cards/result-card.svg").is_file() {
            return Ok(dir.to_path_buf());
        }
    }
    bail!(
        "no assets/cards/result-card.svg above {} — pass a record inside a checkout",
        record.display()
    )
}

/// A benchmark id or a record path -> a record path.
///
/// An existing file wins, so a benchmark that ever shares a name with a real
/// path still resolves the way the user pointed. Otherwise the argument is a
/// benchmark id and this takes the NEWEST committed record for it — which is
/// what "make a card of the run I just did" means in practice.
fn resolve_record(arg: &str) -> Result<std::path::PathBuf> {
    let direct = std::path::Path::new(arg);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }
    let root = repo_root().context(
        "not inside a checkout, so a benchmark id cannot be resolved — pass a record path",
    )?;
    let dir = root.join(".benchmarks").join(arg);
    if !dir.is_dir() {
        bail!(
            "no benchmark or record called `{arg}` ({} does not exist). \
             `spark benchmark list` prints the ids.",
            dir.display()
        );
    }
    // Newest by filename: records are `<date>-<sha>[-<variant>].json`, so a
    // lexical sort is chronological. Ties inside a day are broken by sha, which
    // is arbitrary but stable — and a card names its commit, so a reader can
    // always tell which one they got.
    let mut records: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "json")
                && p.file_name().is_some_and(|n| n != "BASELINE.json")
        })
        .collect();
    records.sort();
    records.pop().with_context(|| {
        format!("`{arg}` has no committed records yet — run it first, or pass a record path")
    })
}

/// Render a card for a benchmark id (or record path), for callers outside this
/// module — the TUI's History pane.
///
/// Shares `resolve_record` and `write_card` with `benchmark card` rather than
/// re-deriving either: two code paths that pick "the newest record" by different
/// rules would eventually disagree, and the disagreement would be a card
/// showing a different run than the row the operator selected.
pub fn render_card_for_benchmark(id: &str, output: Option<&str>) -> Result<std::path::PathBuf> {
    let record_path = resolve_record(id)?;
    let record = atlas_plugin::gate::read_record(&record_path)
        .with_context(|| format!("reading {}", record_path.display()))?;
    let target = output
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}-{}", record.benchmark_id, record.git_sha));
    let root = template_root_for(&record_path).or_else(|_| repo_root())?;
    write_card(&root, &record, &target, &Default::default())
}
