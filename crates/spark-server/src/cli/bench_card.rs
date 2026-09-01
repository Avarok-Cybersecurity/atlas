// SPDX-License-Identifier: AGPL-3.0-only

//! Result cards, split out of `bench_run.rs` at the 500-LoC cap. Exact piecewise
//! move — no logic changed, the same seam `bench_state_history.rs` was cut on.
//!
//! These four are the only members that render an SVG rather than drive a
//! benchmark, so they form the natural boundary.

use anyhow::{Context, Result, bail};

use super::bench_run::repo_root;

/// Where `--output-image` writes.
///
/// A bare NAME becomes `./<name>.svg`. Anything carrying a path separator or an
/// extension is taken literally. The rule is stated in the flag's help so a user
/// never has to discover it by experiment, and `.svg` is appended rather than
/// substituted — a card named `qwen3.8-27b` must not become `qwen3.svg`, the
/// same trap the record sidecars have.
pub(crate) fn card_output_path(target: &str) -> std::path::PathBuf {
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
pub(crate) fn write_card(
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
pub(crate) fn card_cmd(args: crate::cli::bench_args::CardArgs) -> Result<()> {
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
