// SPDX-License-Identifier: AGPL-3.0-only

//! Writing, signing and card-rendering a finished run, split out of
//! `bench_run.rs` at the 500-LoC cap. Exact piecewise move — no logic changed.
//!
//! This is the one member that turns a `RunRecord` into the committed artefacts
//! a merge depends on, so it is the natural seam.

use super::bench_card::write_card;
use super::bench_run::repo_root;
use anyhow::{Result, bail};
use atlas_plugin::TargetEndpoint;
use atlas_plugin::gate;
use std::collections::BTreeMap;

/// Commit this run as a gate record under the repo's `.benchmarks/<id>/`.
///
/// The hardware fingerprint is fetched from the endpoint that did the work —
/// not probed locally — so the record describes the box that actually served
/// the model. A write failure aborts the command with a clear error: the
/// point of the flag is the record, so a run that did not produce one must
/// not report success.
pub(crate) async fn write_gate_record(
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
