// SPDX-License-Identifier: AGPL-3.0-only

//! `--pull-request-gate-check`: does this commit have a passing record for
//! every required gate? Pure reads over `.benchmarks/` plus git ancestry —
//! no endpoint, no GPU, fast enough for every PR in CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::record::{GateBaseline, GateRecord, read_baseline, read_record};
use super::{REQUIRED_GATES, gate_dir};

pub use super::scoring::{Comparison, check_record, compare};

/// One required bench's standing in the committed tree.
#[derive(Debug)]
pub enum GateStatus {
    /// The newest covering record passes the baseline.
    Pass,
    /// The newest covering record exists but fails: the record's own verdict
    /// and each baseline breach.
    Fail(Vec<String>),
    /// No covering record exists, or the newest one is unreadable or never
    /// completed.
    Missing(String),
}

/// The newest-first list of record files in one benchmark's directory, ordered
/// by each record's own `recorded_at`. `BASELINE.json` is not a record.
///
/// ★ **The filename is not a clock.** A record is named
/// `YYYY-MM-DD-<sha>.json`, so a lexical sort orders by DATE and then by SHA —
/// and a sha is random. Two records cut on the same UTC day therefore sorted by
/// which hex digit happened to come first, which is exactly the situation a
/// re-run produces: measure, commit a fix, measure again, both records dated
/// today. The gate takes the first covering record as the branch's current
/// word, so under the old order a FAIL measured after a PASS was silently
/// discarded whenever its sha sorted lower — the gate passing on a superseded
/// result. It fails the other way just as easily, and neither is detectable
/// after the fact.
///
/// `recorded_at` is written by [`super::record::GateRecord::from_run`] from the
/// run itself, and it is the same number the filename's date is derived from,
/// so the two agree by construction and only the within-day tie changes. An
/// unreadable record sorts last (it can never be selected anyway) but stays in
/// the list, so a directory of nothing but corrupt records still reports
/// "unreadable" rather than "no records committed".
pub fn records_newest_first(root: &Path, benchmark_id: &str) -> Vec<PathBuf> {
    let dir = gate_dir(root, benchmark_id);
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    candidates.retain(|p| {
        p.extension().is_some_and(|e| e == "json")
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy() != "BASELINE.json")
    });
    let mut keyed: Vec<(u64, PathBuf)> = candidates
        .into_iter()
        .map(|p| (read_record(&p).map(|r| r.recorded_at).unwrap_or(0), p))
        .collect();
    // Newest first, and the filename breaks a tie so the order is total and
    // reproducible rather than dependent on readdir.
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    keyed.into_iter().map(|(_, p)| p).collect()
}

/// Whether a record measured at `record_sha` still stands for `head`.
///
/// Same commit always covers itself. An ancestor covers `head` while nothing
/// the run measures changed in between — a diff touching any of
/// [`super::coverage::PERF_PATHS`] invalidates every earlier record, because the binary and the
/// prompts it renders are no longer the recorded ones. A record can never be
/// written AT `head` (committing it moves head), so this ancestry rule is what
/// makes "gated at the current commit" achievable at all.
pub fn record_covers(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
) -> bool {
    invalidating_paths(root, head, record_sha, gate).is_some_and(|p| p.is_empty())
}

/// Whether `record` still describes `sha`, path boundary first, closure second.
///
/// Two rungs, in order, and the second can only ever narrow the first:
///
/// 1. **The path boundary** ([`record_covers`]). Nothing in the invalidation
///    set changed ⇒ covered, no further question.
/// 2. **The closure hash** ([`super::closure::excuses`]). Everything that did
///    change is inside `kernels/`, and every target those paths can affect
///    still compiles to byte-identical device code ⇒ covered.
///
/// Rung 2 exists because `kernels/<hw>/common/` holds the majority of kernels
/// and every model inherits from it, so treating one shared edit as "re-test
/// all 28 targets" costs more GPU time than anyone will pay — and a gate people
/// route around is worse than a slower one. It never widens coverage: a path
/// outside `kernels/`, an unattested target, or an uncomputable hash all leave
/// the record invalidated exactly as before.
fn record_still_stands(
    root: &Path,
    sha: &str,
    record: &GateRecord,
    gate: &super::coverage::GateCoverage,
) -> bool {
    match invalidating_paths(root, sha, &record.git_sha, gate) {
        // Not an ancestor, or git failed: unchanged fail-closed doctrine.
        None => false,
        Some(paths) if paths.is_empty() => true,
        Some(paths) => super::closure::excuses(root, &paths, &record.closure),
    }
}

/// The changed paths that invalidate `gate` between two commits.
///
/// `None` means the question could not be answered — git failed, or one of the
/// two commits is not in this clone. Every such case is treated as "not
/// covered" by the caller, keeping the fail-closed doctrine: a gate check that
/// cannot see the trees must never read as a pass.
///
/// # This deliberately does NOT require ancestry
///
/// It used to. `merge-base --is-ancestor record_sha head` gated the diff, and
/// that was wrong in a way that took main down: **Atlas squash-merges.** A
/// record is written on a PR branch, against a commit on that branch; the
/// squash lands a brand-new commit on main with a different sha and no parent
/// link to the branch. Every record the PR paid GPU hours for stops being an
/// ancestor of anything the instant it merges.
///
/// It did exactly that. `.benchmarks/*/2026-08-09-b0be4ba0e6.json` are five
/// real passing records for #389 — `b0be4ba0e` being the branch's merge of
/// #417 — and after #389 squash-landed as `dd2ac46d5` the gate reported
/// "not an ancestor of this commit" for all five. Main went red, and every PR
/// opened afterwards inherited it and demanded 5 fresh GPU legs to fix a
/// typo.
///
/// Ancestry was never what the check needed. `git diff A B` compares TREES; it
/// is defined for any two commits and needs no history relationship. The
/// question a gate record answers is "was the perf-relevant code the same when
/// this was measured?", and the diff answers exactly that. Ancestry only added
/// an assumption about the shape of history — one this repo's merge strategy
/// violates by design.
///
/// The obvious worry — "then a record from an unrelated branch could cover
/// main" — is answered by the diff itself. An unrelated branch differs on the
/// perf paths and is rejected. If it does NOT differ on them, it measured the
/// same code, and the record is valid; that is the whole content-not-ancestry
/// doctrine, and it is why an identical squash lands covered.
///
/// The one thing ancestry incidentally caught was a missing commit (a shallow
/// clone). `git diff` fails outright there, so that case still returns `None`.
/// The gate job checks out with `fetch-depth: 0`.
///
/// The diff is taken with NO pathspec and filtered in Rust. Two reasons, both
/// practical: the filter is then unit-testable without a git fixture, and
/// git's exclude-pathspec precedence rules are subtle enough that expressing
/// per-gate exclusions in them would move the policy somewhere nobody reviews.
pub fn invalidating_paths(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused(root, head, path)
    })
}

fn invalidating_paths_with(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    mut is_excused: impl FnMut(&str) -> bool,
) -> Option<Vec<String>> {
    if head == record_sha {
        return Some(Vec::new());
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", record_sha, head])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .filter(|p| super::coverage::invalidates(gate, p))
            // ★ A one-time content-pinned amnesty: a surviving path whose blob
            // at `head` is exactly the grant's pinned content is excused,
            // loudly. Content-pinned, so any later edit to the file changes
            // the OID and invalidates as before. See `amnesty.rs` for the
            // grant, the fail-closed rule, and the removal condition.
            .filter(|p| {
                if is_excused(p) {
                    tracing::warn!(
                        "amnesty: {p} would re-open {} but its content at {head} is the \
                         pinned one-time grant; excused (see gate/amnesty.rs)",
                        gate.id
                    );
                    return false;
                }
                true
            })
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
pub(crate) fn invalidating_paths_with_amnesty(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    table: &[super::amnesty::AmnestyEntry],
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused_by(root, head, path, table)
    })
}

/// The full gate verdict for `sha`: every required bench, in order.
pub fn check_gates(root: &Path, sha: &str) -> BTreeMap<String, GateStatus> {
    let mut out = BTreeMap::new();
    for id in REQUIRED_GATES {
        out.insert((*id).to_string(), check_one(root, id, sha));
    }
    out
}

/// Does this record actually belong to the gate whose directory it sits in?
///
/// Records were located by DIRECTORY and their own `benchmark_id` was never
/// read back. Nothing stopped one gate's record from satisfying another's, and
/// two of the required gates make that a one-command forgery:
/// `ttft-warm-gate` and `ttft-cold-gate` share a checkpoint, a hardware key and
/// their metric names (`median_ms`, `p90_ms`). On `main` today the committed
/// WARM record reads 1562.58 / 4478.42 against the COLD ceilings of
/// 1728.27 / 4809.76 — so `cp` one file into the other directory turns the cold
/// gate green with no cold leg ever run.
///
/// That is the worst possible pair to be able to confuse: cold-TTFT is the only
/// leg that sees a cold-LOAD regression, and #389 — the change this gate was
/// built alongside — is "GPU-transpose quantized weights at cold load".
///
/// It also needs no malice. Both records are produced minutes apart in one
/// session with near-identical filenames.
///
/// Mismatches are SKIPPED, not failed: a stray file in a directory should leave
/// the gate reading "no covering record" (which is true and actionable), not
/// manufacture a hard failure from someone else's passing run.
fn record_is_for(record: &GateRecord, benchmark_id: &str, path: &Path) -> bool {
    if record.benchmark_id == benchmark_id {
        return true;
    }
    tracing::warn!(
        "ignoring {}: it is a `{}` record sitting in the `{benchmark_id}` directory",
        path.display(),
        record.benchmark_id,
    );
    false
}

/// Is this record a run of the gate's REQUIRED subject — the checkpoint its
/// baseline marks `default = true` for the record's box class?
///
/// A benchmark with model variants commits records for ALL of them, in one
/// directory. Each is scored against its own variant's thresholds
/// (`check_record` resolves by the record's `target_model`), so a non-default
/// record can legitimately PASS — but a pass on the dense 27B is not evidence
/// for the required gate, whose declared subject is the default checkpoint.
/// Without this filter, a newer dense record would quietly become "the
/// branch's current word" on a 35B gate: the worst outcome this feature can
/// produce, a plausible green attached to the wrong subject.
///
/// A record from a box class the baseline does not know passes through: it has
/// no declared default to differ from, and `check_record` already hard-fails
/// it by name ("no baseline for hardware …"), which is the honest verdict.
fn record_is_required_subject(
    baseline: &GateBaseline,
    record: &GateRecord,
    benchmark_id: &str,
    path: &Path,
) -> bool {
    let hardware = record.hardware.gate_key();
    let Some(hw) = baseline.hardware.get(&hardware) else {
        return true;
    };
    if hw.default == record.target_model {
        return true;
    }
    tracing::warn!(
        "ignoring {} for the required {benchmark_id} gate: it measured the {} variant, \
         and the gate's declared subject on {hardware} is {}",
        path.display(),
        record.target_model,
        hw.default,
    );
    false
}

pub(super) fn check_one(root: &Path, benchmark_id: &str, sha: &str) -> GateStatus {
    // A GROUP may be satisfied EITHER by a whole-draw record under its own id —
    // the way every gate worked before the shard split, and the way every
    // record committed on main today was produced — OR by all of its members.
    //
    // Falling back rather than replacing is not politeness, it is the
    // difference between a transition and an outage. `bfcl-subset` is a
    // REQUIRED context: if becoming a group invalidated the whole-draw records
    // already on main, every open PR would go red the moment this landed and
    // stay red until someone ran a sharded campaign. Six tests in this file
    // caught exactly that by writing a whole-draw record and expecting a
    // verdict.
    // The group path is taken only once sharding has actually STARTED — i.e.
    // some member has a record on disk. Otherwise this falls through to the
    // ordinary gate, which knows how to say "no covering record" and, crucially,
    // WHICH perf-path files invalidated the last one. Reporting "your four
    // shards are missing" to someone whose whole-draw record was invalidated by
    // a kernel change would replace a 20-second fix with a bisect.
    if let Some(group) = super::group::find(benchmark_id)
        && !has_covering_record(root, benchmark_id, sha)
        && group
            .members
            .iter()
            .any(|m| !records_newest_first(root, m).is_empty())
    {
        return check_group(root, group, sha);
    }
    let Some(gate) = super::coverage::find(benchmark_id) else {
        // Unreachable through `check_gates`, which iterates the coverage table
        // itself, and a test pins that every required id resolves. Refusing
        // beats defaulting to "no exclusions": a silent fallback here would be
        // a second, undeclared coverage policy.
        return GateStatus::Missing(format!("{benchmark_id} has no coverage entry"));
    };
    let paths = records_newest_first(root, benchmark_id);
    if paths.is_empty() {
        return GateStatus::Missing("no gate records committed".into());
    }
    let baseline = match read_baseline(root, benchmark_id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };
    // The newest record that still stands for `sha`. A record measured at an
    // ancestor covers head while no perf-path file changed in between; a
    // record whose commit is unrelated, or was invalidated since, is skipped
    // rather than failed — the branch's current word is the newest one still
    // valid, and an old clean record is better than none.
    // The PATH travels with the record: the signature lives in a sidecar beside
    // it, and re-deriving that path from the record would be wrong — the variant
    // filename depends on the baseline, not on the record alone.
    let mut covered: Option<(GateRecord, std::path::PathBuf)> = None;
    for path in &paths {
        if let Ok(record) = read_record(path)
            && record_is_for(&record, benchmark_id, path)
            && record_is_required_subject(&baseline, &record, benchmark_id, path)
            && record_still_stands(root, sha, &record, gate)
        {
            covered = Some((record, path.clone()));
            break;
        }
    }
    let Some((record, record_path)) = covered else {
        let newest = read_record(&paths[0]).ok();
        return GateStatus::Missing(match newest {
            // ★ Name the files that invalidated it. "does not cover this
            // commit" tells an author a gate is open but not what re-opened
            // it, which turns a 20-second fix into a bisect.
            //
            // `None` here is the fail-closed arm of `invalidating_paths`: git
            // could not diff the two trees, which in practice means the record's
            // commit is not in this clone (a shallow fetch). Say that, rather
            // than reporting an empty path list as if nothing had changed.
            Some(newest_record) => {
                if newest_record.benchmark_id != benchmark_id {
                    return GateStatus::Missing(format!(
                        "latest record belongs to {}, not {benchmark_id} ({})",
                        newest_record.benchmark_id,
                        paths[0].file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
                // A newest record that is another VARIANT's is not stale — it
                // is off-subject, and saying "build inputs do not match" would
                // send the reader diffing commits instead of reading the
                // model name.
                if !record_is_required_subject(&baseline, &newest_record, benchmark_id, &paths[0]) {
                    let hardware = newest_record.hardware.gate_key();
                    let subject = baseline
                        .hardware
                        .get(&hardware)
                        .map(|hw| hw.default.clone())
                        .unwrap_or_default();
                    return GateStatus::Missing(format!(
                        "latest record measured the {} variant; the required subject on \
                         {hardware} is {subject}, which has no covering record",
                        newest_record.target_model
                    ));
                }
                let newest = newest_record.git_sha.clone();
                let Some(why) = invalidating_paths(root, sha, &newest, gate) else {
                    return GateStatus::Missing(format!(
                        "latest record is for {newest} ({}) — git cannot diff that commit \
                         against this one; is it in this clone? (the gate job needs \
                         `fetch-depth: 0`)",
                        paths[0].file_name().unwrap_or_default().to_string_lossy()
                    ));
                };
                let because = if why.is_empty() {
                    // Reachable only if a record was skipped for a reason other
                    // than its path diff — today, a closure-hash mismatch that
                    // `excuses` refused.
                    "its recorded build inputs do not match this commit".to_string()
                } else {
                    // Naming the TARGETS as well as the files is the difference
                    // between "a kernel changed" and "this is why you owe a
                    // 3.5-hour run": a shared edit that re-opens one model
                    // reads very differently from one that re-opens all 22.
                    let targets =
                        super::closure::changed_targets(root, &why, &newest_record.closure);
                    match targets.len() {
                        0 => format!("invalidated by {}", super::check_fmt::summarize_paths(&why)),
                        n => format!(
                            "invalidated by {} — device code changed for {n} target(s): {}",
                            super::check_fmt::summarize_paths(&why),
                            super::check_fmt::summarize_paths(&targets)
                        ),
                    }
                };
                format!(
                    "latest record is for {newest} ({}) — {because}",
                    paths[0].file_name().unwrap_or_default().to_string_lossy()
                )
            }
            None => "latest record is unreadable".to_string(),
        });
    };
    if record.frame_status_failed() {
        return GateStatus::Fail(vec![format!(
            "the run itself failed: {}",
            record.verdict_reason
        )]);
    }
    let mut problems = Vec::new();
    // ★ A record measured from a dirty tree does not describe its own sha.
    //
    // `record_covers` above proved that nothing in the invalidation set changed
    // between the record's commit and head. That proof is worthless if the
    // binary already differed from the record's commit when the run started —
    // the diff was never committed, so no ancestry walk can ever see it. Fail
    // rather than skip: the record's numbers are real, but they belong to no
    // commit, and the only thing that makes the file true again is a re-run on
    // a clean tree. Records written before this field existed carry an empty
    // vector and are unaffected.
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
    if !record.verdict_passes() {
        problems.push(format!(
            "run verdict is not PASS: {}",
            record.verdict_reason
        ));
    }
    if let Some(breaches) = check_record(&record, &baseline) {
        problems.extend(breaches);
    }
    // ★ FAIL, not skip — see `signing::verify_record`, which states why, and
    // why records before the cutover are exempt.
    if let Err(why) =
        super::signing::verify_record(root, &record_path, &record.git_sha, record.recorded_at)
    {
        problems.push(format!("{why}"));
    }
    if problems.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail(problems)
    }
}

/// The exit code, as a function of the VERDICTS ALONE.
///
/// ★ This signature is the advisory boundary, and it is deliberately narrow.
///
/// The intent half — [`super::required::RequiredReport`], the escalation
/// preview, the classifier's opinion — is about to start being reported next to
/// these verdicts. Everything reported next to a verdict eventually gets
/// consulted by something. Making the exit code a function that CANNOT SEE the
/// advisory data means the separation is enforced by the type checker rather
/// than by whoever edits the printing loop next.
///
/// `atlas-governance`'s own doctrine puts it plainly: the ledger is "advisory,
/// permanently — adding a ledger read would make [the gate] depend on a file
/// any job can append to". Flipping that later is a deliberate act that has to
/// widen THIS signature, which is exactly the review moment it deserves.
pub fn exit_code(statuses: &BTreeMap<String, GateStatus>) -> i32 {
    let open = statuses
        .values()
        .filter(|s| !matches!(s, GateStatus::Pass))
        .count();
    i32::from(open > 0)
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod check_tests;

/// A benchmark group: every member must have a covering record at this commit,
/// and their combined tallies must clear the GROUP's thresholds.
///
/// Reuses `records_newest_first`, `record_still_stands` and `check_record`
/// unchanged — a group changes where the metrics come FROM, not how a gate is
/// judged. What it adds is the all-or-nothing rule: three members of four is
/// not 75% measured, it is an aggregate over a sample set the group's
/// thresholds were never drawn against.
fn check_group(root: &Path, group: &'static super::group::BenchmarkGroup, sha: &str) -> GateStatus {
    use crate::benchmarks::bfcl::aggregate;

    let baseline = match read_baseline(root, group.id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };

    let mut shards: Vec<BTreeMap<String, aggregate::Tally>> = Vec::new();
    let mut members: Vec<super::group::MemberRecord> = Vec::new();
    let mut newest: Option<GateRecord> = None;
    let mut missing: Vec<&str> = Vec::new();

    for member in group.members {
        // Same selection rule as a plain gate: the newest record that is FOR
        // this benchmark, is the required subject, and still stands at `sha`.
        let Some(member_gate) =
            super::coverage::find(member).or_else(|| super::coverage::find(group.id))
        else {
            return GateStatus::Missing(format!("{member} has no coverage entry"));
        };
        let mut found: Option<GateRecord> = None;
        for path in &records_newest_first(root, member) {
            if let Ok(r) = read_record(path)
                && record_is_for(&r, member, path)
                && record_still_stands(root, sha, &r, member_gate)
            {
                found = Some(r);
                break;
            }
        }
        let Some(record) = found else {
            missing.push(member);
            continue;
        };
        // A member that ran but carries no per-subset tallies cannot be folded
        // in. Counting it as an empty contribution would shrink the union and
        // score the group over fewer samples than the draw.
        let Some(t) = aggregate::tallies_from_metrics(&record.metrics) else {
            return GateStatus::Missing(format!(
                "{member} has a covering record but no per-subset tallies — it was \
                 measured by a binary older than the shard split, so the group \
                 cannot be aggregated. Re-run {member} at this commit."
            ));
        };
        members.push(super::group::MemberRecord {
            id: (*member).to_string(),
            git_sha: record.git_sha.clone(),
        });
        shards.push(t);
        newest = Some(record);
    }

    if !missing.is_empty() {
        return GateStatus::Missing(
            super::group::GroupFault::Missing {
                group: group.id,
                missing,
            }
            .to_string(),
        );
    }
    if let Err(fault) = super::group::composition_ok(group, &members) {
        return GateStatus::Fail(vec![fault.to_string()]);
    }

    let agg = aggregate::aggregate(&aggregate::union(&shards));
    let Some(mut record) = newest else {
        return GateStatus::Missing(format!("{} has no members", group.id));
    };
    // Judge the AGGREGATE, carrying one member's provenance (checkpoint, serve
    // overrides, hardware) — composition_ok has already established they agree
    // on the commit, and `check_record` needs a record shape, not a new one.
    record.benchmark_id = group.id.to_string();
    record
        .metrics
        .insert("overall_accuracy".into(), agg.overall_accuracy);
    record.metrics.insert(
        "normalized_single_turn_score".into(),
        agg.normalized_single_turn_score,
    );
    record
        .metrics
        .insert("samples".into(), agg.total_samples as f64);

    match super::scoring::check_record(&record, &baseline) {
        None => GateStatus::Pass,
        Some(breaches) => GateStatus::Fail(breaches),
    }
}

/// Is there a covering record under this id itself? Used to decide whether a
/// group still has a whole-draw measurement, before falling back to its shards.
///
/// Deliberately the same three predicates `check_one` uses, so "covering" means
/// one thing in this file.
fn has_covering_record(root: &Path, benchmark_id: &str, sha: &str) -> bool {
    let Some(gate) = super::coverage::find(benchmark_id) else {
        return false;
    };
    let Ok(baseline) = read_baseline(root, benchmark_id) else {
        return false;
    };
    records_newest_first(root, benchmark_id).iter().any(|path| {
        read_record(path).is_ok_and(|r| {
            record_is_for(&r, benchmark_id, path)
                && record_is_required_subject(&baseline, &r, benchmark_id, path)
                && record_still_stands(root, sha, &r, gate)
        })
    })
}
