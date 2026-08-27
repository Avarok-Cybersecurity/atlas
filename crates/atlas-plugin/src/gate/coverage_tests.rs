// SPDX-License-Identifier: AGPL-3.0-only

//! Coverage tests: which committed record still speaks for HEAD.
//!
//! Split from `tests.rs` for the 500-LoC cap. These are the ones that
//! need a real git repo, because `record_covers` walks ancestry: a record
//! can never be written AT head (committing moves head), so an ancestor
//! must be able to speak for it — until a perf path changes between them.

use super::tests::{tempdir, *};
use super::*;
use crate::result::{RunStatus, Verdict};
use std::collections::BTreeMap;

/// A gate that excludes nothing.
///
/// These tests are about the boundary and git ancestry, not about any one
/// gate's exclusions, so they use the strictest possible coverage: everything
/// on the boundary invalidates. Using a real gate here would couple ancestry
/// tests to whichever exclusions that gate happens to carry today.
pub(super) fn any_gate() -> super::coverage::GateCoverage {
    super::coverage::GateCoverage {
        id: "test-strictest",
        excludes: &[],
    }
}

pub(super) mod scratch_repo {
    use std::path::Path;
    use std::process::Command;

    fn git(root: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {args:?}: {:?}", out);
    }

    pub fn init(root: &Path) {
        git(root, &["init", "-q"]);
        std::fs::write(root.join("README.md"), "first").unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", "first"]);
    }

    pub fn head(root: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--short=10", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Create and switch to `name`.
    pub fn branch(root: &Path, name: &str) {
        git(root, &["checkout", "-q", "-b", name]);
    }

    /// Switch back to whatever branch `git init` created. NOT hardcoded to
    /// `master`/`main`: `init.defaultBranch` is user config, so a hardcoded
    /// name passes on one machine and fails on the next.
    pub fn checkout_default(root: &Path, name: &str) {
        git(root, &["checkout", "-q", name]);
    }

    /// The current branch name.
    pub fn current_branch(root: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Whether `a` is an ancestor of `b`. Used only to ASSERT that a fixture
    /// reproduces the squash shape — the production check no longer asks.
    pub fn is_ancestor(root: &Path, a: &str, b: &str) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["merge-base", "--is-ancestor", a, b])
            .output()
            .is_ok_and(|o| o.status.success())
    }

    pub fn commit(root: &Path, file: &str, contents: &str, message: &str) {
        let path = root.join(file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", message]);
    }
}

#[test]
fn an_ancestor_record_covers_head_until_a_perf_path_changes() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);

    // Committed BEFORE sha_a. `write_baseline` now scaffolds a small kernel
    // tree (HARDWARE.toml, MODEL.toml, BENCH.toml) because that is where the
    // thresholds live, and `kernels/` is a boundary path — left uncommitted,
    // the next `git add .` would sweep the scaffolding into the commit under
    // test and this would be measuring the fixture rather than the policy.
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let sha_a = scratch_repo::head(root);

    for id in REQUIRED_GATES {
        plant(root, id, &sha_a, 1_785_891_382, "PASS");
    }

    // A docs-only commit afterwards: every record still covers head.
    scratch_repo::commit(root, "docs/notes.md", "hello", "docs only");
    let sha_b = scratch_repo::head(root);
    assert!(
        record_covers(root, &sha_b, &sha_a, &any_gate()),
        "docs-only diff is inert"
    );
    let gates = check_gates(root, &sha_b);
    for id in REQUIRED_GATES {
        assert!(
            matches!(gates[id], GateStatus::Pass),
            "{id}: {:?}",
            gates[id]
        );
    }

    // A change under crates/ invalidates every earlier record.
    scratch_repo::commit(root, "crates/x.rs", "// code", "touch a crate");
    let sha_c = scratch_repo::head(root);
    assert!(
        !record_covers(root, &sha_c, &sha_a, &any_gate()),
        "crates/ diff invalidates"
    );
    let gates = check_gates(root, &sha_c);
    for id in REQUIRED_GATES {
        assert!(
            matches!(&gates[id], GateStatus::Missing(m) if m.contains(&sha_a)),
            "{id}: {:?}",
            gates[id]
        );
    }
}

/// A record measured LATER wins, even when its sha sorts lower.
///
/// ★ Regression, and the dangerous direction of it. Records are named
/// `YYYY-MM-DD-<sha>.json` and were ordered by that name, so two records cut on
/// the same UTC day were ranked by a random hex string. `check_one` takes the
/// first covering record as the branch's current word — so a FAIL measured
/// after a PASS was discarded whenever its sha happened to sort lower, and the
/// gate reported PASS on a result that had already been superseded. Nothing
/// downstream could see it: both files are valid, both cover head, and the
/// chosen one is a genuine passing run.
///
/// The shas here are real commits, so which one sorts higher is not ours to
/// choose — the roles are assigned from the observed order instead, which is
/// what makes this deterministic rather than a coin flip that passes half the
/// time.
#[test]
fn the_newest_record_is_the_one_measured_last_not_the_higher_sha() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    let sha_a = scratch_repo::head(root);
    scratch_repo::commit(root, "docs/a.md", "a", "docs only");
    let sha_b = scratch_repo::head(root);
    scratch_repo::commit(root, "docs/b.md", "b", "docs only");
    let head = scratch_repo::head(root);

    std::fs::create_dir_all(gate_dir(root, "bfcl-subset")).unwrap();
    write_baseline(root, "bfcl-subset", &bfcl_baseline());

    // Same UTC day for both, so only the within-day tie is under test. The
    // PASS is the EARLIER measurement and gets the lexically GREATER sha —
    // the arrangement a filename sort gets backwards.
    let day = 1_785_891_382;
    let (earlier_pass, later_fail) = if sha_a > sha_b {
        (&sha_a, &sha_b)
    } else {
        (&sha_b, &sha_a)
    };
    plant(root, "bfcl-subset", earlier_pass, day, "PASS");
    plant(root, "bfcl-subset", later_fail, day + 3_600, "FAIL");

    let ordered = records_newest_first(root, "bfcl-subset");
    assert!(
        ordered[0].to_string_lossy().contains(later_fail.as_str()),
        "the record measured last must come first, got {ordered:?}"
    );
    // Both records cover head — only docs changed — so the ordering alone
    // decides the verdict.
    assert!(record_covers(root, &head, earlier_pass, &any_gate()));
    assert!(record_covers(root, &head, later_fail, &any_gate()));
    match &check_gates(root, &head)["bfcl-subset"] {
        GateStatus::Fail(reasons) => assert_eq!(reasons, &["run verdict is not PASS: ok"]),
        other => panic!("a superseded PASS must not speak for the branch, got {other:?}"),
    }
}

/// Every path the invalidation rule names must exist in THIS repo.
///
/// A guard entry that matches nothing is silently inert: `git diff -- gone/`
/// is always empty, so the rule keeps returning "covered" while looking
/// thorough. A rename is exactly how that happens, and it produces no error
/// anywhere.
#[test]
fn every_invalidating_path_exists_in_this_repo() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    for path in PERF_PATHS {
        assert!(
            root.join(path).exists(),
            "{path} is in PERF_PATHS but not in the tree — the guard matches \
             nothing and invalidates nothing"
        );
    }
}

/// A chat template or a toolchain bump invalidates an earlier record.
///
/// Neither is under `crates/`, and both change the number a re-run would
/// produce: `jinja-templates/<model_type>.jinja` is read from the repo root at
/// serve time and OVERRIDES the checkpoint's own chat template, so it decides
/// the exact bytes of every prompt; `rust-toolchain.toml` decides which
/// compiler built the binary.
#[test]
fn a_prompt_template_or_toolchain_change_invalidates_an_earlier_record() {
    for (file, contents) in [
        ("jinja-templates/qwen3_5_moe.jinja", "{{ messages }}"),
        ("rust-toolchain.toml", "[toolchain]\nchannel = \"1.94.0\"\n"),
    ] {
        let dir = tempdir::Dir::new();
        let root = dir.path();
        scratch_repo::init(root);
        let before = scratch_repo::head(root);
        scratch_repo::commit(root, "docs/n.md", "inert", "docs only");
        assert!(
            record_covers(root, &scratch_repo::head(root), &before, &any_gate()),
            "a docs commit must stay inert"
        );
        scratch_repo::commit(root, file, contents, "change what gets measured");
        assert!(
            !record_covers(root, &scratch_repo::head(root), &before, &any_gate()),
            "{file} changes what a run measures, so an earlier record cannot speak for head"
        );
    }
}

/// A baseline entry with no thresholds must be refused, not passed.
///
/// The comparison loop is a no-op over an empty metric map, so the weakest
/// possible baseline would otherwise produce the strongest possible verdict:
/// Pass, unconditionally, whatever the run measured.
#[test]
fn a_baseline_entry_with_no_thresholds_is_not_a_pass() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline_for(MODEL, BTreeMap::new())).expect("refused");
    assert_eq!(
        problems,
        [format!(
            "the baseline entry for {MODEL} on {TEST_HW} declares no thresholds — \
             there is nothing here for this run to have passed"
        )]
    );
}

#[test]
fn a_failed_frame_fails_the_gate_even_with_passing_numbers() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    std::fs::create_dir_all(gate_dir(root, "bfcl-subset")).unwrap();
    write_baseline(root, "bfcl-subset", &bfcl_baseline());
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let mut record = run_record(metrics.clone(), Verdict::fail("scoring crashed"));
    record.frame = frame(RunStatus::Failed, metrics, Verdict::fail("scoring crashed"));
    let mut gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    gate.recorded_at = 1_785_891_382;
    write_record(root, &gate).unwrap();

    let gates = check_gates(root, SHA);
    match &gates["bfcl-subset"] {
        GateStatus::Fail(reasons) => {
            assert_eq!(reasons, &["the run itself failed: scoring crashed"])
        }
        other => panic!("wanted Fail, got {other:?}"),
    }
}

#[test]
fn the_summary_names_the_model_the_numbers_and_the_verdict() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        gate.summary,
        format!("{MODEL} · overall_accuracy=87.74 · Pass: ok")
    );
}

#[test]
fn required_gates_are_registered_benchmarks() {
    for id in REQUIRED_GATES {
        assert!(
            crate::registry::find(id).is_some(),
            "{id} is not registered"
        );
    }
}

/// A two-sided bound is a RANGE, and an equal pair is an EXACT pin.
///
/// ★ Regression: only the one-sided arms existed, so `{"min": 995, "max": 995}`
/// fell through to "malformed bound". Fail-closed, so nothing scored leniently
/// — but the gate then failed every run and blamed the baseline's syntax
/// instead of the measurement, which makes an exact pin unusable. The BFCL draw
/// size is pinned exactly this way.
#[test]
fn an_exact_pin_passes_only_on_the_pinned_value() {
    use crate::gate::{Bound, Comparison, compare};

    let pin = Bound {
        min: Some(1004.0),
        max: Some(1004.0),
        noise: None,
    };
    assert!(matches!(compare("samples", 1004.0, &pin), Comparison::Pass));

    // The exact failure this pin exists to catch: the echolp draw silently
    // becoming 972 because a subset floor defaulted to 0.
    let Comparison::Fail(msg) = compare("samples", 972.0, &pin) else {
        panic!("a draw of 972 against a pin of 1004 must FAIL, not pass or skip");
    };
    assert_eq!(
        msg,
        "samples is 972, but this gate is pinned to exactly 1004 — \
         the run measured something other than what the baseline describes"
    );
}

#[test]
fn a_two_sided_range_accepts_its_interior_and_rejects_outside() {
    use crate::gate::{Bound, Comparison, compare};

    let range = Bound {
        min: Some(10.0),
        max: Some(20.0),
        noise: None,
    };
    for v in [10.0, 15.0, 20.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Pass),
            "{v} is inside [10, 20]"
        );
    }
    for v in [9.0, 21.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Fail(_)),
            "{v} is outside [10, 20]"
        );
    }
}

#[test]
fn a_bound_with_no_side_at_all_is_still_reported() {
    use crate::gate::{Bound, Comparison, compare};

    let empty = Bound {
        min: None,
        max: None,
        noise: None,
    };
    let Comparison::Skip(reason) = compare("m", 1.0, &empty) else {
        panic!("a bound with neither side must be reported as uncheckable");
    };
    assert_eq!(reason, "m has no bound");
}

/// The baselines COMMITTED IN THIS REPO must be loadable and checkable.
///
/// ★ Every other gate test builds a synthetic baseline in a scratch repo, so
/// nothing here ever read the real `.benchmarks/*/BASELINE.json`. A malformed
/// one was therefore discovered at GATE time — potentially after a 3.5-hour
/// BFCL run — rather than in a 30 ms unit test.
///
/// The `compare` assertion is the load-bearing one and is not hypothetical: a
/// two-sided `{"min": 1004, "max": 1004}` draw pin was briefly unsupported and
/// fell through to `Skip("malformed bound")`, which `check_record` counts as a
/// problem. The gate would have failed every run while blaming the baseline's
/// syntax instead of reporting the measurement. A bound the comparator cannot
/// act on is not a threshold.
#[test]
fn every_committed_baseline_parses_resolves_and_is_checkable() {
    use crate::gate::{Comparison, compare, read_baseline};

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf();

    for id in REQUIRED_GATES {
        // No per-gate file to stat any more: the thresholds are assembled
        // from every model's BENCH.toml, so "it loads and has entries" IS the
        // existence check.
        let baseline = read_baseline(&root, id)
            .unwrap_or_else(|e| panic!("{id}: committed baseline does not load: {e:#}"));
        assert_eq!(baseline.schema, 2, "{id}: unexpected schema version");
        assert!(!baseline.hardware.is_empty(), "{id}: no hardware entries");

        for (hw, entry) in &baseline.hardware {
            // The fallback must name a model that actually has thresholds,
            // otherwise a run that does not request one resolves to nothing.
            assert!(
                entry.models.contains_key(&entry.default),
                "{id}/{hw}: default {:?} has no entry in models",
                entry.default
            );
            // Resolving the default is what a gate run does first.
            let (model, mb) = baseline
                .resolve(hw, None)
                .unwrap_or_else(|e| panic!("{id}/{hw}: default does not resolve: {e:#}"));
            assert_eq!(&model, &entry.default);

            for (model, mb) in entry.models.iter().chain(std::iter::once((&model, mb))) {
                // The recipe binding is the ONLY machine-readable link from a
                // benchmark to the serve config its thresholds were measured
                // under; without it a self-start has nothing to launch.
                let recipe = mb.recipe.as_deref().unwrap_or_default();
                assert!(
                    recipe.contains('/'),
                    "{id}/{hw}/{model}: recipe {recipe:?} is not <family>/<stem>"
                );
                assert!(!mb.metrics.is_empty(), "{id}/{hw}/{model}: no thresholds");

                for (name, bound) in &mb.metrics {
                    assert!(
                        bound.min.is_some() || bound.max.is_some(),
                        "{id}/{hw}/{model}/{name}: bound has neither min nor max"
                    );
                    // Probe both far sides: whatever the shape, the comparator
                    // must reach a verdict rather than abstain.
                    for probe in [-1e9, 0.0, 1e9] {
                        assert!(
                            !matches!(compare(name, probe, bound), Comparison::Skip(_)),
                            "{id}/{hw}/{model}/{name}: compare abstains on {probe} — \
                             a bound the comparator cannot act on is not a threshold"
                        );
                    }
                }
            }
        }
    }
}
