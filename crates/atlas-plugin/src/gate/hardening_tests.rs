// SPDX-License-Identifier: AGPL-3.0-only

//! Three holes an adversarial review found, and the rules that close them.
//!
//! Each test below fails without its production change. They are grouped
//! because they share one theme: the gate trusted things it had never checked
//! — the directory a record sat in, the files that decide a verdict, and the
//! size of a "measurement noise" allowance.

use super::tests::{tempdir, *};
use super::*;

// ── 1. A record must belong to the gate whose directory it sits in ──────────

/// `ttft-warm-gate` and `ttft-cold-gate` share a checkpoint, a hardware key
/// and their metric names. On `main` the committed WARM record reads
/// 1562.58 / 4478.42 against COLD ceilings of 1728.27 / 4809.76 — so copying
/// one file into the other directory used to turn the cold gate green with no
/// cold leg ever run.
///
/// Cold-TTFT is the only leg that sees a cold-LOAD regression, and #389 (the
/// change this gate was built alongside) is "GPU-transpose quantized weights
/// at cold load" — so this was the worst pair in the suite to be able to
/// confuse, and it needed no malice: both records are written minutes apart in
/// one session with near-identical filenames.
#[test]
fn a_record_from_another_gate_does_not_satisfy_this_one() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }

    // A perfectly good record — for the WRONG gate.
    let src = gate_dir(root, "ttft-warm-gate");
    let dst = gate_dir(root, "ttft-cold-gate");
    plant(root, "ttft-warm-gate", "abc1234567", 1_785_891_382, "PASS");
    let planted = std::fs::read_dir(&src)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::copy(&planted, dst.join(planted.file_name().unwrap())).unwrap();

    // The copy is present, readable, PASSing, and names a covering commit.
    assert_eq!(std::fs::read_dir(&dst).unwrap().count(), 1);

    let status = check_gates(root, "abc1234567");
    match &status["ttft-cold-gate"] {
        GateStatus::Missing(reason) => assert_eq!(
            reason,
            "latest record belongs to ttft-warm-gate, not ttft-cold-gate \
             (2026-08-05-abc1234567.json)"
        ),
        other => panic!("a warm record must not satisfy the cold gate: {other:?}"),
    }
}

// ── 2. The files that decide a verdict are inside the boundary ─────────────

/// ★ The files that DECIDE a verdict are inside the boundary, and re-open
/// everything.
///
/// This inverts what `gate_machinery_changes_cost_no_gpu_hours` used to
/// assert, deliberately. `GATE_MACHINERY` excludes the whole
/// `crates/atlas-plugin/src/gate` prefix, and `BOUNDARY_FILES` listed only
/// `coverage.rs` — so a PR rewriting `record_covers` invalidated NOTHING and
/// then reported itself covered by its own new logic.
///
/// That happened. PR #420 rewrote `record_covers` and the gate named only an
/// unrelated `atlas-kernels` file as invalidating; it read red by accident,
/// not by rule.
///
/// The exclusion's argument — "it never runs a model, `cargo test` covers it"
/// — is right about `record.rs` and wrong about `check.rs`: a verdict function
/// cannot be trusted to certify the commit that changes it, however good the
/// unit tests are. Cost of the stricter rule: a gate-logic PR owes one gate
/// run. That is the correct price for editing the thing that decides whether
/// anything owes a gate run.
#[test]
fn verdict_logic_is_inside_the_boundary() {
    for f in [
        "crates/atlas-plugin/src/gate/coverage.rs",
        "crates/atlas-plugin/src/gate/check.rs",
        "crates/atlas-plugin/src/gate/scoring.rs",
        "crates/atlas-plugin/src/gate/closure.rs",
        "crates/atlas-plugin/src/gate/taxon.rs",
        "crates/atlas-plugin/src/gate/bench.rs",
    ] {
        let hit = super::coverage::invalidated_by([f]);
        assert_eq!(
            hit.len(),
            super::coverage::REQUIRED.len(),
            "{f} decides a verdict and must re-open every gate, got {hit:?}"
        );
    }
}

/// ★ The boundary follows the SYMBOLS, not the filenames.
///
/// The list above pins names, and names go stale: the 6c6fcb2b1 split moved
/// `check_record`/`compare` from `check.rs` into a new `scoring.rs` that was
/// in `GATE_MACHINERY`'s excluded prefix and NOT in `BOUNDARY_FILES` — so an
/// edit to the pass/fail comparison invalidated nothing and would have been
/// judged by its own new logic, the exact PR #420 shape the boundary exists
/// to close. This test finds each verdict function's DEFINING file in the
/// gate sources and asserts that file re-opens every gate, so the next
/// 500-line split moves the boundary with the function or fails here.
/// Every source under `src/gate` is classified, and the classification agrees
/// with `BOUNDARY_FILES`.
///
/// ★ WHAT THIS CATCHES THAT NOTHING ELSE DID. The test below walks a hardcoded
/// list of seven verdict FUNCTIONS and proves each one's defining file re-opens
/// every gate. It is exactly as complete as that list, so it catches a function
/// MOVING and is blind to a verdict function that is NEW. `agreement.rs` was
/// added with `check` and `sensitivity_of` — which decide whether a record SET
/// is acceptable — and no check in this crate noticed, because nobody added
/// those symbols to the list below. A new file under `src/gate` was invisible.
///
/// So the guard is inverted: a file is not asked to prove it is dangerous, it
/// is required to declare which it is. `BOUNDARY_FILES` and
/// `GATE_MACHINERY_FILES` must together be exactly the directory listing, and
/// no file may be in both. Adding a source here fails this test until its
/// author classifies it, and classifying it as machinery is a claim someone
/// wrote down and can be argued with in review — which is the whole point.
#[test]
fn gate_sources_are_all_classified() {
    let gate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gate");
    let mut on_disk = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&gate_dir).expect("gate dir listable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        // Test sources carry no verdict; they are excluded by name, the same
        // rule the symbol walker uses.
        if name.ends_with("_tests.rs") || name == "tests.rs" {
            continue;
        }
        on_disk.insert(format!("crates/atlas-plugin/src/gate/{name}"));
    }

    let boundary: std::collections::BTreeSet<&str> = super::coverage::BOUNDARY_FILES
        .iter()
        .copied()
        .filter(|p| p.starts_with("crates/atlas-plugin/src/gate/"))
        .collect();
    let machinery: std::collections::BTreeSet<&str> = super::coverage::gate_machinery_files()
        .iter()
        .copied()
        .collect();

    let both: Vec<_> = boundary.intersection(&machinery).collect();
    assert!(
        both.is_empty(),
        "classified as BOTH boundary and machinery: {both:?} — a file decides a \
         verdict or it does not"
    );

    let classified: std::collections::BTreeSet<String> = boundary
        .iter()
        .chain(machinery.iter())
        .map(|s| s.to_string())
        .collect();

    let unclassified: Vec<_> = on_disk.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "unclassified gate source(s): {unclassified:?}\n\
         Every file under src/gate must be in BOUNDARY_FILES (it decides a \
         verdict, so editing it re-opens every gate) or in \
         GATE_MACHINERY_FILES (it does not, and someone is on record saying \
         so). `agreement.rs` reached main unclassified and could have judged \
         its own PR by its own new rule."
    );

    let stale: Vec<_> = classified.difference(&on_disk).collect();
    assert!(
        stale.is_empty(),
        "classified but not on disk: {stale:?} — a rename or delete left the \
         classification pointing at nothing, which silently shrinks the boundary"
    );
}

#[test]
fn every_verdict_symbol_is_defined_inside_the_boundary() {
    let gate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gate");
    let verdict_fns = [
        "fn record_covers",
        "fn invalidating_paths",
        "fn check_record",
        "fn compare",
        "fn excuses",
        "fn changed_targets",
        "fn baseline_for",
    ];
    for symbol in verdict_fns {
        let mut defined_in = Vec::new();
        for entry in std::fs::read_dir(&gate_dir).expect("gate dir listable") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("gate source readable");
            let declares_symbol = src.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(symbol)
                    || line.starts_with(&format!("pub {symbol}"))
                    || line.starts_with(&format!("pub(crate) {symbol}"))
            });
            if declares_symbol {
                let rel = format!(
                    "crates/atlas-plugin/src/gate/{}",
                    path.file_name().unwrap().to_str().unwrap()
                );
                // Test files mention the symbols without defining verdicts.
                if !rel.ends_with("_tests.rs") && !rel.ends_with("/tests.rs") {
                    defined_in.push(rel);
                }
            }
        }
        assert!(
            !defined_in.is_empty(),
            "{symbol} not found anywhere under src/gate — if it was renamed, \
             rename it here too; the boundary must keep tracking it"
        );
        for rel in defined_in {
            let hit = super::coverage::invalidated_by([rel.as_str()]);
            assert_eq!(
                hit.len(),
                super::coverage::REQUIRED.len(),
                "{rel} contains `{symbol}` but is not in BOUNDARY_FILES — a \
                 verdict function moved out of the boundary (the PR #420 hole, \
                 reintroduced by a file split)"
            );
        }
    }
}

// ── 3. `noise` is a measurement allowance, not a threshold change ───────────

fn bench_toml(root: &std::path::Path, metrics: &str) -> std::path::PathBuf {
    let dir = root.join("kernels/gb10/qwen3.6-27b/nvfp4");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\narch = \"sm_121f\"\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(root.join("kernels/gb10/qwen3.6-27b/MODEL.toml"), "").unwrap();
    let p = root.join("kernels/gb10/qwen3.6-27b/BENCH.toml");
    std::fs::write(
        &p,
        format!(
            "[[benchmarks]]\ngate = \"bfcl-subset\"\nquant = \"nvfp4\"\n\
             checkpoint = \"x/y\"\nrecipe = \"a/b\"\nstatus = \"measured\"\n\
             default = true\n{metrics}"
        ),
    )
    .unwrap();
    p
}

/// A `noise` big enough to clear any record is a threshold change wearing a
/// measurement-noise label — and the most review-invisible one available,
/// because `BENCH.toml` is deliberately exempt from invalidating any gate.
#[test]
fn an_absurd_noise_allowance_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = 1000.0\n",
    );
    let err = super::bench::load_all(root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / x/y metric overall_accuracy: noise 1000 exceeds 5% of the bound \
             (87.44) — that is a threshold change wearing a measurement-noise label. Move the \
             bound instead, so the ratchet is visible in review.",
            root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
        )
    );
}

/// The values actually in the tree (0.4 against floors of 83-89, ~0.46%) must
/// keep loading — a rule that rejects the status quo is not a rule, it is a
/// migration.
#[test]
fn the_real_noise_values_still_load() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = 0.4\n",
    );
    let loaded =
        super::bench::load_all(root).expect("0.4 on an 87.44 floor is real measurement noise");
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded[0].1.metrics.as_ref().unwrap()["overall_accuracy"].noise,
        Some(0.4)
    );
}

/// `min == max` is how the BFCL draw size is pinned. `check::compare` applies
/// noise to the two-sided arm too, so noise on a pin silently disables the one
/// guard against a changed draw — and a changed draw is undetectable after the
/// fact, because a different category mix moves `normalized_single_turn_score`
/// by ~1.8 points while leaving `overall_accuracy` in the same place.
#[test]
fn noise_on_an_exact_pin_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.samples]\nmin = 995.0\nmax = 995.0\nnoise = 1.0\n",
    );
    let err = super::bench::load_all(root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / x/y metric samples is an EXACT pin (min == max == Some(995.0)) \
             and carries noise 1. Noise on a pin disables it — and a pin is used for things \
             like the BFCL draw size, where a changed draw is undetectable after the fact.",
            root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
        )
    );
}

/// Negative and non-finite slack are nonsense; refuse rather than propagate.
#[test]
fn negative_and_non_finite_noise_are_refused() {
    for (literal, rendered) in [("-5.0", "-5"), ("nan", "NaN"), ("inf", "inf")] {
        let dir = tempdir::Dir::new();
        let root = dir.path();
        bench_toml(
            root,
            &format!("[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = {literal}\n"),
        );
        let err = super::bench::load_all(root).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "{}: bfcl-subset / x/y metric overall_accuracy: noise must be finite and \
                 non-negative, got {rendered}",
                root.join("kernels/gb10/qwen3.6-27b/BENCH.toml").display()
            )
        );
    }
}
