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
    assert!(
        matches!(&status["ttft-cold-gate"], GateStatus::Missing(_)),
        "a ttft-warm-gate record must not satisfy ttft-cold-gate, got {:?}",
        status["ttft-cold-gate"]
    );
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
    assert!(err.contains("exceeds"), "{err}");
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
    super::bench::load_all(root).expect("0.4 on an 87.44 floor is real measurement noise");
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
    assert!(err.contains("EXACT pin"), "{err}");
}

/// Negative and non-finite slack are nonsense; refuse rather than propagate.
#[test]
fn negative_noise_is_refused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    bench_toml(
        root,
        "[benchmarks.metrics.overall_accuracy]\nmin = 87.44\nnoise = -5.0\n",
    );
    let err = super::bench::load_all(root).unwrap_err().to_string();
    assert!(err.contains("non-negative"), "{err}");
}
