// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the deterministic floor.
//!
//! Split from `coverage_tests.rs` for the 500-LoC cap. These are the tests that
//! pin the safety property itself, so they are deliberately literal: they
//! assert the *shape* of the policy, not just that today's table happens to
//! produce today's answers.

use super::REQUIRED_GATES;
use super::coverage::{
    self, BOUNDARY_FILES, Exclusion, GateCoverage, NOT_REQUIRED, PERF_PATHS, REQUIRED,
};

/// The repo root, two levels above this crate's manifest — the same derivation
/// `coverage_tests::every_invalidating_path_exists_in_this_repo` uses.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf()
}

/// ★ THE INVARIANT.
///
/// The required set is a join-semilattice: adding files can only add gates.
/// This is what makes the floor safe against anything a pull request can say —
/// there is no input, hostile or otherwise, that shrinks the answer, because
/// `invalidated_by` has no branch that removes an element.
#[test]
fn adding_a_changed_file_never_removes_a_required_gate() {
    let base = ["crates/atlas-plugin/src/benchmarks/bfcl/report.rs"];
    let additions = [
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/spark-model/src/layers/ops/fp8_moe.rs",
        "Cargo.lock",
        "3rdparty_patches/gdn_aot/libatlasgdn.so",
        "crates/atlas-plugin/src/gate/check.rs",
        "some/unclassified/new/subsystem.rs",
    ];
    let before: Vec<_> = coverage::invalidated_by(base);
    assert!(!before.is_empty(), "the monotonicity oracle must execute");
    for extra in additions {
        let mut with = base.to_vec();
        with.push(extra);
        let after = coverage::invalidated_by(with.iter().copied());
        for gate in &before {
            assert!(
                after.contains(gate),
                "adding {extra} removed {gate} from the required set"
            );
        }
    }
}

/// ★ The bypass this module was written to close.
///
/// `layers/ops/gdn_flashinfer.rs:107` dlopens the library named by
/// `ATLAS_GDN_LIB`, and a committed recipe fixture points that at
/// `3rdparty_patches/gdn_aot/libatlasgdn.so` on a config claiming +17-20% on
/// GDN chunked prefill. Before this, swapping that artefact invalidated
/// nothing — the engine's behaviour could change materially while every
/// record still read as covering.
#[test]
fn replacing_the_aot_gdn_library_invalidates_every_gate() {
    let hit = coverage::invalidated_by(["3rdparty_patches/gdn_aot/libatlasgdn.so"]);
    assert_eq!(
        hit, REQUIRED_GATES,
        "an AOT kernel swap must re-open every gate, got {hit:?}"
    );
}

/// ★ Fail-closed: a path nobody has classified requires everything.
///
/// The alternative — unclassified means unaffected — is the design that fails
/// open, where a new subsystem silently gates nothing until someone remembers
/// to claim it.
#[test]
fn an_unclassified_path_on_the_boundary_invalidates_everything() {
    for path in [
        "crates/spark-model/src/some_new_module.rs",
        "kernels/gb10/brand_new_kernel.cu",
        "vendor/whatever",
        "rust-toolchain.toml",
    ] {
        let hit = coverage::invalidated_by([path]);
        assert_eq!(hit, REQUIRED_GATES, "{path} under-invalidated: {hit:?}");
    }
}

/// ★ Component-wise matching, not `starts_with`.
///
/// `Cargo.toml.orig` starts with `Cargo.toml`, and `crates2/x` starts with
/// `crates` — neither is under the entry it appears to match. Getting this
/// wrong invalidates gates for unrelated files, which teaches people the gate
/// is noise, which ends with someone turning it off.
#[test]
fn lookalike_paths_do_not_match_the_boundary() {
    for path in [
        "Cargo.toml.orig",
        "Cargo.lockfile",
        "crates2/src/lib.rs",
        "kernels_old/x.cu",
        "vendored/thing.rs",
        "3rdparty_patches_backup/x",
    ] {
        assert!(
            !coverage::on_boundary(path),
            "{path} must not count as a boundary path"
        );
        assert!(
            coverage::invalidated_by([path]).is_empty(),
            "{path} must invalidate nothing"
        );
    }
}

/// The exact entries still match, prefix and file alike.
#[test]
fn real_boundary_paths_match() {
    for path in [
        "crates",
        "crates/spark-model/src/lib.rs",
        "Cargo.toml",
        "Cargo.lock",
        "kernels/gb10/common/x.cu",
        "3rdparty_patches/gdn_aot/libatlasgdn.so",
        "rust-toolchain.toml",
    ] {
        assert!(coverage::on_boundary(path), "{path} should be on-boundary");
    }
}

/// ★ The map may not exempt the file that defines the map.
///
/// Otherwise a PR could add "exclude everything" and that very edit would
/// trigger no gate — a lock whose key is kept inside it.
///
/// Note this is asserted **behaviourally**, not structurally. An earlier
/// version of this test forbade any exclusion prefix from containing a
/// boundary file, and it failed immediately: `GATE_MACHINERY` excludes
/// `crates/atlas-plugin/src/gate`, which contains `coverage.rs`. That overlap
/// is fine and the structural rule was simply wrong — `invalidates` checks
/// `BOUNDARY_FILES` *before* it consults exclusions, so the containment cannot
/// take effect. What matters is the outcome, so the outcome is what is pinned.
#[test]
fn no_gate_can_exempt_the_file_that_defines_the_rules() {
    for gate in REQUIRED.iter() {
        for boundary in BOUNDARY_FILES {
            assert!(
                coverage::invalidates(gate, boundary),
                "{} does not re-open when {boundary} changes",
                gate.id
            );
        }
    }
}

#[test]
fn editing_the_coverage_map_invalidates_every_gate() {
    for boundary in BOUNDARY_FILES {
        let hit = coverage::invalidated_by([boundary]);
        assert_eq!(
            hit, REQUIRED_GATES,
            "changing {boundary} must re-open everything, got {hit:?}"
        );
    }
}

/// ★ Every exclusion carries a real rationale.
///
/// An exclusion is a claim that a class of change cannot move a number. A claim
/// nobody wrote down cannot be reviewed, and cannot be refuted when it turns
/// out to be wrong.
#[test]
fn every_exclusion_states_why() {
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(!ex.prefix.is_empty(), "{}: empty prefix", gate.id);
            assert!(
                ex.rationale.trim().len() > 20,
                "{} excludes {} with no real rationale: {:?}",
                gate.id,
                ex.prefix,
                ex.rationale
            );
        }
    }
}

/// A rule that matches nothing in the tree is rot: it either describes a path
/// that was renamed away, or it was wrong when written. Either way it is a
/// claim about code that does not exist.
#[test]
fn no_exclusion_is_a_dead_glob() {
    let root = repo_root();
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(
                root.join(ex.prefix).exists(),
                "{} excludes {}, which does not exist in this repo",
                gate.id,
                ex.prefix
            );
        }
    }
}

/// Exclusions must not name a path that is off-boundary anyway — that is a
/// rule with no effect, and a reader would reasonably assume it has one.
#[test]
fn every_exclusion_is_actually_on_the_boundary() {
    for gate in REQUIRED.iter() {
        for ex in gate.excludes {
            assert!(
                coverage::on_boundary(ex.prefix),
                "{} excludes {}, which is not on the boundary — the rule does nothing",
                gate.id,
                ex.prefix
            );
        }
    }
}

#[path = "coverage_driver_tests.rs"]
mod coverage_driver_tests;

/// `REQUIRED_GATES` is now a view over the coverage table rather than a second
/// hand-maintained list — the two can no longer disagree.
#[test]
fn required_gates_is_derived_from_the_coverage_table() {
    let ids: Vec<&str> = REQUIRED.iter().map(|g| g.id).collect();
    assert_eq!(REQUIRED_GATES.to_vec(), ids);
}

/// Every registered benchmark is accounted for: gated, or explicitly not gated
/// with a reason. Silence about a benchmark is how one drifts out of the gate
/// without anyone deciding that it should.
#[test]
fn every_registered_benchmark_is_either_required_or_explicitly_excused() {
    for descriptor in crate::registry::all() {
        let gated = REQUIRED.iter().any(|g| g.id == descriptor.id);
        let excused = NOT_REQUIRED.iter().any(|(id, _)| *id == descriptor.id);
        assert!(
            gated ^ excused,
            "{} is {}",
            descriptor.id,
            if gated {
                "both gated and excused"
            } else {
                "neither gated nor listed in NOT_REQUIRED with a reason"
            }
        );
    }
}

#[test]
fn every_excusal_names_a_real_benchmark_and_a_reason() {
    let mut seen = std::collections::BTreeSet::new();
    for (id, why) in NOT_REQUIRED {
        assert!(seen.insert(id), "{id} is excused more than once");
        assert!(
            crate::registry::find(id).is_some(),
            "{id} is excused but not registered"
        );
        assert!(
            why.trim().len() > 20,
            "{id} is excused without a real reason"
        );
    }
}

/// The per-gate distinction the old single-bit rule could not express: a change
/// to one benchmark's driver re-opens that benchmark and leaves the others.
#[test]
fn a_driver_change_invalidates_only_its_own_gate() {
    let hit = coverage::invalidated_by(["crates/atlas-plugin/src/benchmarks/bfcl/report.rs"]);
    assert_eq!(hit, ["bfcl-subset", "bfcl-subset-echolp"]);
}

/// Gate BOOKKEEPING does not re-open GPU measurements — the change that
/// motivated this whole module.
///
/// Note what is NOT in this list any more: `check.rs`. See the test below.
#[test]
fn gate_bookkeeping_changes_cost_no_gpu_hours() {
    let hit = coverage::invalidated_by([
        "crates/atlas-plugin/src/gate/record.rs",
        "crates/atlas-plugin/src/gate/telemetry.rs",
        "crates/atlas-plugin/src/gate/codeowners.rs",
    ]);
    assert!(
        hit.is_empty(),
        "record IO, telemetry rendering and CODEOWNERS parsing cannot move a \
         measurement; they should not re-open any gate, got {hit:?}"
    );
}

/// Off-boundary paths cost nothing — the other half of the value, and the case
/// that used to be invisible rather than cheap.
#[test]
fn documentation_only_changes_require_no_gate() {
    let hit = coverage::invalidated_by([
        "docs/adr/0011-ep-batched-decode-optimization.md",
        "README.md",
        "CONTRIBUTING.md",
    ]);
    assert!(hit.is_empty(), "{hit:?}");
}

/// A gate with no exclusions is invalidated by every boundary path — the
/// degenerate case must still behave, since that is what a newly added gate
/// looks like before anyone writes exclusions for it.
#[test]
fn a_gate_with_no_exclusions_is_invalidated_by_any_boundary_path() {
    let bare = GateCoverage {
        id: "brand-new",
        excludes: &[],
    };
    assert!(coverage::invalidates(&bare, "crates/anything.rs"));
    assert!(!coverage::invalidates(&bare, "docs/anything.md"));
}

/// An exclusion cannot rescue a boundary file, even if someone writes one.
#[test]
fn an_exclusion_cannot_override_a_boundary_file() {
    let sneaky = GateCoverage {
        id: "sneaky",
        excludes: &[Exclusion {
            prefix: "crates",
            rationale: "a maximally broad exclusion, as an attacker would write it",
        }],
    };
    for boundary in BOUNDARY_FILES {
        assert!(
            coverage::invalidates(&sneaky, boundary),
            "{boundary} escaped via a blanket exclusion"
        );
    }
}

#[test]
fn the_boundary_has_no_duplicate_entries() {
    let mut seen = PERF_PATHS.to_vec();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(before, seen.len(), "PERF_PATHS contains a duplicate");
}

// ---------------------------------------------------------------------------
// BENCH.toml: read by the gate, compiled by nothing
// ---------------------------------------------------------------------------

/// ★ A threshold ratchet must not destroy the record that justified it.
///
/// `BENCH.toml` lives under `kernels/`, which is a boundary path, so without an
/// exemption every gate would re-open the moment someone raised a bar — and the
/// run proving the new bar was reachable would be the first casualty.
#[test]
fn a_bench_toml_edit_invalidates_nothing() {
    for gate in REQUIRED.iter() {
        assert!(
            !coverage::invalidates(gate, "kernels/gb10/qwen3.6-27b/BENCH.toml"),
            "{}: a threshold edit re-opened the gate",
            gate.id
        );
    }
}

/// The exemption is by exact file NAME, so neighbours keep invalidating.
#[test]
fn the_bench_toml_exemption_does_not_leak_to_neighbours() {
    for gate in REQUIRED.iter() {
        for path in [
            "kernels/gb10/qwen3.6-27b/MODEL.toml",
            "kernels/gb10/qwen3.6-27b/nvfp4/KERNEL.toml",
            "kernels/gb10/qwen3.6-27b/nvfp4/BENCH.toml.cu",
            "kernels/gb10/qwen3.6-27b/BENCH.toml/inner.cu",
            "kernels/gb10/common/NOT-BENCH.toml",
        ] {
            assert!(
                coverage::invalidates(gate, path),
                "{}: {path} was exempted but is not a BENCH.toml",
                gate.id
            );
        }
    }
}

/// The exemption is scoped to `kernels/`. A file of that name anywhere else —
/// notably inside the gate crate — must keep its normal behaviour.
#[test]
fn the_bench_toml_exemption_is_scoped_to_the_kernel_tree() {
    let gate = REQUIRED
        .iter()
        .find(|g| {
            !g.excludes
                .iter()
                .any(|e| e.prefix.starts_with("crates/spark-model"))
        })
        .expect("a gate that does not exclude spark-model");
    assert!(
        coverage::invalidates(gate, "crates/spark-model/BENCH.toml"),
        "the exemption must not apply outside kernels/"
    );
}
