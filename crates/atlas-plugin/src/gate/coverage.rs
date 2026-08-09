// SPDX-License-Identifier: AGPL-3.0-only

//! Which changes invalidate which gate — the deterministic floor.
//!
//! Before this module there was one bit of information: did the diff touch
//! [`PERF_PATHS`]? If yes, every record for every gate was invalid. That is
//! simultaneously too coarse and too narrow. Too coarse because `PERF_PATHS`
//! contains the string `crates`, so editing argument parsing re-opened two BFCL
//! accuracy legs at roughly three and a half GPU-hours each. Too narrow because
//! a path outside the list invalidated nothing at all, however dangerous.
//!
//! # The polarity is deliberate: exclude, do not claim
//!
//! The obvious design is for each benchmark to *claim* the regions it covers,
//! and to require a gate when a changed path is claimed. That design fails
//! **open**: the moment someone adds a new module and forgets to claim it, it
//! is covered by nothing and silently gates nothing.
//!
//! So this inverts it. Every boundary path invalidates every gate, and the only
//! way to subtract is an [`Exclusion`] carrying a written [`Exclusion::rationale`].
//! Forgetting therefore costs a re-run, never a missed regression — the same
//! asymmetry `mod.rs` already states about the boundary itself: over-broad
//! costs a re-run, under-broad is a lie.
//!
//! # This file guards itself
//!
//! An exclusion table that could exempt the file it lives in would be a lock
//! whose key is kept inside it: a PR could add "exclude everything", and that
//! very edit would trigger no gate. So a diff touching any [`BOUNDARY_FILES`]
//! entry invalidates **every** gate, and a test asserts those files appear in
//! no exclusion set.

/// A path prefix that does **not** invalidate a particular gate.
///
/// The rationale is a required field, not documentation. An exclusion is a
/// claim that a category of change cannot move this benchmark's numbers, and a
/// claim nobody wrote down is one nobody can review or refute later.
#[derive(Debug, Clone, Copy)]
pub struct Exclusion {
    pub prefix: &'static str,
    pub rationale: &'static str,
}

/// One gate and everything that does not invalidate it.
#[derive(Debug, Clone, Copy)]
pub struct GateCoverage {
    pub id: &'static str,
    pub excludes: &'static [Exclusion],
}

/// Paths whose contents can change what the engine computes.
///
/// ★ `3rdparty_patches` is the eighth entry and it closes a real bypass that
/// existed for the whole life of this gate. `layers/ops/gdn_flashinfer.rs:107`
/// dlopens the library named by `ATLAS_GDN_LIB`, and a committed recipe fixture
/// points that at `3rdparty_patches/gdn_aot/libatlasgdn.so` on a config
/// claiming +17-20% on GDN chunked prefill. Until now, replacing that AOT
/// artefact invalidated **nothing**: the engine's behaviour could change
/// materially while every committed record still read as covering.
///
/// Deliberately absent: `.benchmarks` (the records are the verdict, not its
/// subject), `bench/` and `scripts/` (harness tooling), and documentation.
pub const PERF_PATHS: [&str; 8] = [
    "crates",
    "kernels",
    "Cargo.toml",
    "Cargo.lock",
    "vendor",
    "jinja-templates",
    "rust-toolchain.toml",
    "3rdparty_patches",
];

/// Files that define the boundary itself, and therefore invalidate everything.
///
/// Editing these changes what "invalidates" means. Letting them be excluded
/// would let a change to the rules escape the rules.
///
/// ★ This list held ONE entry and that was not enough. `GATE_MACHINERY`
/// excludes the whole `crates/atlas-plugin/src/gate` prefix from every gate,
/// so a PR editing `check.rs` — `record_covers`, `invalidating_paths`,
/// `check_record`, `compare` — invalidated nothing, and then reported itself
/// covered BY ITS OWN NEW LOGIC. `coverage.rs` alone was "a lock whose key is
/// kept inside it" with the key moved one room over.
///
/// It was not theoretical: PR #420 rewrote `record_covers` and the gate listed
/// only an unrelated `atlas-kernels` file as invalidating. It read red purely
/// by accident.
///
/// The four files here are the ones that decide a verdict. `GATE_MACHINERY`
/// still covers the rest of the directory — record IO, telemetry rendering,
/// the CODEOWNERS parser — where the exclusion's argument does hold.
pub const BOUNDARY_FILES: [&str; 5] = [
    "crates/atlas-plugin/src/gate/coverage.rs",
    // `record_covers` / `invalidating_paths` / `check_record` / `compare`:
    // decides whether a record stands and whether its numbers pass.
    "crates/atlas-plugin/src/gate/check.rs",
    // `excuses` / `changed_targets`: decides which invalidating paths are
    // forgiven by the closure hash.
    "crates/atlas-plugin/src/gate/closure.rs",
    // `sources` / `configs` / `affected`: decides which targets a kernel edit
    // reaches, i.e. the input to `excuses`.
    "crates/atlas-plugin/src/gate/taxon.rs",
    // `baseline_for`: decides WHICH thresholds a record is judged against.
    "crates/atlas-plugin/src/gate/bench.rs",
];

/// Basenames under `kernels/` that are read by the gate and compiled by nothing.
///
/// `kernels/` is a boundary path, so everything beneath it invalidates every
/// gate. That is right for source, and wrong for `BENCH.toml`: it holds the
/// THRESHOLDS a record is judged against, and if editing it invalidated every
/// record, then ratcheting a bar would destroy the very record that justified
/// the ratchet. The records are the verdict, not its subject — the same
/// reasoning that keeps `.benchmarks/` out of [`PERF_PATHS`], one directory in.
///
/// This is safe only because nothing compiles it: `taxon::configs` lists
/// `HARDWARE.toml`, `MODEL.toml` and `KERNEL.toml` and deliberately not this,
/// so no target's closure hash contains it, and `bench.rs` is its only reader.
/// `bench_toml_is_not_a_closure_input` pins that.
///
/// Matched on the exact file NAME, so a directory or source file that merely
/// ends with the same characters is unaffected.
const NON_COMPILED_KERNEL_FILES: [&str; 1] = ["BENCH.toml"];

/// Whether `path` is one of the gate-read, never-compiled files above.
fn is_non_compiled_kernel_file(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("kernels/") else {
        return false;
    };
    rest.rsplit('/')
        .next()
        .is_some_and(|name| NON_COMPILED_KERNEL_FILES.contains(&name))
}

/// Gate machinery: reads records, compares them to baselines, prints a verdict.
///
/// It cannot change an inference number — it never runs a model. What it *can*
/// do is get the pass/fail logic wrong, and the right verification for that is
/// `cargo test`, a required check, which already covers this directory
/// densely. Re-measuring BFCL because a comparison operator moved buys nothing.
const GATE_MACHINERY: Exclusion = Exclusion {
    prefix: "crates/atlas-plugin/src/gate",
    rationale: "gate bookkeeping never runs a model; its correctness is covered by cargo test",
};

/// Every other benchmark's driver.
///
/// A change to the BFCL driver can change the BFCL numbers and must invalidate
/// that gate — but it cannot change what the TTFT probe measures. This is the
/// per-gate distinction that the old single-bit rule could not express.
///
/// ★ Load-bearing precondition: benchmark drivers must not import each other,
/// or excluding one from another's gate becomes false. `coverage_map_tests`
/// asserts the absence of those imports, so a future cross-import fails a test
/// rather than silently invalidating an exclusion.
const fn other_driver(prefix: &'static str, mine: &'static str) -> Exclusion {
    Exclusion {
        prefix,
        rationale: mine,
    }
}

const TTFT_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change what a first-token latency probe measures",
    ),
];

const BFCL_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change a tool-calling accuracy score",
    ),
];

const AGENTIC_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether the agent's webserver task succeeds",
    ),
];

/// The gates whose records must pass, and what each one ignores.
pub const REQUIRED: [GateCoverage; 5] = [
    GateCoverage {
        id: "agentic-webserver",
        excludes: AGENTIC_EXCLUDES,
    },
    GateCoverage {
        id: "ttft-warm-gate",
        excludes: TTFT_EXCLUDES,
    },
    GateCoverage {
        id: "ttft-cold-gate",
        excludes: TTFT_EXCLUDES,
    },
    GateCoverage {
        id: "bfcl-subset",
        excludes: BFCL_EXCLUDES,
    },
    GateCoverage {
        id: "bfcl-subset-echolp",
        excludes: BFCL_EXCLUDES,
    },
];

/// Registered benchmarks that are deliberately **not** gates, each with the
/// reason. Stated rather than implied: a reader asking "why doesn't
/// `bfcl-full` gate?" should find the answer here, not infer it from absence.
pub const NOT_REQUIRED: [(&str, &str); 3] = [
    (
        "bfcl-full",
        "the unsampled ~3600-sample draw; the two subset gates cover the same code at a \
         fraction of the GPU time, and a full run would dominate every PR",
    ),
    (
        "concurrency-sweep",
        "an exploratory table, not a pass/fail measurement — it has no baseline thresholds",
    ),
    (
        "serve-matrix",
        "a multi-checkpoint survey used for release notes; it measures breadth, not regression",
    ),
];

/// True when `path` is `entry` or lies beneath it.
///
/// ★ Component-wise, never a bare `starts_with`. `"Cargo.toml.orig"` starts
/// with `"Cargo.toml"` and `"crates2/x"` starts with `"crates"`, and neither is
/// under the entry it appears to match. Getting this wrong invalidates gates
/// for unrelated files, which trains people to distrust the gate — the failure
/// mode that ends with someone disabling it.
fn under(path: &str, entry: &str) -> bool {
    path == entry
        || path
            .strip_prefix(entry)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Whether a changed path lies on the performance boundary at all.
pub fn on_boundary(path: &str) -> bool {
    PERF_PATHS.iter().any(|entry| under(path, entry))
}

/// Whether changing `path` invalidates `gate`'s existing records.
///
/// The order of the three questions is the whole policy:
///
/// 1. Is it a boundary-defining file? Then everything is invalid — the rules
///    themselves moved.
/// 2. Is it off the boundary entirely? Then nothing is invalid.
/// 3. Otherwise it invalidates **unless** an exclusion with a written rationale
///    says why it cannot matter to this gate.
///
/// Step 3's default is the safety property: a path nobody has classified
/// invalidates, so an unclassified new subsystem over-tests instead of
/// escaping.
pub fn invalidates(gate: &GateCoverage, path: &str) -> bool {
    if BOUNDARY_FILES.iter().any(|f| under(path, f)) {
        return true;
    }
    // After the boundary-file check, so a `BENCH.toml` could never exempt the
    // rules that exempt it, and before the per-gate excludes, since this holds
    // for every gate rather than being one gate's claim about its own coverage.
    if is_non_compiled_kernel_file(path) {
        return false;
    }
    if !on_boundary(path) {
        return false;
    }
    !gate.excludes.iter().any(|e| under(path, e.prefix))
}

/// The gates invalidated by a set of changed paths.
///
/// This is the deterministic floor in one call: pure, total, and a function of
/// the paths alone. No network response, model output, environment variable or
/// wall-clock reading is an input, which is what makes the verdict reproducible
/// offline and unreachable by anything a pull request can say.
pub fn invalidated_by<'a, I>(paths: I) -> Vec<&'static str>
where
    I: IntoIterator<Item = &'a str>,
{
    let paths: Vec<&str> = paths.into_iter().collect();
    REQUIRED
        .iter()
        .filter(|gate| paths.iter().any(|p| invalidates(gate, p)))
        .map(|gate| gate.id)
        .collect()
}

/// Look up a gate's coverage by id.
pub fn find(id: &str) -> Option<&'static GateCoverage> {
    REQUIRED.iter().find(|g| g.id == id)
}
