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
pub const BOUNDARY_FILES: [&str; 8] = [
    "crates/atlas-plugin/src/gate/coverage.rs",
    // `required_for` / `union` / `intent_only`: decides what the INTENT half
    // adds on top of the path-derived floor. Once intent can escalate a gate,
    // this file decides a verdict by the same criterion as the four below.
    "crates/atlas-plugin/src/gate/required.rs",
    // ★ The intent half's `coverage.rs`, and it lives OUTSIDE `PERF_PATHS`
    // entirely — so before this entry, deleting every `_benches` line in the
    // taxonomy invalidated nothing at all. That is the same lock-whose-key-is-
    // -kept-inside-it shape this list was created to close, left unapplied to
    // the half added later.
    //
    // `invalidates` checks BOUNDARY_FILES *before* `on_boundary`, so an
    // off-PERF_PATHS entry works here; a test pins that.
    //
    // ★ THE COST IS REAL: a taxonomy edit now re-opens all ten gates (~4h19m
    // of GPU). That is deliberate. The alternative is that removing a `_benches`
    // line silently reduces coverage with nothing to notice — and a cheap edit
    // that quietly weakens the gate is worse than an expensive one that cannot.
    ".github/pr-taxonomy.json",
    // `record_covers` / `invalidating_paths`: decides whether a record
    // stands against the changed paths.
    "crates/atlas-plugin/src/gate/check.rs",
    // `check_record` / `compare`: decides whether a record's numbers pass.
    // Split out of check.rs at the 500-line boundary — the verdict logic
    // moved, so the boundary moves with it (a `hardening_tests` test walks
    // the defining files by symbol so the next split cannot silently drop
    // one out of the boundary again).
    "crates/atlas-plugin/src/gate/scoring.rs",
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

/// A Rust source file whose only module edge is guarded by `#[cfg(test)]`.
///
/// This is an exact registry, not a naming rule. Each entry identifies the
/// parent declaration that proves the file is absent from release builds, and
/// `test_only_coverage_tests` fails if that declaration loses its guard or a
/// second source edge appears. A new file remains fail-closed until it is
/// registered with the same proof.
#[derive(Debug, Clone, Copy)]
pub struct TestOnlyRustModule {
    pub path: &'static str,
    pub parent: &'static str,
    pub name: &'static str,
    pub declared_path: Option<&'static str>,
}

pub const TEST_ONLY_RUST_MODULES: &[TestOnlyRustModule] = &[
    TestOnlyRustModule {
        path: "crates/atlas-core/src/config/tests.rs",
        parent: "crates/atlas-core/src/config.rs",
        name: "tests",
        declared_path: None,
    },
    TestOnlyRustModule {
        path: "crates/atlas-core/src/config/gguf/tests.rs",
        parent: "crates/atlas-core/src/config/gguf.rs",
        name: "tests",
        declared_path: None,
    },
    TestOnlyRustModule {
        path: "crates/atlas-core/src/config/parsers/lora_tests.rs",
        parent: "crates/atlas-core/src/config/parsers/lora.rs",
        name: "tests",
        declared_path: Some("lora_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/atlas-plugin/src/benchmarks/concurrency_tests.rs",
        parent: "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        name: "concurrency_tests",
        declared_path: Some("concurrency_tests.rs"),
    },
];

fn is_test_only_rust_module(path: &str) -> bool {
    TEST_ONLY_RUST_MODULES
        .iter()
        .any(|entry| path == entry.path)
}

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

/// The concurrency benchmark's two production files are flat files rather
/// than a driver directory. Name both exactly: a directory-shaped prefix such
/// as `benchmarks/concurrency` matches neither file, while excluding the whole
/// `benchmarks` directory would also hide shared HTTP and statistics code that
/// can change several instruments.
const fn concurrency_driver(prefix: &'static str, rationale: &'static str) -> Exclusion {
    other_driver(prefix, rationale)
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
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change what a first-token latency probe measures",
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
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change a tool-calling accuracy score",
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
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether the agent's webserver task succeeds",
    ),
];

/// What the SSM state poisoning gate ignores: gate bookkeeping and every
/// OTHER benchmark driver. Its own driver directory is deliberately NOT here
/// — a change to the poison detector re-opens the poison gate, exactly as a
/// change to the TTFT probe re-opens the TTFT gates.
///
/// The gate measures accumulated engine state (prefix-cache / SSM-snapshot
/// restore determinism). A client-side driver that only issues requests —
/// TTFT, BFCL, agentic, contamination — cannot change whether an identical
/// replay comes back byte-identical; only the engine can.
const SSM_POISON_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether an identical replay returns identical bytes",
    ),
];

/// The concurrency curve is a LATENCY/THROUGHPUT measurement of the serving
/// path, so almost nothing is excusable: scheduler, kernels, batching, KV and
/// sampling all move it. Only the other benchmark DRIVERS are — a driver is
/// client-side request-issuing code that cannot change how fast the server
/// answers someone else's requests.
///
/// Deliberately NOT excluded, though the TTFT gates exclude them: nothing in
/// the engine. A change that costs 10% at C=32 while leaving C=1 flat is
/// exactly the regression this curve exists to catch, and the TTFT gates
/// measure a single request — they cannot see it.
const CONCURRENCY_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver issues single requests client-side; it cannot change how fast the \
         server answers a batch of 32",
    ),
];

/// The decode-floor gate measures single-user DECODE throughput of the
/// serving path, so — like the concurrency curve — nothing in the engine is
/// excusable: kernels, scheduler, batching, KV, sampling and the MTP verify
/// path all move it, and so does the usage plumbing its accept pin reads.
/// Only the other benchmark DRIVERS are excluded: client-side request-issuing
/// code cannot change how fast the server decodes. Its own driver file is
/// deliberately NOT here — a change to the pins re-opens the pins.
const DECODE_FLOOR_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change the server's single-user decode rate",
    ),
];

/// What the cross-contamination candidate ignores: gate bookkeeping and the
/// OTHER benchmark drivers, exactly as a required gate would. Its own driver
/// directory is deliberately NOT here — a change to the detector re-opens the
/// detector.
const CONTAMINATION_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change whether one request's state leaks into another's output",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether one request's state leaks into another's output",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change whether one request's state leaks into another's output",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether one request's state leaks into another",
    ),
];

/// The vision gate answers one question — does the served model see the image
/// it was sent, at the resolution its checkpoint permits — so anything that
/// cannot change the pixels reaching the encoder or the tokens they become is
/// excluded. The other benchmark drivers are the clear case: none of them can
/// alter vision preprocessing.
const VISION_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/atlas-plugin/src/benchmarks/ttft",
        "the TTFT driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/bfcl",
        "the BFCL driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/agentic",
        "the agentic driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/atlas-plugin/src/benchmarks/contamination",
        "the contamination driver cannot change how an image is patched or how many tokens it becomes",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change image preprocessing or encoder tokens",
    ),
];

/// The gates whose records must pass, and what each one ignores.
pub const REQUIRED: [GateCoverage; 11] = [
    GateCoverage {
        id: "agentic-webserver",
        excludes: AGENTIC_EXCLUDES,
    },
    // Vision models only, and that constraint lives in BENCH.toml rather than
    // here: coverage is path-based with no per-model dimension, while a
    // `[[benchmarks]] gate = "vision-fidelity"` entry exists on exactly the
    // three targets that ship a vision tower. A text-only target has no entry,
    // so the gate has nothing to run and nothing to satisfy.
    //
    // REQUIRED rather than a promotion candidate because it is measured: all
    // three targets ran it on 2026-08-14 and passed 8/8 geometry, 3/3 probes,
    // control held, with IDENTICAL token counts across a dense NVFP4, an
    // unsloth NVFP4 and an FP8 MoE. Its bounds are absolute and carry no
    // noise term, so there was never a thresholds-less entry problem to
    // solve first (the hurdle that kept `concurrency-sweep` a candidate
    // until its 2026-08-15 calibration).
    GateCoverage {
        id: "vision-fidelity",
        excludes: VISION_EXCLUDES,
    },
    // Video models only. The BENCH.toml mechanism carries the constraint: a
    // target with no `gate = "video-fidelity"` entry has nothing to run.
    //
    // It shares the vision excludes: the question "did the frames reach the
    // model, in order" is answered by the same preprocessing, encoder and
    // splice, so the same set of paths can change the answer.
    GateCoverage {
        id: "video-fidelity",
        excludes: VISION_EXCLUDES,
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
    // 2026-08-11: the batch4 stack poisoned the Marconi SSM-snapshot restore
    // and the agentic gate only caught it by accident (runs 8/9 degenerated).
    // This gate polices that class directly — it is mandatory, not a
    // promotion candidate, because the bug it exists to catch already shipped.
    GateCoverage {
        id: "ssm-state-poisoning-gate",
        excludes: SSM_POISON_EXCLUDES,
    },
    // ── Promoted from PROMOTION_CANDIDATES 2026-08-15 ──────────────────────
    //
    // Both promotions' calibration preconditions are met, on the FIXED
    // instruments the gates now pin (the maintainer wants the concurrency
    // gate MANDATORY, not accruing debt):
    //
    // * `decode-floor`: the promotion ruling demanded a >=10-run sigma
    //   calibration so the bar comes from measured run-to-run noise. The
    //   12-run 2026-08-15 set (28.0–28.1, mean 28.03, sigma ~0.05) satisfied
    //   the RULE but was measured on a hand-built serve with the default lm
    //   head, so it never described this gate; the floor beside it read 21.0
    //   from n=3 for weeks. Settled 2026-09-06: ten consecutive runs on the
    //   gate's own serve (dgx2, mean 22.78, sigma 0.063) put the BENCH.toml
    //   floor at 22.7, noise 0.5.
    // * `concurrency-sweep`: it could not be REQUIRED while its entries were
    //   thresholds-less (check_record refuses a thresholds-less PASS). The
    //   dense Qwen3.8-27B entry now carries measured floors from an n=3
    //   ladder on the fixed instrument (C=1/4/8/16, isl 512, osl 320, zero
    //   vacuous cells), and the entry's param_overrides pin that instrument
    //   so a gate run reproduces it.
    GateCoverage {
        id: "decode-floor",
        excludes: DECODE_FLOOR_EXCLUDES,
    },
    GateCoverage {
        id: "concurrency-sweep",
        excludes: CONCURRENCY_EXCLUDES,
    },
    // The same ladder with the DFlash2 drafter armed, added 2026-08-29. It
    // shares CONCURRENCY_EXCLUDES because it shares the driver: the two gates
    // differ only in the recipe their BENCH.toml entries name, so exactly the
    // same set of foreign drivers cannot reach either number.
    //
    // ★ It is REQUIRED, not a promotion candidate, because the speculative
    // path had NO gate at all. Every other required gate serves a no-drafter
    // recipe, which is why the 2026-08 C4 investigation could only rule
    // DFlash2 out by reading source: no committed record spoke for it. A path
    // that ships and is never certified is the gap this closes; the ongoing
    // cost is that every future PR owes an eleventh record.
    GateCoverage {
        id: "concurrency-sweep-dflash2",
        excludes: CONCURRENCY_EXCLUDES,
    },
];

/// Registered benchmarks that are deliberately **not** gates, each with the
/// reason. Stated rather than implied: a reader asking "why doesn't
/// `bfcl-full` gate?" should find the answer here, not infer it from absence.
/// Gates that are NOT required **yet**, but are on a declared promotion path.
///
/// ★ The difference from [`NOT_REQUIRED`] is the whole point. Its permanently
/// excused entries carry a reason: `bfcl-full` duplicates cheaper coverage,
/// `quick-speed-bench` is a measurement tool, `serve-matrix` measures breadth.
/// Nothing is owed for them, ever.
///
/// A promotion candidate is different: it is a gate we intend to require once
/// it has proven itself, run on release cuts in the meantime. That leaves a gap
/// with a shape this repository has been bitten by repeatedly — **listing only
/// what was gated silently converts "ungated" into "unaffected"**. A PR whose
/// paths the candidate cares about can merge with nothing measured, and nothing
/// anywhere says so.
///
/// So a candidate carries a FULL [`GateCoverage`], exactly as a required gate
/// does, and [`promotion_debt`] joins it against changed paths. The telemetry
/// renders the result as an explicit debt row. No model is involved and none is
/// needed: it is a deterministic join between paths and records.
///
/// The first entry is `cross-contamination` (owner pattern: NOT_REQUIRED,
/// then promote once proven on release cuts). `memory-convergence` is the
/// next intended entry and cannot be listed until the benchmark exists — a
/// candidate naming an unregistered id would be a debt row nobody can ever
/// discharge, which is worse than no row.
/// `every_promotion_candidate_is_a_registered_benchmark` pins that.
///
/// ★ The list is the PIPELINE, not a parking lot: `concurrency-sweep` and
/// `decode-floor` both graduated to [`REQUIRED`] on 2026-08-15 once their
/// calibration preconditions were met (see the comments on their REQUIRED
/// entries). Their old candidate entries are gone from here because a gate
/// cannot be owed and excused at once — the test above pins that.
pub const PROMOTION_CANDIDATES: &[GateCoverage] = &[GateCoverage {
    id: "cross-contamination",
    excludes: CONTAMINATION_EXCLUDES,
}];

pub const NOT_REQUIRED: [(&str, &str); 7] = [
    (
        "quick-speed-bench",
        "a single-user speed probe with no thresholds and no baseline — a MEASUREMENT tool, \
         deliberately never a gate: the required gates already cost hours per PR, and \
         its warm-path numbers (primed prefix cache + SSM snapshot) are not regression evidence",
    ),
    (
        "expert-fidelity",
        "an INSTRUMENT, not a verdict: it measures how far a --expert-category serve has \
         moved from the same model unrestricted. Both numbers come from one checkpoint in \
         two configurations, so there is no cross-model baseline a gate could compare \
         against — the value is ranking configurations, which is a decision a human makes",
    ),
    (
        "expert-categories",
        "a measurement tool that produces a CONFIG ARTIFACT, not a number: it maps prompt \
         categories to the experts they route to so --expert-category can load a subset. \
         There is nothing to gate on — the table describes the MODEL's routing, not the \
         change under test, and every MoE has a different expert space, so no committed \
         baseline could exist",
    ),
    (
        "bfcl-full",
        "the unsampled ~3600-sample draw; the two subset gates cover the same code at a \
         fraction of the GPU time, and a full run would dominate every PR",
    ),
    (
        "serve-matrix",
        "a multi-checkpoint survey used for release notes; it measures breadth, not regression",
    ),
    (
        "cross-contamination",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES) run on release cuts \
         and recorded as debt until it has proven itself; a fresh gate that fails on day one \
         would train people to override it",
    ),
    (
        "mlperf-agentic-subset",
        "not RUNNABLE yet: the official MLPerf Agentic Inference dataset is unpublished \
         upstream (mlcommons/endpoints@7935df4: \"MLCommons storage (link TBD)\"), so the leg \
         cannot be run, scored, or timed, and it refuses proxy datasets on purpose. Not a \
         promotion candidate either — a candidate accrues debt rows, and debt nobody can \
         discharge is worse than no row. Promote only after the dataset ships and a \
         calibration run sizes a <2 h draw",
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
    if is_test_only_rust_module(path) {
        return false;
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

/// Which [`PROMOTION_CANDIDATES`] these changed paths would have invalidated.
///
/// This is the DEBT a merge takes on: each id returned is a gate that wanted to
/// run, was not required to, and therefore did not. Rendering it is the whole
/// mechanism — an unrendered debt is indistinguishable from no debt.
pub fn promotion_debt<'a, I>(paths: I) -> Vec<&'static str>
where
    I: IntoIterator<Item = &'a str>,
{
    let paths: Vec<&str> = paths.into_iter().collect();
    PROMOTION_CANDIDATES
        .iter()
        .filter(|gate| paths.iter().any(|p| invalidates(gate, p)))
        .map(|gate| gate.id)
        .collect()
}
