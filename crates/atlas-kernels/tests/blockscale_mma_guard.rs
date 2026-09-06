// SPDX-License-Identifier: AGPL-3.0-only

//! Every warp-level block-scaled-FP4 instruction in the sources `kernels/hopper`
//! and `kernels/b200` compile sits inside `#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA`.
//!
//! ORACLE — two ptxas rejections, measured on Spark 1 with CUDA 13.0.88 and
//! recorded in `docs/campaigns/hopper-atlas-vs-vllm-2026-09/receipts/`:
//!
//! ```text
//! sm_90a : error : Instruction 'cvt with .e2m1x2' not supported on .target 'sm_90a'
//! sm_100a: error : Instruction 'mma with block scale' not supported on .target 'sm_100a'
//! ```
//!
//! One instruction pair, absent from both new architectures for DIFFERENT
//! reasons. Hopper has no NVFP4 datapath at all. Datacentre Blackwell has one,
//! but issues block-scaled MMA through `tcgen05`, not through the warp-level
//! `mma.sync ... .kind::mxf4nvf4.block_scale` that consumer/GB10 Blackwell
//! (sm_120/sm_121) provides — so `sm_100a` rejects the warp form even though
//! it is the newer chip. Neither architecture is a superset of the other, which
//! is why one define covers both and neither can be waved through by an arch
//! comparison.
//!
//! WHY A TEXT SCAN. The real check is `scripts/hopper_ptx_gate.sh`, and it
//! needs nvcc, which CI does not have. This is the GPU-free half: a new
//! block-scaled site added outside the guard is caught in the same `cargo test`
//! that CI already runs, months before anyone next compiles for sm_90a. It
//! cannot prove the kernels are correct and does not try.
//!
//! The two tokens matched are the two the errors name — `e2m1x2` (the packed
//! FP4 convert's operand type) and `mxf4nvf4` (the block-scaled MMA kind).
//! Both are matched ANYWHERE in a line, comments included: prose that
//! describes the path belongs inside the guard with the code it describes, and
//! a scanner that tried to tell an inline-asm string from a comment would be a
//! C preprocessor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The define that compiles the warp-block-scale path out.
const GUARD: &str = "ATLAS_NO_WARP_BLOCKSCALE_MMA";

/// Instruction tokens that only assemble on sm_120/sm_121. See the module
/// docs: these are the two spellings the ptxas errors name.
const BLOCKSCALE_TOKENS: &[&str] = &["e2m1x2", "mxf4nvf4"];

/// The hardware sets that declare `-D<GUARD>` and therefore must have every
/// such site behind it.
const GUARDED_HW: &[&str] = &["hopper", "b200"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

/// The `.cu` set one target compiles, by the rule `build.rs::collect_cu_files`
/// uses: `common/` is the base layer, the model's quant dir overrides it by
/// file STEM. Keyed by stem so a shadowed common file is not scanned — it is
/// not compiled.
fn sources(hw: &str, model: &str) -> BTreeMap<String, PathBuf> {
    let hw_dir = kernels_root().join(hw);
    let mut by_stem = BTreeMap::new();
    for dir in [hw_dir.join("common"), hw_dir.join(model).join("nvfp4")] {
        let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for path in entries.flatten().map(|e| e.path()) {
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("cu") && ext != Some("cuh") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            by_stem.insert(stem, path);
        }
    }
    assert!(!by_stem.is_empty(), "kernels/{hw}/{model}: no sources");
    by_stem
}

/// Model directories under `kernels/<hw>` — the rule `ATLAS_TARGET_MODEL=*`
/// expands by: a subdirectory carrying a MODEL.toml.
fn models(hw: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(kernels_root().join(hw))
        .unwrap_or_else(|e| panic!("kernels/{hw}: {e}"))
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    names.sort();
    names
}

/// Line numbers (1-based) in `text` that name a block-scaled instruction and
/// are NOT inside an `#ifndef <GUARD>` region.
///
/// Tracks the preprocessor conditional STACK rather than a boolean, so a plain
/// `#ifdef`/`#if` nested inside the guard cannot close it and a site inside an
/// unrelated conditional elsewhere in the file cannot be mistaken for guarded.
/// `#else` inverts the innermost entry: the else-branch of `#ifndef GUARD` is
/// what runs WHEN the define is set, so a block-scaled site there is exactly
/// as broken as one outside.
fn unguarded_sites(text: &str) -> Vec<usize> {
    let mut stack: Vec<bool> = Vec::new();
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim_start();
        let directive = t.strip_prefix('#').map(str::trim_start);
        match directive {
            Some(d) if d.starts_with("ifndef") => {
                let opens_guard = d
                    .strip_prefix("ifndef")
                    .map(|rest| rest.split_whitespace().next() == Some(GUARD))
                    .unwrap_or(false);
                stack.push(opens_guard);
                continue;
            }
            Some(d) if d.starts_with("ifdef") || d.starts_with("if") => {
                stack.push(false);
                continue;
            }
            Some(d) if d.starts_with("else") || d.starts_with("elif") => {
                if let Some(top) = stack.last_mut() {
                    *top = false;
                }
                continue;
            }
            Some(d) if d.starts_with("endif") => {
                stack.pop();
                continue;
            }
            _ => {}
        }
        if stack.iter().any(|g| *g) {
            continue;
        }
        if BLOCKSCALE_TOKENS.iter().any(|tok| line.contains(tok)) {
            out.push(i + 1);
        }
    }
    assert!(
        stack.is_empty(),
        "unbalanced preprocessor conditionals: {} still open at EOF",
        stack.len()
    );
    out
}

// ── the oracle, tested ──

/// The scanner must FAIL on an unguarded site, or `assert!(sites.is_empty())`
/// below passes just as happily against a scanner that always returns nothing.
#[test]
fn the_scanner_reports_a_site_outside_the_guard() {
    let src = "\
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
";
    assert_eq!(unguarded_sites(src), vec![1]);
}

/// …and on one in the ELSE branch, which is the branch that compiles when the
/// define IS set — the branch this whole mechanism exists to keep empty of
/// block-scaled instructions.
#[test]
fn the_scanner_reports_a_site_in_the_guards_else_branch() {
    let src = "\
#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#else
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#endif
";
    assert_eq!(unguarded_sites(src), vec![4]);
}

/// An unrelated conditional nested inside the guard does not close it.
#[test]
fn a_nested_conditional_does_not_close_the_guard() {
    let src = "\
#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA
#ifdef SOMETHING_ELSE
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#endif
";
    assert!(unguarded_sites(src).is_empty());
}

/// A different `#ifndef` is not the guard, however similar it looks.
#[test]
fn another_ifndef_is_not_the_guard() {
    let src = "\
#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA_V2
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
";
    assert_eq!(unguarded_sites(src), vec![2]);
}

// ── the real tree ──

/// Every source `kernels/hopper` and `kernels/b200` compile has its
/// block-scaled sites behind the guard. Reported with file and line, because
/// the fix is always "move that region inside the `#ifndef`".
#[test]
fn no_source_compiled_for_hopper_or_b200_leaves_a_blockscale_site_unguarded() {
    let mut faults: Vec<String> = Vec::new();
    for hw in GUARDED_HW {
        for model in models(hw) {
            for (stem, path) in sources(hw, &model) {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                for line in unguarded_sites(&text) {
                    faults.push(format!("{hw}/{model}: {stem}.cu:{line}"));
                }
            }
        }
    }
    faults.sort();
    faults.dedup();
    assert!(
        faults.is_empty(),
        "block-scaled FP4 instructions outside `#ifndef {GUARD}`. Neither \
         sm_90a nor sm_100a assembles these (see this file's oracle); the \
         region must be compiled out on those targets:\n  {}",
        faults.join("\n  ")
    );
}

/// The guard is not vacuous: the qwen3.6-35b-a3b MoE grouped GEMM — the one
/// kernel of 871 that failed both new architectures — really does still
/// CONTAIN block-scaled sites, inside the guard. A file that lost them
/// entirely would pass the test above while having deleted the GB10 fast path.
#[test]
fn the_qwen36_moe_gemm_still_carries_its_blockscale_path_inside_the_guard() {
    let path = kernels_root().join("gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        text.contains(&format!("#ifndef {GUARD}")),
        "{}: no guard at all",
        path.display()
    );
    let total = text
        .lines()
        .filter(|l| BLOCKSCALE_TOKENS.iter().any(|tok| l.contains(tok)))
        .count();
    assert!(
        total >= 10,
        "{}: only {total} block-scaled lines left — the GB10 W4A4 path looks \
         deleted rather than guarded",
        path.display()
    );
    assert!(
        unguarded_sites(&text).is_empty(),
        "{}: all {total} of them must be inside the guard",
        path.display()
    );
}
