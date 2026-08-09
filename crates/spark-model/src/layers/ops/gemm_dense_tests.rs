// SPDX-License-Identifier: AGPL-3.0-only

//! Source-contract tests for the dense GEMM launchers.
//!
//! Sibling of `gemm_dense.rs` per the house `#[path]` idiom — and because
//! inlining them pushed that file past the repo's 500-line cap.

// ── w4a16_gemm_t `ldb` drift ───────────────────────────────────────────────

/// ★ Every Rust launcher passes NINE args (`w4a16_gemm_n128_ldb`). A kernel
/// copy that kept the 8-arg signature does not fail to launch — the driver
/// IGNORES the extra argument — so B is strided by `N` instead of `ldb`,
/// silently, wherever the two differ.
///
/// That is not hypothetical. On `qwen3.6-35b-a3b` the batched MTP propose
/// passes `N = --mtp-vocab` (default 100000) with `ldb` = the padded lm_head
/// twin stride (248320) once 5+ sequences are in flight, so 2032 of 2048
/// K-positions read the wrong rows and the drafter emits garbage. It does not
/// fault, because both strides are 16-aligned and in-bounds.
///
/// The fix landed on `qwen3.6-27b` on 2026-07-28 and was never propagated:
/// shadow dirs whole-file-replace `common/`, so a fix in one copy is invisible
/// to the other 28. This is the `shadowed_kernels_null` pattern again.
///
/// ★ This test PINS THE DEBT rather than asserting it is zero. A new kernel
/// copy without `ldb` fails here; PORTING one requires deleting its line, which
/// is the direction we want to be easy. Deleting the last line and the list is
/// the end state.
#[test]
fn w4a16_gemm_t_ldb_drift_is_exactly_the_known_set() {
    fn visit(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                visit(&p, out);
            } else if p.file_name().is_some_and(|f| f == "w4a16_gemm.cu") {
                out.push(p);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut files = Vec::new();
    visit(&root, &mut files);
    assert!(
        files.len() > 20,
        "kernel tree not found at {}",
        root.display()
    );

    let known: std::collections::BTreeSet<&str> = [
        "kernels/gb10/common/w4a16_gemm.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/gemma-4-26b-a4b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/gemma-4-31b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/mistral-small-4/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/nemotron-3-nano-30b-a3b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/nemotron-super-120b-a12b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3.5-122b-a10b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3.5-27b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3.5-35b-a3b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3.5-397b-a17b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3-next-80b-a3b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/qwen3-vl-30b-a3b/nvfp4/w4a16_gemm.cu",
        "kernels/gb10/step3p7-flash/nvfp4/w4a16_gemm.cu",
        "kernels/strix/common/w4a16_gemm.cu",
        "kernels/strix-hip/common/w4a16_gemm.cu",
        "kernels/strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu",
    ]
    .into_iter()
    .collect();

    let mut stale = std::collections::BTreeSet::new();
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        let Some(i) = src.find("__global__ void w4a16_gemm_t(") else {
            continue;
        };
        let sig = &src[i..i + src[i..].find(')').unwrap_or(0)];
        if !sig.contains("ldb") {
            let rel = p.to_string_lossy();
            let rel = rel.split("kernels/").nth(1).unwrap_or(&rel).to_string();
            stale.insert(format!("kernels/{rel}"));
        }
    }
    let stale: std::collections::BTreeSet<&str> = stale.iter().map(String::as_str).collect();

    let newly: Vec<&&str> = stale.difference(&known).collect();
    assert!(
        newly.is_empty(),
        "NEW w4a16_gemm_t copies without `ldb` — they will stride B by N: {newly:#?}"
    );
    let fixed: Vec<&&str> = known.difference(&stale).collect();
    assert!(
        fixed.is_empty(),
        "these were ported — delete them from the pinned list: {fixed:#?}"
    );
}
