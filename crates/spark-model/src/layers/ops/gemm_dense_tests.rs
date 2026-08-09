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
/// ★ THE ORIGINAL COUNT HERE WAS WRONG. It claimed 19-26 stale copies, from a
/// scan whose `awk` range terminated before the end of the signature and so
/// read a truncated block. The true figure was 6, of which 4 are now ported;
/// the 3 that remain are all `common/`, whose kernel body is structurally
/// different (zero matches for the load pattern) and needs reading, not
/// scripting. Regenerated with a checker that bounds the signature properly.
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
        // EMPTY: every copy now takes `ldb`. A new one that does not will fail
        // the `newly` assertion below — which is the whole point of the guard.
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
