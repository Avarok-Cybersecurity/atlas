// SPDX-License-Identifier: AGPL-3.0-only

//! Source-contract tests for the dense GEMM launchers.
//!
//! Sibling of `gemm_dense.rs` per the house `#[path]` idiom — and because
//! inlining them pushed that file past the repo's 500-line cap.
//!
//! These are SOURCE tests. They read `kernels/**.cu` and `gemm_dense.rs` as
//! text and pin the launcher/kernel contract on CPU. They compile nothing and
//! run nothing on a GPU — `cargo test` runs with `ATLAS_SKIP_BUILD=1`, so the
//! PTX-level sibling (`atlas-kernels/tests/kernel_arity.rs`) is VACUOUS in CI
//! and these are the only automatic guard the contract has there.

#[path = "gemm_dense_tests_util.rs"]
mod util;

use util::{cu_files, indexed_exprs, kernel_sig_body, multiplies_by_bare_n};

/// Every kernel whose COMPILED signature carries the 9th `ldb` parameter — the
/// ROW STRIDE of the transposed `B_packed[K/2, N]`, which may EXCEED `N`.
///
/// `cuLaunchKernel`'s `void**` param form reads one host word per COMPILED
/// parameter. A copy that kept the 8-arg signature does not fail to launch:
/// the driver simply IGNORES the ninth argument and the kernel strides B by
/// `N`. Silent sheared rows, no fault, no error. The reverse mismatch (9-param
/// kernel, 8-arg launcher) reads one-past-the-end of the arg array —
/// `CUDA_ERROR_INVALID_VALUE` or a host SIGSEGV depending on the neighbouring
/// heap word.
///
/// The motivating case is the padded lm_head twin
/// (`transpose_concat_for_gemm_padded`, `impl_a1.rs`): vocab 248077 is ODD, so
/// the twin is built at `align_up(vocab, 128) = 248320` and every launcher that
/// can see it passes that stride. On `qwen3.6-35b-a3b` the batched MTP propose
/// additionally narrows `N` to `--mtp-vocab`, so N and ldb differ by ~148k.
const LDB_KERNELS: &[&str] = &[
    "w4a16_gemm_t",
    "w4a16_gemm_t_p3",
    "w4a16_gemm_t_m128_bf16_v2",
];

/// ★ PORT STATUS, verified against `origin/main` at 4e34a9e7 on 2026-08-10.
///
/// The debt is ZERO: all 28 copies of `w4a16_gemm_t`, both copies of
/// `w4a16_gemm_t_p3` and both of `w4a16_gemm_t_m128_bf16_v2` declare `ldb` AND
/// use it (see `ldb_kernels_actually_use_the_parameter` below — declaring it is
/// not the same as using it, and only the body test can tell them apart).
///
/// ★ The doc that stood here claimed "the 3 that remain are all `common/`".
/// That was TRUE when written and is now STALE. Those three — `gb10/common/`,
/// `strix/common/`, `strix-hip/common/` — are the SCALAR dialect (single-byte B
/// loads, no 16-byte `cp.async`), which is why a scripted port matched zero of
/// them; they were ported by hand and are green. The same doc also said "6, of
/// which 4 are now ported; the 3 that remain", which does not add up: do not
/// trust a remembered count, re-derive it from the tree.
///
/// This test PINS THE DEBT rather than asserting zero, so that a NEW copy
/// without `ldb` fails here. `known` being empty is the end state.
#[test]
fn w4a16_gemm_t_ldb_drift_is_exactly_the_known_set() {
    let files = cu_files();

    let known: std::collections::BTreeSet<&str> = [
        // EMPTY: every copy now takes `ldb`. A new one that does not will fail
        // the `newly` assertion below — which is the whole point of the guard.
    ]
    .into_iter()
    .collect();

    let mut stale = std::collections::BTreeSet::new();
    let mut seen = 0usize;
    for p in &files {
        let src = std::fs::read_to_string(p).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, _)) = kernel_sig_body(&src, name) else {
                continue;
            };
            seen += 1;
            if !sig.contains("ldb") {
                stale.insert(format!("{}::{name}", util::rel(p)));
            }
        }
    }
    assert!(
        seen > 20,
        "only {seen} ldb-family kernels found — tree moved?"
    );
    let stale: std::collections::BTreeSet<&str> = stale.iter().map(String::as_str).collect();

    let newly: Vec<&&str> = stale.difference(&known).collect();
    assert!(
        newly.is_empty(),
        "NEW ldb-family copies without `ldb` — they will stride B by N: {newly:#?}"
    );
    let fixed: Vec<&&str> = known.difference(&stale).collect();
    assert!(
        fixed.is_empty(),
        "these were ported — delete them from the pinned list: {fixed:#?}"
    );
}

/// A SIGNATURE-ONLY port passes the drift test above and is still broken: the
/// parameter is declared, the body still writes `... * N + gn`, and B is strided
/// by N exactly as before. Shadow dirs whole-file-replace `common/`, so this is
/// the realistic way a copy regresses.
///
/// Both operands matter. `B_packed` and `B_scale` are SEPARATE allocations that
/// share the twin's row pitch (`transpose_impl` allocates `stride * half_k` and
/// `stride * num_groups`), so striding one by LDB and the other by N yields
/// correctly-addressed nibbles scaled by the wrong block — subtly wrong logits
/// rather than obviously wrong ones.
#[test]
fn ldb_kernels_actually_use_the_parameter() {
    let mut checked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(p).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, body)) = kernel_sig_body(&src, name) else {
                continue;
            };
            if !sig.contains("ldb") {
                continue; // reported by the drift test; do not double-fail
            }
            let where_ = format!("{}::{name}", util::rel(p));
            let code = util::strip_line_comments(&body);
            assert!(
                code.contains("const unsigned int LDB = ldb;"),
                "{where_} must derive the body stride LDB from the launcher-supplied ldb"
            );
            for arr in ["B_packed", "B_scale"] {
                let idx = indexed_exprs(&body, arr);
                assert!(
                    !idx.is_empty(),
                    "{where_}: no `{arr}[...]` index found — the extractor or the \
                     kernel changed shape; this guard must be re-read, not deleted"
                );
                for e in &idx {
                    assert!(
                        e.contains("LDB"),
                        "{where_}: `{arr}[{e}]` does not stride by LDB"
                    );
                    assert!(
                        !multiplies_by_bare_n(e),
                        "{where_}: `{arr}[{e}]` still multiplies by N — a padded \
                         twin (ldb > N) will read sheared rows, silently"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 40, "only {checked} operand indexes checked");
}

/// ★ The two dialects diverge here, and getting it backwards is a real bug in
/// each direction.
///
/// SCALAR (`*/common/w4a16_gemm.cu`): B loads are single bytes, so there is no
/// 16-byte alignment constraint and the COLUMN guard stays `gn < N`. `N` is the
/// valid-column bound; the stride is what changed. Widening it to `gn < LDB`
/// would "helpfully" compute the pad columns — reading zero-filled padding into
/// real output columns is only harmless because the store guard is also `N`, and
/// relying on that is one edit away from garbage.
///
/// TILE (every per-target copy): B loads are 16-byte `cp.async` chunks, so the
/// LOAD bound must be `LDB`, not `N` — a chunk straddling `N` is still inside
/// the padded row and must be fetched, and the predicate has to say so.
///
/// ALIGNMENT: `cp.async.cg ... 16` requires a 16-byte-aligned source. Row `r`
/// sits at `r * LDB`, so 16-byte alignment of every row requires `LDB % 16 == 0`
/// — which `ldb` DELIVERS rather than breaks: the unpadded stride is the vocab
/// (248077, odd) and misaligns 15 of every 16 rows, the campaign's CUDA 716.
/// The invariant that keeps it true is pinned by
/// `padded_twin_stride_is_16_byte_aligned` below.
#[test]
fn ldb_kernels_keep_their_dialect_specific_bounds() {
    let (mut scalar, mut tile) = (0usize, 0usize);
    // Of the scalar paths, how many are REAL files rather than symlinks into
    // another hardware set's `common/`. That is the number the alarm below is
    // actually about: a symlinked backend inherits the port for free, a forked
    // one has to be edited by hand.
    let mut scalar_forked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(p).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, body)) = kernel_sig_body(&src, name) else {
                continue;
            };
            if !sig.contains("ldb") {
                continue;
            }
            let where_ = format!("{}::{name}", util::rel(p));
            // ★ Classify by PATH, never by the guard text. Keying the dialect
            // off `gn < N` would make the widening mutation below reclassify
            // the kernel as a tile copy and PASS — the assertion has to be
            // independent of the property it is testing.
            if util::rel(p).contains("/common/") {
                scalar += 1;
                if std::fs::symlink_metadata(p)
                    .map(|m| !m.file_type().is_symlink())
                    .unwrap_or(true)
                {
                    scalar_forked += 1;
                }
                assert!(
                    body.contains("gn < N"),
                    "{where_}: the scalar column guard `gn < N` is gone. N is the \
                     valid-COLUMN bound; only the STRIDE became LDB."
                );
                assert!(
                    !body.contains("gn < LDB"),
                    "{where_}: the scalar column guard was widened from N to LDB. \
                     N is the valid-COLUMN bound; only the STRIDE changed."
                );
            } else {
                tile += 1;
                assert!(
                    body.contains("< LDB"),
                    "{where_}: tile dialect but no load bound by LDB — 16-byte \
                     cp.async chunks past N are inside the padded row and must \
                     still be fetched"
                );
            }
        }
    }
    // ★ 5 PATHS, 2 FILES. `kernels/{strix,hopper,b200}/common/w4a16_gemm.cu`
    // are all SYMLINKS to `../../gb10/common/w4a16_gemm.cu`, so those copies
    // were ported the moment gb10's was. `strix-hip/common/` is a real,
    // separate HIP file and had to be done by hand.
    //
    // BOTH numbers are asserted, and the second is the one that matters. Path
    // count is what the build walks, so it has to track the tree — but a new
    // symlinked hardware set costs nothing, while a new FORKED `common/` copy
    // is a file somebody has to port by hand and is exactly what this test
    // exists to catch. Asserting only the total would fire on the free case
    // and, once bumped, would go quiet on the expensive one.
    assert_eq!(
        scalar, 5,
        "the shared `common/` scalar paths moved (gb10, strix, strix-hip, \
         hopper, b200) — update this count with the tree"
    );
    assert_eq!(
        scalar_forked, 2,
        "expected exactly 2 FORKED `common/` scalar copies (gb10 and \
         strix-hip); a 3rd is a new backend that needs the same hand port, \
         where a symlinked one would have inherited it"
    );
    assert!(tile > 20, "only {tile} tile copies found — tree moved?");
}

/// The launcher and the kernel must agree on ARITY in both directions. This is
/// the CPU-side stand-in for `atlas-kernels/tests/kernel_arity.rs`, which reads
/// the real PTX and is vacuous under `ATLAS_SKIP_BUILD=1`.
#[test]
fn ldb_launcher_arg_count_matches_kernel_param_count() {
    let launchers = include_str!("gemm_dense.rs");
    for fname in ["w4a16_gemm_n128_ldb", "w4a16_gemm_n128_m128_bf16_ldb"] {
        let args = util::launcher_arg_count(launchers, fname)
            .unwrap_or_else(|| panic!("launcher `{fname}` not found in gemm_dense.rs"));
        assert_eq!(
            args, 9,
            "`{fname}` packs {args} kernel args; the ldb family compiles 9 params"
        );
    }

    let mut checked = 0usize;
    for p in &cu_files() {
        let src = std::fs::read_to_string(p).unwrap();
        for name in LDB_KERNELS {
            let Some((sig, _)) = kernel_sig_body(&src, name) else {
                continue;
            };
            let n = util::param_count(&sig);
            assert_eq!(
                n,
                9,
                "{}::{name} compiles {n} params; the launcher packs 9",
                util::rel(p)
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} kernels checked");
}

/// The 8-arg convenience wrapper must forward `n` AS THE STRIDE — that is the
/// entire definition of the packed (unpadded) case. Forwarding anything else,
/// or dropping to a genuinely 8-arg launch, reintroduces the bug for every
/// non-lm_head caller at once.
#[test]
fn non_ldb_wrapper_forwards_n_as_the_stride() {
    let src = include_str!("gemm_dense.rs");
    let body = util::fn_body(src, "w4a16_gemm_n128")
        .expect("`w4a16_gemm_n128` wrapper not found in gemm_dense.rs");
    let call = body
        .lines()
        .find(|l| l.contains("w4a16_gemm_n128_ldb("))
        .unwrap_or_else(|| panic!("wrapper no longer delegates to the _ldb launcher:\n{body}"));
    let args: Vec<String> = util::call_args(call)
        .into_iter()
        .map(|s| s.trim().to_string())
        .collect();
    assert_eq!(
        args.len(),
        10,
        "unexpected delegation shape `{call}` (9 kernel args + stream)"
    );
    assert_eq!(
        args[8], "n",
        "the wrapper forwards `{}` as ldb, not `n` — the packed case is DEFINED \
         by rows being exactly N apart",
        args[8]
    );
}

/// The tile kernels' 16-byte `cp.async` B loads are only legal because the
/// padded twin's row pitch is 16-byte aligned. That is not a property of `ldb`
/// the parameter — it is a property of the ALIGN argument at the single call
/// site that builds the twin. Pin it there.
#[test]
fn padded_twin_stride_is_16_byte_aligned() {
    let src = include_str!("../../model/impl_a1.rs");
    let mut sites = 0usize;
    let mut from = 0usize;
    while let Some(rel) = src[from..].find("transpose_concat_for_gemm_padded(") {
        let at = from + rel;
        let args = util::call_args(&src[at..]);
        from = at + 1;
        sites += 1;
        let align = args
            .last()
            .map(|s| s.trim().trim_end_matches(',').to_string())
            .unwrap_or_default();
        let n: usize = align.parse().unwrap_or_else(|_| {
            panic!(
                "`transpose_concat_for_gemm_padded` align is `{align}`, not a literal. \
                 A non-literal align cannot be checked here and the 16-byte cp.async \
                 invariant it carries is load-bearing — extend this guard, do not delete it."
            )
        });
        assert!(
            n.is_multiple_of(16),
            "padded twin align={n} is not a multiple of 16: rows land at r*align, \
             so the tile kernels' 16-byte cp.async B loads will fault (CUDA 716)"
        );
    }
    assert_eq!(
        sites, 1,
        "expected exactly one padded-twin construction site in impl_a1.rs"
    );
}
