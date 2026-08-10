// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-parameter-arity pin: launcher arg packs are validated against the
//! COMPILED PTX signatures, per target, on CPU (CI has no GPU; PTX is baked
//! into the binary).
//!
//! Why this exists: `cuLaunchKernel`'s `void**` param form reads one host
//! word per COMPILED parameter — a launcher passing fewer args makes the
//! driver read past the end of the arg array. That is exactly how
//! `w4a16_gemm_t_m128_bf16_v2` shipped broken (8-arg launch of a 9-param
//! kernel: CUDA_ERROR_INVALID_VALUE or a host SIGSEGV depending on the
//! neighboring heap word) and no runtime API will ever catch it. Pinning the
//! whole launch family's arities here turns that class of drift into a CPU
//! test failure.
//!
//! When this test fails after adding a kernel param: update BOTH the launcher
//! in spark-model AND the pin here, in the same commit.

/// (module, kernel, expected .param count) for every kernel in the w4a16
/// m128/tile launch family, per target that ships it. A target that does not
/// ship a (module, kernel) pair is skipped — presence is the kernel audit's
/// job; ARITY of what is present is this test's job.
const PINS: &[(&str, &str, usize)] = &[
    ("w4a16", "w4a16_gemm", 8),
    ("w4a16", "w4a16_gemm_t", 9), // +ldb — EVERY target, see `expected_arity`
    ("w4a16", "w4a16_gemm_t_p3", 9),
    // The deep-K twins take NO stride. They are reached through the 9-arg
    // `w4a16_gemm_n128` launcher (dense_ffn's small-M arm), which is safe only
    // because the driver ignores the surplus argument AND the FFN twins are
    // built unpadded. Pinned at 8 so that growing one of them a stride without
    // giving its launcher a real `ldb` to pass fails here.
    ("w4a16", "w4a16_gemm_t_k64", 8),
    ("w4a16", "w4a16_gemm_t_k64_p3", 8),
    ("w4a16", "w4a16_gemm_t_k64_n64_p3", 8),
    ("w4a16", "w4a16_gemm_t_m128", 8),
    ("w4a16", "w4a16_gemm_t_m128_bf16", 8),
    ("w4a16", "w4a16_gemm_t_m128_bf16_v2", 9), // the ldb kernel — the shipped-bug case
    ("w4a16_v2", "w4a16_gemm_t_m128_v2", 8),
    ("w4a16_v3", "w4a16_gemm_t_m128_v3", 8),
    // Load-time transpose (quantized.rs GPU path) — 4-arg launch.
    ("transpose_u8", "transpose_u8", 4),
];

/// Targets whose copy of a kernel legitimately differs in arity from the
/// family pin.
///
/// ★ THERE ARE NONE, and the exception that used to live here was STALE and
/// WRONG. It returned 8 for `w4a16_gemm_t` / `_p3` on every non-27B target,
/// describing the world before the `ldb` port propagated. All 28 copies of
/// `w4a16_gemm_t` in the tree now compile 9 params (verified on `origin/main`
/// at 4e34a9e7), so on any REAL wildcard build this test asserted 9 == 8 and
/// failed. It went unnoticed because CI builds with `ATLAS_SKIP_BUILD=1`, which
/// makes the whole test vacuous — see the early return below.
///
/// Keep the hook: a target legitimately CAN diverge (`_p3` and
/// `_m128_bf16_v2` ship on the 27B only, and absence is skipped, not failed).
/// But record evidence before adding an arm — re-derive the arity from the
/// `.cu` tree, do not trust a remembered count.
fn expected_arity(_model: &str, _module: &str, _kernel: &str, family_pin: usize) -> usize {
    family_pin
}

/// Count `.param` declarations of a PTX `.entry` by name.
fn ptx_param_count(ptx: &str, kernel: &str) -> Option<usize> {
    // `.visible .entry <name>(` then `.param ...` lines until `)`.
    let needle = format!(".entry {kernel}(");
    let start = ptx.find(&needle)?;
    let body = &ptx[start..];
    let close = body.find(')')?;
    Some(body[..close].matches(".param").count())
}

#[test]
fn w4a16_launch_family_arity_pins() {
    // `ATLAS_SKIP_BUILD=1` (the CI environment, and any host without nvcc)
    // makes build.rs emit a STUB `target_ptx.rs` with no compiled kernels.
    // There is no PTX to read arities out of, so the pins are vacuous —
    // report that and stop, rather than failing a test the environment made
    // impossible. The `checked >= 4` floor below stays armed for every real
    // build, which is where drift can actually occur.
    if atlas_kernels::available_targets()
        .iter()
        .all(|s| s.modules.is_empty())
    {
        eprintln!("no compiled PTX in this binary (stub build) — arity pins skipped");
        return;
    }
    let mut checked = 0usize;
    for set in atlas_kernels::available_targets() {
        for (module, blob) in &set.modules {
            let Ok(ptx) = std::str::from_utf8(blob) else {
                continue; // binary object (SCALE/Metal) — NVIDIA-only test
            };
            for &(pin_module, kernel, family_pin) in PINS {
                if *module != pin_module {
                    continue;
                }
                if let Some(count) = ptx_param_count(ptx, kernel) {
                    let want = expected_arity(set.target.model, module, kernel, family_pin);
                    assert_eq!(
                        count, want,
                        "PTX arity drift: {}::{} on target {} has {} params, launcher family \
                         pins {} — update the launcher AND this pin together",
                        module, kernel, set.target.model, count, want
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked >= 4,
        "arity test checked only {checked} kernels — PTX sets missing? (wildcard build expected)"
    );
}
