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
/// ★ There are none left. This used to carve out `w4a16_gemm_t`/`_p3` for
/// every target except the 27B, on the grounds that "only the 27B grew
/// `ldb`". That stopped being true when #429 finished the port: `ldb` is
/// now on **all 28 `w4a16_gemm.cu` paths** (8 distinct files — `strix/common`
/// is a symlink into `gb10/common`, and several model dirs symlink their
/// neighbours), and `w4a16_gemm_t_ldb_drift_is_exactly_the_known_set` pins
/// that with an EMPTY known-drift list.
///
/// Leaving the carve-out behind made this test red on `main` for every
/// non-27B target — deepseek-v4-flash reported "9 params vs pin 8", which is
/// the FIXED kernel being measured against the pre-fix expectation. The
/// stale side was the exception, not the kernels.
///
/// The function is kept (rather than deleted) because it is the designated
/// place for a future legitimate per-target divergence, and because deleting
/// it would scatter that decision back into the call site. Before adding an
/// arm, record evidence — re-derive the arity from the `.cu` tree, do not
/// trust a remembered count.
fn expected_arity(model: &str, module: &str, kernel: &str, family_pin: usize) -> usize {
    let _ = (model, module, kernel);
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
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
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
