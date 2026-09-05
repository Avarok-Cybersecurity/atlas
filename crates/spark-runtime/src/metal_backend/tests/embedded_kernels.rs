// SPDX-License-Identifier: AGPL-3.0-only
//! The anti-vacuity guard for the whole `metal_backend` parity suite.
//!
//! Every other test in this suite calls `helpers::maybe_backend()`, which
//! hands `atlas_kernels::metallib_modules()` to `MetalGpuBackend::new`. That
//! constructor loads one `MTLLibrary` per entry -- and an EMPTY slice is not
//! an error to it: it succeeds with a zero-entry library cache, and the
//! failure only surfaces later, once per test, as
//! `Metal: unknown module '<name>'`.
//!
//! That is exactly what shipped. atlas-kernels' `build.rs` honours
//! `ATLAS_SKIP_BUILD` before anything else and emits a stub whose
//! `metallib_modules()` is `Vec::new()`; `ci.yml` sets `ATLAS_SKIP_BUILD: "1"`
//! at WORKFLOW level so the ubuntu jobs can type-check without nvcc, and the
//! macOS job inherited it. The required context
//! `cargo test --features metal (macOS aarch64)` was therefore red on every
//! non-web PR, reporting on the stub rather than on the kernels -- a verdict
//! that was not about the thing it claimed to test.
//!
//! This test states the suite's precondition as a first-class assertion, so
//! the answer to "did the kernels get built?" is ONE named failure instead of
//! 35 lookup failures, and so the suite can never again be run against a build
//! that embedded nothing. It deliberately has no skip path: a skip here would
//! reintroduce the hole it exists to close.

/// The parity suite is meaningless without embedded metallibs. Refuse.
#[test]
fn embedded_metallib_set_is_not_empty() {
    let modules = atlas_kernels::metallib_modules();
    assert!(
        !modules.is_empty(),
        "atlas_kernels::metallib_modules() is empty -- no Metal kernels were \
         compiled into this binary, so every parity test below would fail with \
         `Metal: unknown module`, and a suite that cannot reach a kernel cannot \
         report on one.\n\
         \n\
         Cause: atlas-kernels/build.rs took its skip branch. Either \
         ATLAS_SKIP_BUILD is set to 1/true, or the build is on macOS with no \
         ATLAS_TARGET_HW set (the auto-skip).\n\
         \n\
         Build the kernels instead of skipping them:\n\
         \x20 ATLAS_SKIP_BUILD=0 ATLAS_TARGET_HW=metal \\\n\
         \x20 ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 ATLAS_TARGET_QUANT=mlx_int8 \\\n\
         \x20 cargo test -p spark-runtime --no-default-features --features metal metal_backend"
    );
}
