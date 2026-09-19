// SPDX-License-Identifier: AGPL-3.0-only

//! Which GPU backend a build selects, resolved once and testable.
//!
//! The owner's requirement is that Metal support be target_os-gated so a plain
//! `cargo build` on macOS needs no flags. Cargo cannot make a *feature*
//! target-conditional, so the selection has to move into build-script cfgs —
//! the same shape `spark-runtime/build.rs` already uses for `avarok_scale`,
//! `avarok_cutlass` and `avarok_flashinfer`.
//!
//! This file holds only the RULE, as a pure function, because a rule that lives
//! inside `fn main()` of a build script is a rule nothing can test. Each build
//! script includes it with `#[path]` and calls [`resolve`]; the table below is
//! asserted in `tests/backend_resolution.rs`.
//!
//! Why a build-script cfg and not a feature: on macOS today a plain
//! `cargo build -p spark-runtime` FAILS — `default = ["cuda"]` pulls in
//! `cudarc`, whose build script shells out to `nvcc --version` and panics with
//! `No such file or directory`. Measured on the target box, 2026-09-18.
//!
//! ★ The cfgs must be REGISTERED before any early return. `spark-storage`'s
//! build script carries the scar: a `rustc-check-cfg` line placed after an
//! early return left `unexpected_cfgs` firing on macOS only.

/// The two backend cfgs a build should set, given the `cuda` feature's state
/// and the target OS.
///
/// * `avarok_cuda` — the `cuda` feature is on AND the target is not macOS.
///   The second half is what stops a macOS build from activating the 130
///   `cfg(feature = "cuda")` blocks while `cudarc` is absent from its
///   dependency graph.
/// * `avarok_metal` — the target IS macOS. Unconditional on that target: the
///   point of the exercise is that no flag is required.
///
/// Deliberately NOT mutually exclusive in the signature. A future target could
/// want both (or neither, on a CPU-only build), and a bool pair says that
/// plainly where an enum would have to be widened.
pub fn resolve(feature_cuda: bool, target_os: &str) -> (bool, bool) {
    let is_macos = target_os == "macos";
    (feature_cuda && !is_macos, is_macos)
}

/// `CARGO_CFG_TARGET_OS` is set for every build script; an absent value means
/// the build script is being run by something other than cargo, and guessing
/// "linux" there would silently produce a CUDA build on a Mac.
pub fn target_os_from_env() -> Option<String> {
    std::env::var("CARGO_CFG_TARGET_OS").ok()
}

/// Print the `rustc-check-cfg` registrations. Call this FIRST in `main()`,
/// before any early return, or `unexpected_cfgs` fires on whichever target
/// takes the early path.
pub fn register_cfgs() {
    println!("cargo:rustc-check-cfg=cfg(avarok_cuda)");
    println!("cargo:rustc-check-cfg=cfg(avarok_metal)");
}

/// Emit the selection. Returns what it emitted, so a caller can log it.
pub fn emit(feature_cuda: bool, target_os: &str) -> (bool, bool) {
    let (cuda, metal) = resolve(feature_cuda, target_os);
    if cuda {
        println!("cargo:rustc-cfg=avarok_cuda");
    }
    if metal {
        println!("cargo:rustc-cfg=avarok_metal");
    }
    (cuda, metal)
}
