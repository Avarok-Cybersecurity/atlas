// SPDX-License-Identifier: AGPL-3.0-only

//! This crate had NO build script, and that is why it needs one.
//!
//! Six of `spark-server`'s source files carry `#[cfg(feature = "cuda")]`, and
//! the target_os migration rewrites those to `#[cfg(avarok_cuda)]`. But
//! `rustc-check-cfg` does NOT cross crates: registering `avarok_cuda` in
//! spark-runtime's build script does nothing for this crate's compilation, so
//! without a build script here every rewritten site would trip
//! `unexpected_cfgs`, which `[workspace.lints.rust] warnings = "deny"` turns
//! into a hard error.
//!
//! `spark-storage/build.rs` carries the comment recording the last time this
//! bit, including the macOS-only ordering landmine where the registration sat
//! after an early return.

#[path = "../avarok-kernels/build_backend.rs"]
mod build_backend;

fn main() {
    // FIRST, before anything that could return early.
    build_backend::register_cfgs();
    if let Some(os) = build_backend::target_os_from_env() {
        build_backend::emit(std::env::var_os("CARGO_FEATURE_CUDA").is_some(), &os);
    }
}
