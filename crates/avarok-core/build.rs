// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../avarok-kernels/build_backend.rs"]
mod build_backend;

fn main() {
    // FIRST, before any early return. `rustc-check-cfg` does not cross crates,
    // so each crate must register the names itself or `unexpected_cfgs` fires
    // on whichever target takes the early path -- the scar spark-storage's
    // build script already carries. The RULE lives in build_backend.rs so it
    // can be tested; see crates/avarok-kernels/tests/backend_resolution.rs.
    build_backend::register_cfgs();
    if let Some(os) = build_backend::target_os_from_env() {
        build_backend::emit(std::env::var_os("CARGO_FEATURE_CUDA").is_some(), &os);
    }
}
