// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../avarok-kernels/build_backend.rs"]
mod build_backend;

fn main() {
    // FIRST, before any early return: `rustc-check-cfg` does not cross crates.
    // The rule lives in build_backend.rs so it can be tested; see
    // crates/avarok-kernels/tests/backend_resolution.rs.
    build_backend::register_cfgs();
    if let Some(os) = build_backend::target_os_from_env() {
        build_backend::emit(std::env::var_os("CARGO_FEATURE_CUDA").is_some(), &os);
    }
    println!("cargo:rerun-if-env-changed=AVAROK_SKIP_BUILD");
    if matches!(
        std::env::var("AVAROK_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        return;
    }

    // Link libcuda for raw CUDA driver API calls in kernel benchmarks.
    println!("cargo:rustc-link-lib=dylib=cuda");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
}
