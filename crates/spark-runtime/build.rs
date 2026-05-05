// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    if matches!(
        std::env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        return;
    }

    // Link libcuda for AtlasCudaBackend's raw CUDA driver API calls.
    // The actual CUDA driver is a stub at compile time; at runtime
    // it resolves to the NVIDIA driver installed on the system.
    println!("cargo:rustc-link-lib=dylib=cuda");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
}
