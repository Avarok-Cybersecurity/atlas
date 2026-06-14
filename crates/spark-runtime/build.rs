// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    // Register the `atlas_scale` cfg so `#[cfg(atlas_scale)]` does not trip
    // the `unexpected_cfgs` lint. `atlas_scale` selects SCALE/AMD (gfx1151)
    // codepaths over NVIDIA ones where the CUDA driver ABI differs — e.g.
    // SCALE's libcuda exports `cuGraphInstantiate` (not the NVIDIA-only
    // `cuGraphInstantiateWithFlags`). Driven by the same `ATLAS_TARGET_HW`
    // signal the atlas-kernels build uses; covers both the SCALE (`strix`)
    // and native-HIP (`strix-hip`) AMD targets.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    if std::env::var("ATLAS_TARGET_HW")
        .as_deref()
        .map(|hw| hw.starts_with("strix"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=atlas_scale");
    }

    if matches!(
        std::env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        return;
    }

    // libcuda is only needed when the cuda feature is on (i.e. when
    // AtlasCudaBackend is compiled in). The metal feature build on
    // Apple Silicon must not request -lcuda.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
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
