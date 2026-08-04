// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=CUTLASS_HOME");
    println!("cargo:rerun-if-env-changed=FLASHINFER_HOME");
    println!("cargo:rerun-if-env-changed=ATLAS_CUDA_ARCH");
    // Register the `atlas_scale` cfg so `#[cfg(atlas_scale)]` does not trip
    // the `unexpected_cfgs` lint. `atlas_scale` selects SCALE/AMD (gfx1151)
    // codepaths over NVIDIA ones where the CUDA driver ABI differs — e.g.
    // SCALE's libcuda exports `cuGraphInstantiate` (not the NVIDIA-only
    // `cuGraphInstantiateWithFlags`). Driven by the same `ATLAS_TARGET_HW`
    // signal the atlas-kernels build uses; covers both the SCALE (`strix`)
    // and native-HIP (`strix-hip`) AMD targets.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    println!("cargo:rustc-check-cfg=cfg(atlas_cutlass)");
    println!("cargo:rustc-check-cfg=cfg(atlas_flashinfer)");
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
        // Even under ATLAS_SKIP_BUILD (CI no-GPU test build, no nvcc), the
        // cublaslt.rs FFI references cublasLt symbols that must resolve at LINK
        // time. cudarc emits -lcuda for us, but -lcublasLt is our own; emit it
        // here so the no-GPU `cargo test` build links (CI provides a stub
        // libcublasLt.so, same treatment as libcuda/libnccl).
        if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
            // -lcuda for the raw cu* driver FFI in cuda_backend/gpu_impl.rs
            // (cudarc is dlopen-mode here, so it doesn't emit it for us).
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cublasLt");
            // cudart: copy_d2d_2d_async uses cudaMemcpy2DAsync (a runtime, not
            // driver, symbol — CI's libcuda stub only has cu* driver symbols).
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
        }
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
    // cuBLASLt for the high-efficiency GEMM path (ATLAS_CUBLAS_GEMM=1). The
    // hand-written mma.sync projection/MoE GEMMs hit only ~30% of the cuBLAS
    // ceiling on GB10; cuBLASLt is a measured 2.7-4.8x lever on those shapes.
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // cudart: copy_d2d_2d_async uses cudaMemcpy2DAsync (a runtime, not driver,
    // symbol). Previously only emitted in the ATLAS_SKIP_BUILD stub path and
    // the optional CUTLASS/FlashInfer object paths below, so a real GPU build
    // without CUTLASS_HOME/FLASHINFER_HOME set fails to link with "undefined
    // reference to cudaMemcpy2DAsync" even though CI (which builds under
    // ATLAS_SKIP_BUILD) is green.
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");

    if let Some(cutlass_home) = std::env::var_os("CUTLASS_HOME") {
        build_cutlass_object(std::path::PathBuf::from(cutlass_home));
    }

    if let Some(fi_home) = std::env::var_os("FLASHINFER_HOME") {
        build_flashinfer_object(std::path::PathBuf::from(fi_home));
    }
}

/// Compile the FlashInfer ragged-prefill wrapper to a static lib (host-callable,
/// like the CUTLASS object). FlashInfer's FA2 (SM80-class) ragged-prefill kernel
/// codegens for sm_121f on GB10. Gated on `FLASHINFER_HOME`. Needs FlashInfer's
/// PINNED CCCL via `-isystem` ahead of the CUDA-13 toolkit CCCL (which lacks
/// `cuda::fast_mod_div`).
fn build_flashinfer_object(fi_home: std::path::PathBuf) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libatlas_flashinfer.a");
    let arch = std::env::var("ATLAS_CUDA_ARCH").unwrap_or_else(|_| "sm_121f".to_string());
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");

    let src = std::path::PathBuf::from("cuda/flashinfer_ragged_prefill.cu");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rustc-cfg=atlas_flashinfer");

    let cccl = fi_home.join("3rdparty/cccl");
    let obj = out_dir.join("flashinfer_ragged_prefill.o");
    let status = Command::new(&nvcc)
        .arg("-c")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg(format!("-arch={arch}"))
        // FlashInfer's pinned CCCL MUST precede the toolkit CCCL.
        .arg("-isystem")
        .arg(cccl.join("libcudacxx/include"))
        .arg("-isystem")
        .arg(cccl.join("cub"))
        .arg("-isystem")
        .arg(cccl.join("thrust"))
        .arg(format!("-I{}", fi_home.join("include").display()))
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to spawn nvcc for FlashInfer wrapper");
    assert!(
        status.success(),
        "nvcc failed while building FlashInfer wrapper"
    );

    let status = Command::new("ar")
        .arg("crus")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to spawn ar for FlashInfer wrapper");
    assert!(
        status.success(),
        "ar failed while archiving FlashInfer wrapper"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=atlas_flashinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn build_cutlass_object(cutlass_home: std::path::PathBuf) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libatlas_cutlass.a");
    let arch = std::env::var("ATLAS_CUDA_ARCH").unwrap_or_else(|_| "sm_121f".to_string());
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");

    let sources = [
        std::path::PathBuf::from("cuda/cutlass_bf16_gemm.cu"),
        std::path::PathBuf::from("cuda/cutlass_nvfp4_gemm.cu"),
        std::path::PathBuf::from("cuda/cutlass_nvfp4_grouped_gemm.cu"),
    ];
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
    }
    println!("cargo:rustc-cfg=atlas_cutlass");

    let mut objects = Vec::new();
    for src in &sources {
        let obj = out_dir.join(
            src.file_stem()
                .expect("CUTLASS wrapper source has a file stem")
                .to_string_lossy()
                .to_string()
                + ".o",
        );
        let status = Command::new(&nvcc)
            .arg("-c")
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--expt-relaxed-constexpr")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg(format!("-arch={arch}"))
            .arg(format!("-I{}", cutlass_home.join("include").display()))
            .arg(format!(
                "-I{}",
                cutlass_home.join("tools/util/include").display()
            ))
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to spawn nvcc for CUTLASS wrapper");
        assert!(
            status.success(),
            "nvcc failed while building CUTLASS wrapper {}",
            src.display()
        );
        objects.push(obj);
    }

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to spawn ar for CUTLASS wrapper");
    assert!(
        status.success(),
        "ar failed while archiving CUTLASS wrapper"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=atlas_cutlass");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
