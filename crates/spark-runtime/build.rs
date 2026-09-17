// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=AVAROK_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=AVAROK_TARGET_HW");
    println!("cargo:rerun-if-env-changed=CUTLASS_HOME");
    println!("cargo:rerun-if-env-changed=FLASHINFER_HOME");
    println!("cargo:rerun-if-env-changed=AVAROK_CUDA_ARCH");
    // Register the `avarok_scale` cfg so `#[cfg(avarok_scale)]` does not trip
    // the `unexpected_cfgs` lint. `avarok_scale` selects SCALE/AMD codepaths
    // over NVIDIA ones where the CUDA driver ABI differs, e.g. SCALE's libcuda
    // exports `cuGraphInstantiate` (not the NVIDIA-only
    // `cuGraphInstantiateWithFlags`).
    println!("cargo:rustc-check-cfg=cfg(avarok_scale)");
    println!("cargo:rustc-check-cfg=cfg(avarok_cutlass)");
    println!("cargo:rustc-check-cfg=cfg(avarok_flashinfer)");

    // Resolved HERE, before every early return: the SM architecture is a
    // property of the selected target, not of whether an optional reference
    // object happens to be compiled, and resolving it once is what stops the
    // CUTLASS and FlashInfer objects from disagreeing with each other or with
    // the PTX they are benchmarked against.
    let hardware_toml = hardware_toml_path();
    println!("cargo:rerun-if-changed={}", hardware_toml.display());
    let cuda_arch = resolve_cuda_arch(&hardware_toml);

    // Vendor, not name. This used to read
    // `AVAROK_TARGET_HW.starts_with("strix")`, which made the driver-ABI
    // choice a function of the directory name. It is a property of the
    // toolchain: `[hardware].vendor` is what avarok-kernels/build.rs already
    // switches `resolve_compute_target` on, so `amd` (SCALE) and `hip` (native
    // ROCm) are exactly the targets whose libcuda is not NVIDIA's. A second
    // SCALE target whose directory is not named "strix" (`kernels/r9700`,
    // gfx1201) would otherwise be built against the NVIDIA driver surface and
    // fail to link, or worse, call an entry point SCALE's libcuda does not
    // export.
    //
    // Same file `resolve_cuda_arch` below reads for `-arch=`, and its
    // `rerun-if-changed` is emitted once above for both. Behaviour is
    // unchanged for every existing target: strix declares `amd`, strix-hip
    // declares `hip`, gb10/hopper/b200 declare `nvidia`, metal `apple`, and an
    // unset AVAROK_TARGET_HW still resolves to gb10, no cfg either way. An
    // unreadable file leaves the cfg unset, as it did before.
    if matches!(
        hardware_vendor(&hardware_toml).as_deref(),
        Some("amd") | Some("hip")
    ) {
        println!("cargo:rustc-cfg=avarok_scale");
    }

    if matches!(
        std::env::var("AVAROK_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        // Even under AVAROK_SKIP_BUILD (CI no-GPU test build, no nvcc), the
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
    // AvarokCudaBackend is compiled in). The metal feature build on
    // Apple Silicon must not request -lcuda.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // Link libcuda for AvarokCudaBackend's raw CUDA driver API calls.
    // The actual CUDA driver is a stub at compile time; at runtime
    // it resolves to the NVIDIA driver installed on the system.
    println!("cargo:rustc-link-lib=dylib=cuda");
    // cuBLASLt for the high-efficiency GEMM path (AVAROK_CUBLAS_GEMM=1). The
    // hand-written mma.sync projection/MoE GEMMs hit only ~30% of the cuBLAS
    // ceiling on GB10; cuBLASLt is a measured 2.7-4.8x lever on those shapes.
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // cudart: copy_d2d_2d_async uses cudaMemcpy2DAsync (a runtime, not driver,
    // symbol). Previously only emitted in the AVAROK_SKIP_BUILD stub path and
    // the optional CUTLASS/FlashInfer object paths below, so a real GPU build
    // without CUTLASS_HOME/FLASHINFER_HOME set fails to link with "undefined
    // reference to cudaMemcpy2DAsync" even though CI (which builds under
    // AVAROK_SKIP_BUILD) is green.
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");

    // ★ Both objects below are REFERENCE IMPLEMENTATIONS we benchmark against
    // and intend to beat — never dependencies. Leaving `CUTLASS_HOME` /
    // `FLASHINFER_HOME` unset is the NORMAL build: it emits no third-party
    // kernel object, and the binary serves entirely on Atlas's own kernels.
    // Setting them only makes the opponent available behind its runtime
    // opt-in (`AVAROK_CUTLASS_GEMM=1` / `AVAROK_FLASHINFER_PREFILL=1`), which
    // stays OFF by default. Canonical rationale: the module docs on
    // `spark_runtime::cutlass` and `spark_runtime::flashinfer`.
    if let Some(cutlass_home) = std::env::var_os("CUTLASS_HOME") {
        build_cutlass_object(std::path::PathBuf::from(cutlass_home), &cuda_arch);
    }

    if let Some(fi_home) = std::env::var_os("FLASHINFER_HOME") {
        build_flashinfer_object(std::path::PathBuf::from(fi_home), &cuda_arch);
    }
}

/// Compile the FlashInfer ragged-prefill wrapper to a static lib (host-callable,
/// like the CUTLASS object). FlashInfer's FA2 (SM80-class) ragged-prefill kernel
/// codegens for sm_121f on GB10. Gated on `FLASHINFER_HOME`. Needs FlashInfer's
/// PINNED CCCL via `-isystem` ahead of the CUDA-13 toolkit CCCL (which lacks
/// `cuda::fast_mod_div`).
fn build_flashinfer_object(fi_home: std::path::PathBuf, arch: &str) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libavarok_flashinfer.a");
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = std::path::Path::new(&cuda_home).join("bin/nvcc");

    let src = std::path::PathBuf::from("cuda/flashinfer_ragged_prefill.cu");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rustc-cfg=avarok_flashinfer");

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
    println!("cargo:rustc-link-lib=static=avarok_flashinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn build_cutlass_object(cutlass_home: std::path::PathBuf, arch: &str) {
    use std::process::Command;

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let lib = out_dir.join("libavarok_cutlass.a");
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
    println!("cargo:rustc-cfg=avarok_cutlass");

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
    println!("cargo:rustc-link-lib=static=avarok_cutlass");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// The SM architecture the CUTLASS / FlashInfer reference objects compile for.
///
/// SSOT is `kernels/<hw>/HARDWARE.toml` `[hardware].arch` — the same file
/// `avarok-kernels/build.rs` compiles Atlas's own kernels from. Hard-coding
/// `sm_121f` here meant a build for any other hardware silently produced
/// reference objects for GB10, which is a benchmark comparing two different
/// GPUs' code. `AVAROK_CUDA_ARCH` still wins when set explicitly.
///
/// The literal survives only as the fallback for an unreadable file, announced
/// as a `cargo:warning` rather than applied silently.
fn resolve_cuda_arch(hardware_toml: &std::path::Path) -> String {
    const FALLBACK: &str = "sm_121f";
    if let Ok(explicit) = std::env::var("AVAROK_CUDA_ARCH") {
        return explicit;
    }
    match hardware_arch(hardware_toml) {
        Some(arch) => arch,
        None => {
            println!(
                "cargo:warning=spark-runtime: no [hardware].arch in {} — reference objects fall \
                 back to {FALLBACK}; set AVAROK_TARGET_HW to a target under kernels/, or \
                 AVAROK_CUDA_ARCH to override",
                hardware_toml.display()
            );
            FALLBACK.to_string()
        }
    }
}

/// `kernels/<AVAROK_TARGET_HW>/HARDWARE.toml`: the one hardware declaration
/// this build reads, for both `-arch=` and the vendor cfg above. Defaults to
/// gb10 when `AVAROK_TARGET_HW` is unset, exactly as the arch resolution did.
fn hardware_toml_path() -> std::path::PathBuf {
    let hw = std::env::var("AVAROK_TARGET_HW").unwrap_or_else(|_| "gb10".to_string());
    std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|crates| crates.parent())
        .expect("crates/<crate> sits two levels below the workspace root")
        .join("kernels")
        .join(&hw)
        .join("HARDWARE.toml")
}

/// `[hardware].arch` from a `HARDWARE.toml`, or `None` if it cannot be read.
fn hardware_arch(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: toml::Value = text.parse().ok()?;
    Some(doc.get("hardware")?.get("arch")?.as_str()?.to_string())
}

/// `[hardware].vendor` from a `HARDWARE.toml`, or `None` if it cannot be read.
///
/// The compiler-family key: `nvidia` | `apple` | `amd` | `hip`, the same
/// values `avarok-kernels/build_target.rs::resolve_compute_target` switches on.
fn hardware_vendor(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: toml::Value = text.parse().ok()?;
    Some(doc.get("hardware")?.get("vendor")?.as_str()?.to_string())
}
