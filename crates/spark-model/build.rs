// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    // `atlas_scale` mirrors spark-runtime's build.rs cfg: it marks the SCALE/
    // AMD (gfx1151) targets (`strix`, `strix-hip`). On these the GPU-visible
    // pool is a unified APU GTT (~60 GB) that cannot hold the FP8 source
    // checkpoint co-resident with its NVFP4 requant result, so the weight
    // loader frees each FP8 source tensor right after requant (see
    // `quantized_from_fp8`). NVIDIA targets leave the cfg unset and keep the
    // current resident-source behavior byte-for-byte.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    // `atlas_hip` is the strict subset of atlas_scale for the NATIVE-HIP target
    // (`strix-hip`, hipcc — not the SCALE PTX-recompile `strix`). HIP lacks the
    // FP8 *prefill* GEMM kernels (fp8_gemm*/w8a16* are inline-PTX, not yet
    // WMMA-ported), so the FP8→FP8 predequant-for-prefill path has no kernel
    // there; on atlas_hip we skip predequant and use the NVFP4 (w4a16 WMMA)
    // prefill instead. SCALE recompiles the PTX and keeps the FP8 prefill path.
    println!("cargo:rustc-check-cfg=cfg(atlas_hip)");
    // `atlas_gdn_aot`: the FlashInfer GDN AOT kernel (gdn_holo_0.o) is LINKED
    // into the crate instead of reached through dlopen(libatlasgdn.so). Set by
    // `build_gdn_aot` below when the gb10/aarch64 gate passes.
    println!("cargo:rustc-check-cfg=cfg(atlas_gdn_aot)");
    let hw = std::env::var("ATLAS_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=atlas_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=atlas_hip");
    }
    build_gdn_aot(&hw);
}

/// Link-time vendoring of the FlashInfer GDN AOT kernel (Track B, shape (c)).
///
/// Compiles the committed shim sources and archives them together with the
/// AOT-exported kernel object (`3rdparty_patches/gdn_aot/gdn_holo_0.o` — an
/// aarch64 host object embedding the sm_121a cubin plus the MLIR-generated
/// launch code) into `libatlas_gdn_aot.a`, statically linked into this crate.
/// `gdn_cute_rt_stub.cpp` supplies the 8 `_cuda*`/`_cu*` symbols the .o
/// imports, replacing the proprietary `libcute_dsl_runtime.so`.
///
/// Replaces the dlopen/dlsym/transmute path in `layers/ops/gdn_flashinfer.rs`
/// with real `extern "C"` prototypes (cfg `atlas_gdn_aot`): a signature drift
/// between shim and Rust now fails at link time instead of corrupting memory
/// at first prefill. The dlopen path remains as fallback on builds where this
/// gate does not pass. Runtime opt-in is unchanged: `ATLAS_GDN_FLASHINFER=1`.
///
/// Gate: gb10 target + aarch64-linux host + cuda feature + artifact present +
/// toolchain present. The .o is aarch64-only (there is no source form — the
/// kernel is CuTe-DSL Python; see 3rdparty_patches/gdn_aot/STATUS.md for the
/// pinned re-export recipe), so non-gb10 and cross builds skip silently. A
/// gb10 build with a broken toolchain fails LOUD — a silent skip here would
/// ship a gb10 binary without its GDN kernel and nobody would notice until a
/// perf regression.
fn build_gdn_aot(hw: &str) {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let gdn_dir = manifest_dir.join("../../3rdparty_patches/gdn_aot");
    let kernel_o = gdn_dir.join("gdn_holo_0.o");

    let arch_ok = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let cuda_feature = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    if hw != "gb10" || !arch_ok || !cuda_feature {
        return;
    }
    if !kernel_o.is_file() {
        // Reachable only on a checkout that stripped the committed artifact
        // (e.g. a filtered export). Loud, because the gate above says this
        // build WANTS the kernel.
        println!(
            "cargo:warning=gdn_aot: {} missing — building WITHOUT the linked FlashInfer GDN \
             kernel (dlopen fallback only)",
            kernel_o.display()
        );
        return;
    }

    for f in [
        "gdn_holo_0.o",
        "gdn_holo_0.h",
        "gdn_shim.cpp",
        "gdn_transpose.cu",
        "gdn_cute_rt_stub.cpp",
    ] {
        println!("cargo:rerun-if-changed={}", gdn_dir.join(f).display());
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    let cuda_home = std::path::PathBuf::from(cuda_home);
    let nvcc_path = cuda_home.join("bin/nvcc");
    let nvcc = if nvcc_path.is_file() {
        nvcc_path.display().to_string()
    } else {
        "nvcc".to_string() // PATH fallback
    };
    let cuda_include = cuda_home.join("include");

    let run = |what: &str, cmd: &mut std::process::Command| {
        let out = cmd
            .output()
            .unwrap_or_else(|e| panic!("gdn_aot: failed to spawn {what}: {e}"));
        if !out.status.success() {
            panic!(
                "gdn_aot: {what} failed ({}):\n{}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    };

    // k<->v state transpose + device-side cu_seqlens write (plain CUDA C++).
    run(
        "nvcc gdn_transpose.cu",
        std::process::Command::new(&nvcc).args([
            "-arch=sm_121a",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-c",
            gdn_dir.join("gdn_transpose.cu").to_str().unwrap(),
            "-o",
            out_dir.join("gdn_transpose.o").to_str().unwrap(),
        ]),
    );
    // C-ABI shim over the AOT header + the cute-runtime replacement stub.
    for (src, obj) in [
        ("gdn_shim.cpp", "gdn_shim.o"),
        ("gdn_cute_rt_stub.cpp", "gdn_cute_rt_stub.o"),
    ] {
        run(
            &format!("g++ {src}"),
            std::process::Command::new("g++").args([
                "-O2",
                "-fPIC",
                "-std=c++17",
                "-I",
                gdn_dir.to_str().unwrap(),
                "-I",
                cuda_include.to_str().unwrap(),
                "-c",
                gdn_dir.join(src).to_str().unwrap(),
                "-o",
                out_dir.join(obj).to_str().unwrap(),
            ]),
        );
    }
    let archive = out_dir.join("libatlas_gdn_aot.a");
    let _ = std::fs::remove_file(&archive); // `ar r` appends into a stale archive
    run(
        "ar rcs libatlas_gdn_aot.a",
        std::process::Command::new("ar").args([
            "rcs",
            archive.to_str().unwrap(),
            out_dir.join("gdn_shim.o").to_str().unwrap(),
            out_dir.join("gdn_cute_rt_stub.o").to_str().unwrap(),
            out_dir.join("gdn_transpose.o").to_str().unwrap(),
            kernel_o.to_str().unwrap(),
        ]),
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=atlas_gdn_aot");
    // cudart STATIC, so the deployed binary stays self-contained: no
    // libcudart.so.13 and no libcute_dsl_runtime.so to ship next to it. The
    // driver (libcuda) is still dlopen'd at runtime — see gdn_cute_rt_stub.cpp.
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=culibos");
    println!("cargo:rustc-cfg=atlas_gdn_aot");
}
